"""Persistent terminal-callback outbox and bounded retry dispatcher."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import threading
from typing import Dict, Optional

from .backend_events import TerminalCallback


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    EXHAUSTED = "exhausted"


def _utc_now():
    return datetime.now(timezone.utc)


def _time_text(value):
    return value.isoformat() if value is not None else None


def _parse_time(value):
    return datetime.fromisoformat(value) if value is not None else None


def callback_from_dict(values):
    return TerminalCallback(**dict(values))


@dataclass(frozen=True)
class CallbackOutboxRecord:
    callback: TerminalCallback
    delivery_status: DeliveryStatus
    attempt_count: int
    next_attempt_at: Optional[datetime]
    last_error: Optional[str]
    created_at: datetime
    delivered_at: Optional[datetime] = None

    @property
    def event_id(self):
        return self.callback.event_id

    @property
    def run_id(self):
        return self.callback.run_id

    @property
    def task_id(self):
        return self.callback.task_id

    def to_dict(self):
        return {
            "event_id": self.event_id, "run_id": self.run_id,
            "task_id": self.task_id, "callback": self.callback.to_dict(),
            "delivery_status": self.delivery_status.value,
            "attempt_count": self.attempt_count,
            "next_attempt_at": _time_text(self.next_attempt_at),
            "last_error": self.last_error,
            "created_at": _time_text(self.created_at),
            "delivered_at": _time_text(self.delivered_at),
        }

    @classmethod
    def from_dict(cls, values):
        return cls(
            callback=callback_from_dict(values["callback"]),
            delivery_status=DeliveryStatus(values["delivery_status"]),
            attempt_count=int(values["attempt_count"]),
            next_attempt_at=_parse_time(values.get("next_attempt_at")),
            last_error=values.get("last_error"),
            created_at=_parse_time(values["created_at"]),
            delivered_at=_parse_time(values.get("delivered_at")),
        )


class CallbackOutbox(ABC):
    @abstractmethod
    def put_if_absent(self, callback, now=None):
        raise NotImplementedError

    @abstractmethod
    def get(self, event_id):
        raise NotImplementedError

    @abstractmethod
    def get_by_task_id(self, task_id):
        raise NotImplementedError

    @abstractmethod
    def due(self, now=None):
        raise NotImplementedError

    @abstractmethod
    def save(self, record):
        raise NotImplementedError


class InMemoryCallbackOutbox(CallbackOutbox):
    def __init__(self):
        self._records: Dict[str, CallbackOutboxRecord] = {}
        self._event_by_task: Dict[str, str] = {}
        self._lock = threading.RLock()

    def put_if_absent(self, callback, now=None):
        with self._lock:
            existing = self.get_by_task_id(callback.task_id)
            if existing is not None:
                return existing
            created = now or _utc_now()
            record = CallbackOutboxRecord(
                callback, DeliveryStatus.PENDING, 0, created, None, created
            )
            self._records[callback.event_id] = record
            self._event_by_task[callback.task_id] = callback.event_id
            return record

    def get(self, event_id):
        with self._lock:
            return self._records.get(event_id)

    def get_by_task_id(self, task_id):
        with self._lock:
            event_id = self._event_by_task.get(task_id)
            return self._records.get(event_id) if event_id else None

    def due(self, now=None):
        current = now or _utc_now()
        with self._lock:
            return tuple(
                item for item in self._records.values()
                if item.delivery_status == DeliveryStatus.PENDING
                and item.next_attempt_at is not None
                and item.next_attempt_at <= current
            )

    def save(self, record):
        with self._lock:
            if record.event_id not in self._records:
                raise KeyError("outbox record does not exist: %s" % record.event_id)
            self._records[record.event_id] = record


class RedisCallbackOutbox(CallbackOutbox):
    """JSON-only Redis outbox; task index makes terminal creation idempotent."""

    def __init__(self, client, namespace="simulation"):
        self._client = client
        self.namespace = namespace

    def put_if_absent(self, callback, now=None):
        existing = self.get_by_task_id(callback.task_id)
        if existing is not None:
            return existing
        created = now or _utc_now()
        record = CallbackOutboxRecord(
            callback, DeliveryStatus.PENDING, 0, created, None, created
        )
        script = """
        local existing = redis.call('GET', KEYS[1])
        if existing then return existing end
        redis.call('SET', KEYS[1], ARGV[1])
        redis.call('SET', KEYS[2], ARGV[2])
        redis.call('SADD', KEYS[3], ARGV[1])
        return ARGV[1]
        """
        selected = self._client.eval(
            script, 3, self._task_key(callback.task_id),
            self._record_key(callback.event_id), self._index_key(),
            callback.event_id, self._encode(record),
        )
        if isinstance(selected, bytes):
            selected = selected.decode()
        return record if str(selected) == callback.event_id else self.get(str(selected))

    def get(self, event_id):
        value = self._client.get(self._record_key(event_id))
        return self._decode(value) if value is not None else None

    def get_by_task_id(self, task_id):
        event_id = self._client.get(self._task_key(task_id))
        if event_id is None:
            return None
        if isinstance(event_id, bytes):
            event_id = event_id.decode()
        return self.get(str(event_id))

    def due(self, now=None):
        current = now or _utc_now()
        event_ids = self._client.smembers(self._index_key())
        records = []
        for event_id in event_ids:
            if isinstance(event_id, bytes):
                event_id = event_id.decode()
            record = self.get(str(event_id))
            if (
                record is not None
                and record.delivery_status == DeliveryStatus.PENDING
                and record.next_attempt_at is not None
                and record.next_attempt_at <= current
            ):
                records.append(record)
        return tuple(records)

    def save(self, record):
        self._client.set(self._record_key(record.event_id), self._encode(record))

    def _encode(self, record):
        return json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True)

    def _decode(self, value):
        if isinstance(value, bytes):
            value = value.decode()
        return CallbackOutboxRecord.from_dict(json.loads(value))

    def _record_key(self, event_id):
        return "%s:callback_outbox:event:%s" % (self.namespace, event_id)

    def _task_key(self, task_id):
        return "%s:callback_outbox:task:%s" % (self.namespace, task_id)

    def _index_key(self):
        return "%s:callback_outbox:index" % self.namespace


class CallbackDispatcher:
    def __init__(
        self, outbox, backend, max_attempts=5, base_delay_seconds=1.0,
        clock=None,
    ):
        if max_attempts < 1 or base_delay_seconds < 0:
            raise ValueError("retry configuration is invalid")
        self.outbox = outbox
        self.backend = backend
        self.max_attempts = int(max_attempts)
        self.base_delay_seconds = float(base_delay_seconds)
        self.clock = clock or _utc_now

    def dispatch_once(self):
        now = self.clock()
        for record in self.outbox.due(now):
            self.dispatch_event(record.event_id, now=now)

    def dispatch_event(self, event_id, now=None):
        record = self.outbox.get(event_id)
        if record is None or record.delivery_status != DeliveryStatus.PENDING:
            return record
        current = now or self.clock()
        if record.next_attempt_at is not None and record.next_attempt_at > current:
            return record
        attempts = record.attempt_count + 1
        try:
            self.backend.publish_terminal_callback(record.callback)
        except Exception as exc:
            if not getattr(exc, "retryable", True) or attempts >= self.max_attempts:
                updated = replace(
                    record, delivery_status=DeliveryStatus.EXHAUSTED,
                    attempt_count=attempts, next_attempt_at=None,
                    last_error=str(exc),
                )
            else:
                delay = self.base_delay_seconds * (2 ** (attempts - 1))
                updated = replace(
                    record, attempt_count=attempts,
                    next_attempt_at=current + timedelta(seconds=delay),
                    last_error=str(exc),
                )
        else:
            updated = replace(
                record, delivery_status=DeliveryStatus.DELIVERED,
                attempt_count=attempts, next_attempt_at=None, last_error=None,
                delivered_at=current,
            )
        self.outbox.save(updated)
        return updated

    def confirm_controlled_recovery(self, event_id, callback, now=None):
        """Mark an exhausted delivery recovered after Backend service acceptance."""
        record = self.outbox.get(event_id)
        if record is None or record.delivery_status != DeliveryStatus.EXHAUSTED:
            raise ValueError("controlled recovery requires an exhausted outbox record")
        if (
            callback.event_id != record.event_id
            or callback.run_id != record.run_id
            or callback.task_id != record.task_id
            or callback.status != record.callback.status
        ):
            raise ValueError("recovered Callback identity conflicts with outbox record")
        original = record.callback.to_dict()
        recovered = callback.to_dict()
        original.pop("evaluation", None)
        recovered.pop("evaluation", None)
        if original != recovered:
            raise ValueError("controlled recovery may only restore evaluation evidence")
        updated = replace(
            record,
            callback=callback,
            delivery_status=DeliveryStatus.DELIVERED,
            next_attempt_at=None,
            last_error=None,
            delivered_at=now or self.clock(),
        )
        self.outbox.save(updated)
        return updated
