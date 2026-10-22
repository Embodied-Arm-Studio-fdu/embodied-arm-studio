"""Pure-data contracts for simulation-to-backend reporting."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse
from uuid import uuid4


SCHEMA_VERSION = "1.0"
TASK_STATUSES = frozenset({
    "queued", "initializing", "running", "finalizing", "completed",
    "stopping", "stopped", "failed",
})


class EventType(str, Enum):
    STATE = "state"
    METRIC = "metric"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class TerminalStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_event_id() -> str:
    return "evt_%s" % uuid4().hex


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value


def _validate_time(value: str, name: str) -> datetime:
    _required_text(value, name)
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("%s must be an ISO 8601 timestamp" % name) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("%s must use UTC" % name)
    return parsed


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must contain only JSON-compatible values" % name) from exc


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _validate_metrics(metrics: Mapping[str, Any], name: str) -> None:
    for key, value in metrics.items():
        _required_text(key, "%s key" % name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("%s.%s must be numeric" % (name, key))
        if not math.isfinite(value):
            raise ValueError("%s.%s must be finite" % (name, key))


def _validate_error(payload: Mapping[str, Any], name: str = "error") -> None:
    _required_text(payload.get("code"), "%s.code" % name)
    _required_text(payload.get("message"), "%s.message" % name)
    if not isinstance(payload.get("retryable"), bool):
        raise ValueError("%s.retryable must be a boolean" % name)


def _validate_event_payload(event_type: EventType, payload: Mapping[str, Any]) -> None:
    if event_type == EventType.STATE:
        previous = _required_text(payload.get("previous_status"), "payload.previous_status")
        status = _required_text(payload.get("status"), "payload.status")
        if previous not in TASK_STATUSES or status not in TASK_STATUSES:
            raise ValueError("state payload must use business task statuses")
    elif event_type == EventType.METRIC:
        _non_negative_int(payload.get("step"), "payload.step")
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("payload.metrics must be a non-empty object")
        allowed = {
            "update_index", "average_reward", "policy_loss", "value_loss", "entropy",
            "episode_length", "episodes", "approx_kl", "clip_fraction",
            "explained_variance",
        }
        unknown = set(metrics).difference(allowed)
        if unknown:
            raise ValueError("unsupported metric fields: %s" % ", ".join(sorted(unknown)))
        _positive_int(metrics.get("update_index"), "payload.metrics.update_index")
        for key, value in metrics.items():
            if key == "update_index":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("payload.metrics.%s must be numeric" % key)
    elif event_type == EventType.ERROR:
        _validate_error(payload, "payload")
    elif event_type == EventType.HEARTBEAT:
        status = _required_text(payload.get("status"), "payload.status")
        if status not in TASK_STATUSES:
            raise ValueError("heartbeat status must be a business task status")
        _non_negative_int(payload.get("training_steps"), "payload.training_steps")


@dataclass(frozen=True)
class SimulationEvent:
    event_id: str
    sequence: int
    type: EventType
    run_id: str
    task_id: str
    occurred_at: str
    payload: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.event_id, "event_id")
        _required_text(self.run_id, "run_id")
        _required_text(self.task_id, "task_id")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("schema_version must be %s" % SCHEMA_VERSION)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        try:
            event_type = EventType(self.type)
        except ValueError as exc:
            raise ValueError("unsupported event type: %s" % self.type) from exc
        _validate_time(self.occurred_at, "occurred_at")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be an object")
        payload = _json_copy(dict(self.payload), "payload")
        _validate_event_payload(event_type, payload)
        object.__setattr__(self, "type", event_type)
        object.__setattr__(self, "payload", payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id, "sequence": self.sequence,
            "type": self.type.value, "run_id": self.run_id,
            "task_id": self.task_id, "occurred_at": self.occurred_at,
            "schema_version": self.schema_version, "payload": dict(self.payload),
        }


def _validate_artifact_reference(value: Any, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object or null" % name)
    if value.get("path") is None and value.get("uri") is None:
        raise ValueError("%s must contain path or uri" % name)
    for key in ("path", "uri"):
        reference = value.get(key)
        if reference is None:
            continue
        _required_text(reference, "%s.%s" % (name, key))
        if PurePosixPath(reference).is_absolute() or PureWindowsPath(reference).is_absolute():
            raise ValueError("%s.%s must not be a host absolute path" % (name, key))
        parsed = urlparse(reference)
        if parsed.scheme:
            raise ValueError("%s.%s must be Run-relative" % (name, key))
        if ".." in PurePosixPath(reference).parts:
            raise ValueError("%s.%s must not escape the Run root" % (name, key))


@dataclass(frozen=True)
class TerminalCallback:
    event_id: str
    sequence: int
    run_id: str
    task_id: str
    status: TerminalStatus
    started_at: str
    ended_at: str
    training_steps: int
    update_steps: int
    stop_reason: Optional[str]
    metrics: Mapping[str, Any]
    model: Optional[Mapping[str, Any]]
    artifacts: Sequence[Mapping[str, Any]]
    error: Optional[Mapping[str, Any]]
    evaluation: Optional[Mapping[str, Any]] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.event_id, "event_id")
        _required_text(self.run_id, "run_id")
        _required_text(self.task_id, "task_id")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("schema_version must be %s" % SCHEMA_VERSION)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        try:
            status = TerminalStatus(self.status)
        except ValueError as exc:
            raise ValueError("unsupported terminal status: %s" % self.status) from exc
        started = _validate_time(self.started_at, "started_at")
        ended = _validate_time(self.ended_at, "ended_at")
        if ended < started:
            raise ValueError("ended_at must not precede started_at")
        _non_negative_int(self.training_steps, "training_steps")
        _non_negative_int(self.update_steps, "update_steps")
        metrics = _json_copy(dict(self.metrics), "metrics")
        _validate_metrics(metrics, "metrics")
        model = _json_copy(dict(self.model), "model") if self.model is not None else None
        artifacts = tuple(_json_copy(dict(item), "artifacts") for item in self.artifacts)
        error = _json_copy(dict(self.error), "error") if self.error is not None else None
        evaluation = (
            _json_copy(dict(self.evaluation), "evaluation")
            if self.evaluation is not None else None
        )
        _validate_artifact_reference(model, "model")
        for index, artifact in enumerate(artifacts):
            _validate_artifact_reference(artifact, "artifacts[%d]" % index)
        artifact_types = [
            _required_text(item.get("type"), "artifacts[%d].type" % index)
            for index, item in enumerate(artifacts)
        ]
        if len(artifact_types) != len(set(artifact_types)):
            raise ValueError("artifacts must use each type at most once")
        if status == TerminalStatus.COMPLETED:
            if error is not None:
                raise ValueError("completed callback error must be null")
            # Training completions retain their original model contract.  A
            # PPO-free demo has zero training/update steps and intentionally
            # completes without fabricating a checkpoint reference.
            if model is None and (self.training_steps > 0 or self.update_steps > 0):
                raise ValueError("completed callback must reference a model")
            if not any(item.get("type") == "manifest" for item in artifacts):
                raise ValueError("completed callback must reference the manifest artifact")
        elif status == TerminalStatus.FAILED:
            if error is None:
                raise ValueError("failed callback requires error")
            _validate_error(error)
        elif status == TerminalStatus.STOPPED:
            _required_text(self.stop_reason, "stop_reason")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "evaluation", evaluation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version, "event_id": self.event_id,
            "sequence": self.sequence, "run_id": self.run_id,
            "task_id": self.task_id, "status": self.status.value,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "training_steps": self.training_steps, "update_steps": self.update_steps,
            "stop_reason": self.stop_reason, "metrics": dict(self.metrics),
            "model": dict(self.model) if self.model is not None else None,
            "artifacts": [dict(item) for item in self.artifacts],
            "error": dict(self.error) if self.error is not None else None,
            "evaluation": (
                dict(self.evaluation) if self.evaluation is not None else None
            ),
        }


class RunSequenceGenerator:
    """Thread-safe, process-local sequence space, starting at one per Run."""

    def __init__(self) -> None:
        self._last_by_run: Dict[str, int] = {}
        self._lock = threading.Lock()

    def next(self, run_id: str) -> int:
        _required_text(run_id, "run_id")
        with self._lock:
            value = self._last_by_run.get(run_id, 0) + 1
            self._last_by_run[run_id] = value
            return value


class RedisRunSequenceGenerator:
    """Cross-process Run sequence using Redis' atomic INCR operation."""

    def __init__(self, client, namespace: str = "simulation") -> None:
        if not namespace or any(character.isspace() for character in namespace):
            raise ValueError("Redis namespace must be non-empty without whitespace")
        self._client = client
        self.namespace = namespace

    @classmethod
    def from_url(cls, redis_url: str, namespace: str = "simulation"):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Python package 'redis' is required for Redis sequence") from exc
        try:
            client = redis.Redis.from_url(redis_url, decode_responses=False)
            client.ping()
        except Exception as exc:
            raise RuntimeError("cannot connect to Redis for event sequence: %s" % exc) from exc
        return cls(client, namespace)

    def next(self, run_id: str) -> int:
        _required_text(run_id, "run_id")
        try:
            return int(self._client.incr(self._key(run_id)))
        except Exception as exc:
            raise RuntimeError("Redis event sequence INCR failed: %s" % exc) from exc

    def _key(self, run_id: str) -> str:
        return "%s:run:%s:event_sequence" % (self.namespace, run_id)


class BackendMessageFactory:
    """Allocates IDs and the shared event/callback sequence for new messages."""

    def __init__(self, sequence_generator=None, event_id_factory: Callable[[], str] = None):
        self._sequences = sequence_generator or RunSequenceGenerator()
        self._event_id_factory = event_id_factory or new_event_id

    def event(self, event_type, run_id, task_id, payload, occurred_at=None):
        return SimulationEvent(
            event_id=self._event_id_factory(), sequence=self._sequences.next(run_id),
            type=event_type, run_id=run_id, task_id=task_id,
            occurred_at=occurred_at or utc_now_text(), payload=payload,
        )

    def terminal_callback(self, run_id, task_id, **fields):
        return TerminalCallback(
            event_id=self._event_id_factory(), sequence=self._sequences.next(run_id),
            run_id=run_id, task_id=task_id, **fields
        )
