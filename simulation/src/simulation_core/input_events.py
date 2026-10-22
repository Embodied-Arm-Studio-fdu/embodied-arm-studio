"""Strict recording-only keyboard/mouse input contract and adapters."""

import asyncio
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import math
import re
from threading import RLock
from typing import Mapping, Optional


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_KEY = re.compile(r"[A-Za-z0-9]|Arrow(?:Up|Down|Left|Right)")
_COMMON_FIELDS = frozenset({"event_id", "run_id", "task_id", "type", "occurred_at"})
_TYPE_FIELDS = {
    "KEYBOARD": frozenset({"key", "action"}),
    "MOUSE_MOVE": frozenset({"x", "y"}),
    "MOUSE_BUTTON": frozenset({"button", "action", "x", "y"}),
}


class InputEventType(str, Enum):
    KEYBOARD = "KEYBOARD"
    MOUSE_MOVE = "MOUSE_MOVE"
    MOUSE_BUTTON = "MOUSE_BUTTON"


@dataclass(frozen=True)
class InputEvent:
    event_id: str
    run_id: str
    task_id: str
    type: InputEventType
    occurred_at: datetime
    key: Optional[str] = None
    action: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    button: Optional[str] = None

    def __post_init__(self):
        for name in ("event_id", "run_id", "task_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise ValueError("invalid %s" % name)
        try:
            object.__setattr__(self, "type", InputEventType(self.type))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid input event type") from exc
        if (
            not isinstance(self.occurred_at, datetime)
            or self.occurred_at.tzinfo is None
            or self.occurred_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("occurred_at must be timezone-aware UTC")
        if self.type == InputEventType.KEYBOARD:
            if not isinstance(self.key, str) or not _KEY.fullmatch(self.key):
                raise ValueError("invalid keyboard key")
            if self.action not in ("pressed", "released"):
                raise ValueError("invalid keyboard action")
            if any(value is not None for value in (self.x, self.y, self.button)):
                raise ValueError("keyboard event contains mouse fields")
        elif self.type == InputEventType.MOUSE_MOVE:
            self._coordinates()
            if any(value is not None for value in (self.key, self.action, self.button)):
                raise ValueError("mouse move contains unrelated fields")
        else:
            self._coordinates()
            if self.button not in ("left", "middle", "right"):
                raise ValueError("invalid mouse button")
            if self.action not in ("pressed", "released"):
                raise ValueError("invalid mouse button action")
            if self.key is not None:
                raise ValueError("mouse button contains key")

    def _coordinates(self):
        for value in (self.x, self.y):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError("mouse coordinates must be normalized to 0..1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]):
        if not isinstance(values, Mapping):
            raise ValueError("input must be an object")
        kind = values.get("type")
        if not isinstance(kind, str) or kind not in _TYPE_FIELDS:
            raise ValueError("invalid input event type")
        allowed = _COMMON_FIELDS | _TYPE_FIELDS[kind]
        if set(values) - allowed:
            raise ValueError("input event contains unknown fields")
        missing = _COMMON_FIELDS - set(values)
        if missing:
            raise ValueError("input event is missing required fields")
        timestamp = values["occurred_at"]
        if not isinstance(timestamp, str):
            raise ValueError("occurred_at must be an ISO 8601 string")
        try:
            timestamp_value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid occurred_at") from exc
        optional = {
            name: values[name]
            for name in ("key", "action", "x", "y", "button")
            if name in values
        }
        return cls(
            event_id=values["event_id"],
            run_id=values["run_id"],
            task_id=values["task_id"],
            type=kind,
            occurred_at=timestamp_value,
            **optional
        )

    @classmethod
    def from_json(cls, message):
        try:
            return cls.from_mapping(json.loads(message))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("invalid input event") from exc


class InputEventConflict(ValueError):
    """The retained event id is already associated with different content."""


@dataclass(frozen=True)
class InputRecordResult:
    accepted: bool
    duplicate: bool


class RecordingInputEventSink:
    """Thread-safe, bounded, in-process recording grouped by task id."""

    def __init__(self, max_events_per_task=10000):
        if (
            isinstance(max_events_per_task, bool)
            or not isinstance(max_events_per_task, int)
            or max_events_per_task <= 0
        ):
            raise ValueError("max_events_per_task must be a positive integer")
        self.max_events_per_task = max_events_per_task
        self._events_by_task = {}
        self._lock = RLock()

    def record(self, event):
        if not isinstance(event, InputEvent):
            raise TypeError("event must be InputEvent")
        with self._lock:
            retained = self._events_by_task.setdefault(event.task_id, OrderedDict())
            existing = retained.get(event.event_id)
            if existing is not None:
                if existing == event:
                    return InputRecordResult(accepted=True, duplicate=True)
                raise InputEventConflict("event id conflicts with retained content")
            retained[event.event_id] = event
            while len(retained) > self.max_events_per_task:
                retained.popitem(last=False)
            return InputRecordResult(accepted=True, duplicate=False)

    def events_for_task(self, task_id):
        with self._lock:
            return deepcopy(list(self._events_by_task.get(task_id, {}).values()))

    @property
    def events(self):
        """Compatibility view used by the loopback-only test adapter."""
        with self._lock:
            return deepcopy(
                [event for retained in self._events_by_task.values() for event in retained.values()]
            )


class MockInputWebSocketServer:
    """Loopback-only test server. Events are recorded, never executed."""

    def __init__(self, sink, host="127.0.0.1", port=0):
        if host != "127.0.0.1":
            raise ValueError("mock input server must bind 127.0.0.1")
        self.sink, self.host, self.port = sink, host, int(port)
        self.errors = []
        self._server = None

    async def start(self):
        import websockets

        self._server = await websockets.serve(self.handle, self.host, self.port)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def handle(self, websocket):
        async for message in websocket:
            try:
                event = InputEvent.from_json(message)
                result = self.sink.record(event)
                await websocket.send(
                    json.dumps(
                        {
                            "ok": True,
                            "event_id": event.event_id,
                            "duplicate": result.duplicate,
                        }
                    )
                )
            except Exception as exc:
                self.errors.append(str(exc))
                await websocket.send(json.dumps({"ok": False, "error": str(exc)}))

    async def close(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


class MockInputWebSocketClient:
    def __init__(self, url):
        self.url = url

    async def send(self, values):
        import websockets

        async with websockets.connect(self.url) as websocket:
            replies = []
            for value in values:
                await websocket.send(value if isinstance(value, str) else json.dumps(value))
                replies.append(json.loads(await websocket.recv()))
            return replies
