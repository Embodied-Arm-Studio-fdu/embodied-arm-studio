"""Stream lifecycle boundary layered over the Phase 6.4 video runtime."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import math
import re
import threading
import uuid
from typing import Callable, Dict, Optional, Protocol, Tuple
import json

from .domain import SimulationTask, TaskStatus


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BUSINESS_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class StreamSessionStatus(str, Enum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    READY = "READY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


_TRANSITIONS = {
    StreamSessionStatus.CREATED: frozenset({StreamSessionStatus.STARTING, StreamSessionStatus.STOPPING}),
    StreamSessionStatus.STARTING: frozenset({StreamSessionStatus.READY, StreamSessionStatus.ERROR, StreamSessionStatus.STOPPING}),
    StreamSessionStatus.READY: frozenset({StreamSessionStatus.ERROR, StreamSessionStatus.STOPPING}),
    StreamSessionStatus.ERROR: frozenset({StreamSessionStatus.STOPPING, StreamSessionStatus.STOPPED}),
    StreamSessionStatus.STOPPING: frozenset({StreamSessionStatus.STOPPED}),
    StreamSessionStatus.STOPPED: frozenset(),
}


class InvalidStreamTransition(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc)


def _required_id(name, value):
    if not isinstance(value, str) or not _BUSINESS_ID.fullmatch(value):
        raise ValueError("%s must be a safe non-empty identifier" % name)


def new_stream_id() -> str:
    """Opaque ID: uniqueness does not depend on unsafe platform identifiers."""
    return "stream-" + uuid.uuid4().hex


@dataclass(frozen=True)
class StreamSession:
    stream_id: str
    run_id: str
    task_id: str
    camera_id: str
    status: StreamSessionStatus
    rtsp_internal_url: str
    webrtc_playback_url: str
    whep_url: str
    width: int
    height: int
    fps: float
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    client_connected: bool = False
    video_error: Optional[str] = None
    reconnect_count: int = 0
    last_client_connected_at: Optional[datetime] = None
    last_client_disconnected_at: Optional[datetime] = None

    def __post_init__(self):
        if not isinstance(self.stream_id, str) or not _ID.fullmatch(self.stream_id):
            raise ValueError("stream_id must be a URL-safe single path segment")
        for name in ("run_id", "task_id", "camera_id"):
            _required_id(name, getattr(self, name))
        object.__setattr__(self, "status", StreamSessionStatus(self.status))
        for name in ("rtsp_internal_url", "webrtc_playback_url", "whep_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError("%s is required" % name)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (self.width, self.height)):
            raise ValueError("width and height must be positive integers")
        if isinstance(self.fps, bool) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        for name in ("created_at", "started_at", "stopped_at", "last_client_connected_at", "last_client_disconnected_at"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, datetime) or value.tzinfo is None):
                raise ValueError("%s must be timezone-aware" % name)
        if isinstance(self.reconnect_count, bool) or self.reconnect_count < 0:
            raise ValueError("reconnect_count must be non-negative")

    def transition_to(self, target, at=None):
        target = StreamSessionStatus(target)
        if target not in _TRANSITIONS[self.status]:
            raise InvalidStreamTransition("invalid stream transition: %s -> %s" % (self.status.value, target.value))
        values = {"status": target}
        if target == StreamSessionStatus.READY and self.started_at is None:
            values["started_at"] = at or _now()
        if target == StreamSessionStatus.STOPPED:
            values["stopped_at"] = at or _now()
            values["client_connected"] = False
        return replace(self, **values)

    def to_dict(self):
        return {
            "stream_id": self.stream_id, "run_id": self.run_id, "task_id": self.task_id,
            "camera_id": self.camera_id, "status": self.status.value,
            "rtsp_internal_url": self.rtsp_internal_url,
            "webrtc_playback_url": self.webrtc_playback_url, "whep_url": self.whep_url,
            "width": self.width, "height": self.height, "fps": self.fps,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "client_connected": self.client_connected, "video_error": self.video_error,
            "reconnect_count": self.reconnect_count,
        }


class StreamRuntime(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


class StreamSessionManager:
    """Thread-safe owner of serializable sessions and separate runtime handles."""
    def __init__(self, runtime_factory: Callable[[StreamSession], StreamRuntime],
                 rtsp_base="rtsp://127.0.0.1:18554", webrtc_base="http://127.0.0.1:18889",
                 session_observer=None):
        self._runtime_factory = runtime_factory
        self._rtsp_base = rtsp_base.rstrip("/")
        self._webrtc_base = webrtc_base.rstrip("/")
        self._sessions: Dict[str, StreamSession] = {}
        self._runtime: Dict[str, StreamRuntime] = {}
        self._by_run_camera: Dict[Tuple[str, str], str] = {}
        self._session_observer = session_observer
        self._lock = threading.RLock()

    def create_session(self, run_id, task_id, camera_id="overview", width=640, height=480, fps=15.0):
        for name, value in (("run_id", run_id), ("task_id", task_id), ("camera_id", camera_id)):
            _required_id(name, value)
        key = (run_id, camera_id)
        with self._lock:
            if key in self._by_run_camera:
                existing = self._sessions[self._by_run_camera[key]]
                if existing.task_id != task_id:
                    raise ValueError("run/camera is already bound to another task")
                return existing
            stream_id = new_stream_id()
            while stream_id in self._sessions:
                stream_id = new_stream_id()
            session = StreamSession(stream_id, run_id, task_id, camera_id,
                StreamSessionStatus.CREATED, "%s/%s" % (self._rtsp_base, stream_id),
                "%s/%s" % (self._webrtc_base, stream_id),
                "%s/%s/whep" % (self._webrtc_base, stream_id), width, height, fps, _now())
            self._sessions[stream_id] = session
            self._by_run_camera[key] = stream_id
            return session

    def get_session(self, stream_id):
        with self._lock:
            if stream_id not in self._sessions:
                raise KeyError("stream session not found: %s" % stream_id)
            return self._sessions[stream_id]

    def start_session(self, stream_id):
        with self._lock:
            session = self.get_session(stream_id)
            if session.status in (StreamSessionStatus.STARTING, StreamSessionStatus.READY):
                return session
            if session.status != StreamSessionStatus.CREATED:
                raise InvalidStreamTransition("cannot start stream in %s" % session.status.value)
            session = session.transition_to(StreamSessionStatus.STARTING)
            self._sessions[stream_id] = session
            try:
                runtime = self._runtime_factory(session)
                self._runtime[stream_id] = runtime
                runtime.start()
            except Exception as exc:
                self._runtime.pop(stream_id, None)
                self._sessions[stream_id] = replace(session.transition_to(StreamSessionStatus.ERROR), video_error=str(exc))
                return self._sessions[stream_id]
            self._sessions[stream_id] = session.transition_to(StreamSessionStatus.READY)
            setter = getattr(runtime, "set_error_callback", None)
            if callable(setter):
                setter(lambda error: self.record_video_error(stream_id, error))
            return self._sessions[stream_id]

    def stop_session(self, stream_id):
        with self._lock:
            session = self.get_session(stream_id)
            if session.status == StreamSessionStatus.STOPPED:
                return session
            if session.status != StreamSessionStatus.STOPPING:
                session = session.transition_to(StreamSessionStatus.STOPPING)
                self._sessions[stream_id] = session
            runtime = self._runtime.pop(stream_id, None)
            try:
                if runtime is not None:
                    runtime.stop()
            except Exception as exc:
                session = replace(session, video_error=str(exc))
            self._sessions[stream_id] = session.transition_to(StreamSessionStatus.STOPPED)
            return self._sessions[stream_id]

    def mark_client_connected(self, stream_id):
        with self._lock:
            session = self.get_session(stream_id)
            if session.status != StreamSessionStatus.READY:
                raise InvalidStreamTransition("client can connect only to READY stream")
            reconnects = session.reconnect_count + int(session.last_client_disconnected_at is not None)
            self._sessions[stream_id] = replace(session, client_connected=True,
                reconnect_count=reconnects, last_client_connected_at=_now())
            return self._sessions[stream_id]

    def mark_client_disconnected(self, stream_id):
        with self._lock:
            session = self.get_session(stream_id)
            self._sessions[stream_id] = replace(session, client_connected=False,
                last_client_disconnected_at=_now())
            return self._sessions[stream_id]

    def record_video_error(self, stream_id, error):
        with self._lock:
            session = self.get_session(stream_id)
            if session.status in (StreamSessionStatus.STOPPING, StreamSessionStatus.STOPPED):
                return session
            if session.status != StreamSessionStatus.ERROR:
                session = session.transition_to(StreamSessionStatus.ERROR)
            self._sessions[stream_id] = replace(session, video_error=str(error))
            updated = self._sessions[stream_id]
        if self._session_observer is not None:
            try:
                self._session_observer(updated)
            except Exception:
                pass
        return updated

    def list_sessions_for_run(self, run_id):
        with self._lock:
            return tuple(item for item in self._sessions.values() if item.run_id == run_id)


class TaskStreamLifecycle:
    """Optional collaborator; video errors never mutate task state."""
    def __init__(self, manager: StreamSessionManager, camera_id="overview",
                 width=640, height=480, fps=15.0):
        self.manager = manager
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

    def on_task_state(self, task: SimulationTask):
        return self.on_status(task.run_id, task.task_id, task.status)

    def on_status(self, run_id, task_id, status):
        status = TaskStatus(status)
        sessions = self.manager.list_sessions_for_run(run_id)
        if status == TaskStatus.RUNNING:
            if not sessions:
                sessions = (self.manager.create_session(
                    run_id, task_id, self.camera_id,
                    self.width, self.height, self.fps,
                ),)
            return tuple(self.manager.start_session(item.stream_id) for item in sessions)
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED):
            return tuple(self.manager.stop_session(item.stream_id) for item in sessions)
        return sessions


class FakeTaskLifecycleDriver:
    """Test-only task sequencer; it never calls a backend or queue."""
    def __init__(self, task: SimulationTask, collaborator: TaskStreamLifecycle):
        self.task, self.collaborator = task, collaborator
        self.history = [task.status]

    def transition(self, target):
        self.task = self.task.transition_to(TaskStatus(target))
        self.history.append(self.task.status)
        self.collaborator.on_task_state(self.task)
        return self.task

    def complete(self):
        self.transition(TaskStatus.INITIALIZING); self.transition(TaskStatus.RUNNING)
        self.transition(TaskStatus.FINALIZING); return self.transition(TaskStatus.COMPLETED)

    def stop(self):
        self.transition(TaskStatus.INITIALIZING); self.transition(TaskStatus.RUNNING)
        self.transition(TaskStatus.STOPPING); return self.transition(TaskStatus.STOPPED)

    def fail(self):
        self.transition(TaskStatus.INITIALIZING); self.transition(TaskStatus.RUNNING)
        return self.transition(TaskStatus.FAILED)


class RecordingStreamGateway:
    def __init__(self): self.records = []
    def publish(self, session):
        record = {"stream_id": session.stream_id, "run_id": session.run_id,
            "task_id": session.task_id, "camera_id": session.camera_id,
            "status": session.status.value, "webrtc_url": session.webrtc_playback_url,
            "whep_url": session.whep_url, "width": session.width,
            "height": session.height, "fps": session.fps}
        self.records.append(record)
        return dict(record)


class RedisRecordingStreamGateway(RecordingStreamGateway):
    """Cross-process Mock Backend stream descriptor store."""
    def __init__(self, client, namespace="simulation"):
        self._client, self.namespace = client, namespace

    def publish(self, session):
        record = RecordingStreamGateway().publish(session)
        payload = json.dumps(record, separators=(",", ":"), sort_keys=True)
        self._client.set(self._key(session.run_id), payload)
        self._client.rpush(self._history_key(session.run_id), payload)
        return record

    def get_for_run(self, run_id):
        value = self._client.get(self._key(run_id))
        if value is None: return None
        return json.loads(value.decode() if isinstance(value, bytes) else value)

    def history_for_run(self, run_id):
        values = self._client.lrange(self._history_key(run_id), 0, -1)
        return [json.loads(v.decode() if isinstance(v, bytes) else v) for v in values]

    def _key(self, run_id): return "%s:run:%s:stream" % (self.namespace, run_id)
    def _history_key(self, run_id): return "%s:run:%s:stream_history" % (self.namespace, run_id)


@dataclass(frozen=True)
class StreamDescriptor:
    stream_id: str
    run_id: str
    task_id: str
    status: str
    webrtc_url: str
    whep_url: str
    camera_id: str
    width: int
    height: int
    fps: float

    @classmethod
    def from_session(cls, session):
        return cls(session.stream_id, session.run_id, session.task_id,
                   session.status.value, session.webrtc_playback_url,
                   session.whep_url, session.camera_id, session.width,
                   session.height, session.fps)

    def to_dict(self):
        return {
            "stream_id": self.stream_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "camera_id": self.camera_id,
            "status": self.status,
            "webrtc_url": self.webrtc_url,
            "whep_url": self.whep_url,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
