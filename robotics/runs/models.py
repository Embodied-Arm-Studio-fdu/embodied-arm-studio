"""Canonical platform Run and Event domain models."""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional

from robotics.runs.domain import RunStatus


SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value


def _validate_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("%s must be a datetime" % name)
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("%s must be timezone-aware UTC" % name)
    return value


def _copy_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("%s must be a dict" % name)
    return deepcopy(value)


class ExecutionMode(str, Enum):
    MOCK = "mock"
    SIMULATION = "simulation"


class RunKind(str, Enum):
    TRAIN = "train"
    PLAY = "play"


class RunEventType(str, Enum):
    STATE = "state"
    METRIC = "metric"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass(frozen=True)
class RunError:
    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        _required_text(self.code, "error.code")
        _required_text(self.message, "error.message")
        if not isinstance(self.retryable, bool):
            raise ValueError("error.retryable must be a boolean")


@dataclass(frozen=True)
class Run:
    run_id: str
    user_id: str
    project_id: str
    status: RunStatus
    execution_mode: ExecutionMode
    code_sha256: str
    class_id: Optional[str] = None
    run_kind: RunKind = RunKind.TRAIN
    parameters: Dict[str, Any] = field(default_factory=dict)
    project_config_snapshot: Dict[str, Any] = field(default_factory=dict)
    simulation_task_id: Optional[str] = None
    progress: float = 0.0
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[RunError] = None
    last_sequence: int = 0
    training_steps: int = 0
    update_steps: int = 0
    stop_reason: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _required_text(self.run_id, "run_id")
        _required_text(self.user_id, "user_id")
        _required_text(self.project_id, "project_id")
        _required_text(self.code_sha256, "code_sha256")
        if self.class_id is not None:
            _required_text(self.class_id, "class_id")
        try:
            status = RunStatus(self.status)
        except ValueError as exc:
            raise ValueError("unsupported Run status: %s" % self.status) from exc
        try:
            execution_mode = ExecutionMode(self.execution_mode)
        except ValueError as exc:
            raise ValueError(
                "unsupported execution mode: %s" % self.execution_mode
            ) from exc
        try:
            run_kind = RunKind(self.run_kind)
        except ValueError as exc:
            raise ValueError("unsupported Run kind: %s" % self.run_kind) from exc
        if self.simulation_task_id is not None:
            _required_text(self.simulation_task_id, "simulation_task_id")
        if (
            isinstance(self.progress, bool)
            or not isinstance(self.progress, (int, float))
            or not 0 <= self.progress <= 1
        ):
            raise ValueError("progress must be between 0 and 1")
        if (
            isinstance(self.last_sequence, bool)
            or not isinstance(self.last_sequence, int)
            or self.last_sequence < 0
        ):
            raise ValueError("last_sequence must be a non-negative integer")
        for value, name in (
            (self.training_steps, "training_steps"),
            (self.update_steps, "update_steps"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("%s must be a non-negative integer" % name)
        if self.stop_reason is not None:
            _required_text(self.stop_reason, "stop_reason")
        if self.error is not None and not isinstance(self.error, RunError):
            raise ValueError("error must be a RunError or None")
        _validate_utc(self.created_at, "created_at")
        _validate_utc(self.updated_at, "updated_at")
        if self.started_at is not None:
            _validate_utc(self.started_at, "started_at")
        if self.ended_at is not None:
            _validate_utc(self.ended_at, "ended_at")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "execution_mode", execution_mode)
        object.__setattr__(self, "run_kind", run_kind)
        object.__setattr__(self, "progress", float(self.progress))
        object.__setattr__(self, "parameters", _copy_dict(self.parameters, "parameters"))
        object.__setattr__(
            self,
            "project_config_snapshot",
            _copy_dict(self.project_config_snapshot, "project_config_snapshot"),
        )
        object.__setattr__(
            self,
            "metrics_summary",
            _copy_dict(self.metrics_summary, "metrics_summary"),
        )


@dataclass(frozen=True)
class RunEvent:
    event_id: str
    sequence: int
    type: RunEventType
    run_id: str
    occurred_at: datetime
    payload: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.event_id, "event_id")
        _required_text(self.run_id, "run_id")
        if self.task_id is not None:
            _required_text(self.task_id, "task_id")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("sequence must be a positive integer")
        try:
            event_type = RunEventType(self.type)
        except ValueError as exc:
            raise ValueError("unsupported event type: %s" % self.type) from exc
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("schema_version must be %s" % SCHEMA_VERSION)
        _validate_utc(self.occurred_at, "occurred_at")
        object.__setattr__(self, "type", event_type)
        object.__setattr__(self, "payload", _copy_dict(self.payload, "payload"))
