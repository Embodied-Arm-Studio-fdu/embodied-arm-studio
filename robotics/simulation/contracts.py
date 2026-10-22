"""JSON-safe request and response models for the Simulation Service."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_STREAM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("%s must be a non-empty safe identifier" % name)
    return value


def _mapping(name: str, value: object) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % name)
    return deepcopy(dict(value))


class SimulationTaskStatus(str, Enum):
    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class SimulationCatalogParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENUM = "enum"
    NUMBER_PAIR = "number_pair"
    NUMBER_TRIPLET = "number_triplet"


def _same_catalog_enum_value(left: Any, right: Any) -> bool:
    """Compare enum values using JSON number semantics.

    JSON has one numeric type, so a browser cannot preserve Python's distinction
    between integral floats such as ``0.0`` and integers such as ``0``.  Boolean
    values stay distinct even though ``bool`` is an ``int`` subclass in Python.
    """
    numeric = (int, float)
    if (
        isinstance(left, numeric)
        and not isinstance(left, bool)
        and isinstance(right, numeric)
        and not isinstance(right, bool)
    ):
        return left == right
    return type(left) is type(right) and left == right


class StreamSessionStatus(str, Enum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    READY = "READY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class InputEventType(str, Enum):
    KEYBOARD = "KEYBOARD"
    MOUSE_MOVE = "MOUSE_MOVE"
    MOUSE_BUTTON = "MOUSE_BUTTON"


_INPUT_COMMON_FIELDS = frozenset(
    {"event_id", "run_id", "task_id", "type", "occurred_at"}
)
_INPUT_TYPE_FIELDS = {
    InputEventType.KEYBOARD.value: frozenset({"key", "action"}),
    InputEventType.MOUSE_MOVE.value: frozenset({"x", "y"}),
    InputEventType.MOUSE_BUTTON.value: frozenset(
        {"button", "action", "x", "y"}
    ),
}
_INPUT_KEY = re.compile(r"[A-Za-z0-9]|Arrow(?:Up|Down|Left|Right)")


@dataclass(frozen=True)
class InputEventRequest:
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

    def __post_init__(self) -> None:
        for name in ("event_id", "run_id", "task_id"):
            object.__setattr__(
                self, name, validate_identifier(name, getattr(self, name))
            )
        if not isinstance(self.type, InputEventType):
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
            if not isinstance(self.key, str) or not _INPUT_KEY.fullmatch(self.key):
                raise ValueError("invalid keyboard key")
            if self.action not in {"pressed", "released"}:
                raise ValueError("invalid keyboard action")
            if any(value is not None for value in (self.x, self.y, self.button)):
                raise ValueError("keyboard event contains mouse fields")
        elif self.type == InputEventType.MOUSE_MOVE:
            self._validate_coordinates()
            if any(value is not None for value in (self.key, self.action, self.button)):
                raise ValueError("mouse move contains unrelated fields")
        else:
            self._validate_coordinates()
            if self.button not in {"left", "middle", "right"}:
                raise ValueError("invalid mouse button")
            if self.action not in {"pressed", "released"}:
                raise ValueError("invalid mouse button action")
            if self.key is not None:
                raise ValueError("mouse button contains key")

    def _validate_coordinates(self) -> None:
        for value in (self.x, self.y):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError("mouse coordinates must be normalized to 0..1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "InputEventRequest":
        payload = _mapping("input event", values)
        kind = payload.get("type")
        if not isinstance(kind, str) or kind not in _INPUT_TYPE_FIELDS:
            raise ValueError("invalid input event type")
        expected = _INPUT_COMMON_FIELDS | _INPUT_TYPE_FIELDS[kind]
        if set(payload) != expected:
            raise ValueError("input event fields do not match its type")
        occurred_at = payload["occurred_at"]
        if not isinstance(occurred_at, str):
            raise ValueError("occurred_at must be an ISO 8601 string")
        try:
            payload["occurred_at"] = datetime.fromisoformat(
                occurred_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("occurred_at must be an ISO 8601 datetime") from exc
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "type": self.type.value,
            "occurred_at": self.occurred_at.isoformat(),
        }
        for name in _INPUT_TYPE_FIELDS[self.type.value]:
            payload[name] = getattr(self, name)
        return payload


@dataclass(frozen=True)
class InputEventAck:
    accepted: bool
    event_id: str
    duplicate: bool

    def __post_init__(self) -> None:
        if self.accepted is not True:
            raise ValueError("accepted must be true")
        object.__setattr__(
            self, "event_id", validate_identifier("event_id", self.event_id)
        )
        if not isinstance(self.duplicate, bool):
            raise ValueError("duplicate must be boolean")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "InputEventAck":
        payload = _mapping("response", values)
        if set(payload) != {"accepted", "event_id", "duplicate"}:
            raise ValueError("response fields do not match InputEventAck")
        return cls(**payload)


@dataclass(frozen=True)
class CreateStreamSessionRequest:
    camera_id: str = "overview"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_id",
            validate_identifier("camera_id", self.camera_id),
        )
        if self.camera_id != "overview":
            raise ValueError("only the overview camera contract is confirmed")

    def to_dict(self) -> Dict[str, str]:
        return {"camera_id": self.camera_id}


@dataclass(frozen=True)
class StreamDescriptor:
    stream_id: str
    run_id: str
    task_id: str
    camera_id: str
    status: StreamSessionStatus
    webrtc_url: str
    whep_url: str
    width: int
    height: int
    fps: float

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not _STREAM_ID.fullmatch(
            self.stream_id
        ):
            raise ValueError("stream_id must be an opaque URL-safe identifier")
        for name in ("run_id", "task_id", "camera_id"):
            object.__setattr__(
                self, name, validate_identifier(name, getattr(self, name))
            )
        if not isinstance(self.status, StreamSessionStatus):
            try:
                object.__setattr__(
                    self, "status", StreamSessionStatus(self.status)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("status is not a recognized stream status") from exc
        for name in ("webrtc_url", "whep_url"):
            _validate_browser_stream_url(name, getattr(self, name))
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("width and height must be positive integers")
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("fps must be finite and positive")
        object.__setattr__(self, "fps", float(self.fps))

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        *,
        forbidden_values: Sequence[str] = (),
    ) -> "StreamDescriptor":
        payload = _mapping("response", values)
        fields = {
            "stream_id",
            "run_id",
            "task_id",
            "camera_id",
            "status",
            "webrtc_url",
            "whep_url",
            "width",
            "height",
            "fps",
        }
        if set(payload) != fields:
            raise ValueError("response fields do not match StreamDescriptor")
        for secret in forbidden_values:
            if isinstance(secret, str) and secret and any(
                secret in payload[name] for name in ("webrtc_url", "whep_url")
            ):
                raise ValueError("stream URL contains forbidden credentials")
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "camera_id": self.camera_id,
            "status": self.status.value,
            "webrtc_url": self.webrtc_url,
            "whep_url": self.whep_url,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


def _validate_browser_stream_url(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("stream URL is required")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("stream URL is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("stream URL must be a credential-free HTTP(S) URL")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("stream URL has an invalid port")
    return value


@dataclass(frozen=True)
class CreateTaskRequest:
    run_id: str
    project_id: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", validate_identifier("run_id", self.run_id))
        object.__setattr__(
            self, "project_id", validate_identifier("project_id", self.project_id)
        )
        object.__setattr__(self, "config", _mapping("config", self.config))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "config": deepcopy(dict(self.config)),
        }


@dataclass(frozen=True)
class SimulationCatalogParameter:
    name: str
    type: SimulationCatalogParameterType
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusive_minimum: Optional[float] = None
    step: Optional[float] = None
    enum: Optional[Sequence[Any]] = None
    nullable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_identifier("name", self.name))
        if not isinstance(self.nullable, bool):
            raise ValueError("nullable must be boolean")
        if not isinstance(self.type, SimulationCatalogParameterType):
            try:
                object.__setattr__(
                    self, "type", SimulationCatalogParameterType(self.type)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("unsupported catalog parameter type") from exc
        for name in ("minimum", "maximum", "exclusive_minimum", "step"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("%s must be a finite number" % name)
        if self.step is not None and self.step <= 0:
            raise ValueError("step must be positive")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum cannot exceed maximum")
        if self.type == SimulationCatalogParameterType.ENUM:
            if (
                not isinstance(self.enum, Sequence)
                or isinstance(self.enum, (str, bytes))
                or not self.enum
            ):
                raise ValueError("enum parameters require non-empty enum values")
            values = tuple(deepcopy(value) for value in self.enum)
            if not any(
                _same_catalog_enum_value(self.default, item) for item in values
            ):
                raise ValueError("enum default must be one of the enum values")
            object.__setattr__(self, "enum", values)
        elif self.enum is not None:
            raise ValueError("enum values are only valid for enum parameters")
        self._validate_default()
        object.__setattr__(self, "default", deepcopy(self.default))

    def _validate_default(self) -> None:
        if self.default is None and self.nullable:
            return
        if self.type == SimulationCatalogParameterType.INTEGER:
            valid = isinstance(self.default, int) and not isinstance(self.default, bool)
            numbers = (self.default,) if valid else ()
        elif self.type == SimulationCatalogParameterType.NUMBER:
            valid = (
                not isinstance(self.default, bool)
                and isinstance(self.default, (int, float))
                and math.isfinite(self.default)
            )
            numbers = (self.default,) if valid else ()
        elif self.type == SimulationCatalogParameterType.BOOLEAN:
            valid = isinstance(self.default, bool)
            numbers = ()
        elif self.type in {
            SimulationCatalogParameterType.NUMBER_PAIR,
            SimulationCatalogParameterType.NUMBER_TRIPLET,
        }:
            expected_length = (
                2 if self.type == SimulationCatalogParameterType.NUMBER_PAIR else 3
            )
            valid = (
                isinstance(self.default, Sequence)
                and not isinstance(self.default, (str, bytes))
                and len(self.default) == expected_length
                and all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(value)
                    for value in self.default
                )
            )
            numbers = tuple(self.default) if valid else ()
        elif self.type == SimulationCatalogParameterType.STRING:
            valid = isinstance(self.default, str)
            numbers = ()
        else:
            valid = True
            numbers = ()
        if not valid:
            raise ValueError("catalog parameter default does not match its type")
        if self.minimum is not None and any(value < self.minimum for value in numbers):
            raise ValueError("catalog parameter default is below minimum")
        if self.exclusive_minimum is not None and any(
            value <= self.exclusive_minimum for value in numbers
        ):
            raise ValueError("catalog parameter default is below exclusive minimum")
        if self.maximum is not None and any(value > self.maximum for value in numbers):
            raise ValueError("catalog parameter default exceeds maximum")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SimulationCatalogParameter":
        payload = _mapping("catalog parameter", values)
        required = {"name", "type", "default"}
        allowed = required | {
            "minimum", "maximum", "exclusive_minimum", "step", "enum",
            "fixed", "nullable", "planning_access", "required",
        }
        if set(payload) - allowed:
            raise ValueError("catalog parameter contains unknown fields")
        if required - set(payload):
            raise ValueError("catalog parameter is missing required fields")
        contract_payload = {
            name: payload[name]
            for name in (
                "name", "type", "default", "minimum", "maximum",
                "exclusive_minimum", "step", "enum", "nullable",
            )
            if name in payload
        }
        if contract_payload.get("type") == "string" and contract_payload.get("enum"):
            contract_payload["type"] = "enum"
        return cls(**contract_payload)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "type": self.type.value,
            "default": deepcopy(self.default),
        }
        for name in ("minimum", "maximum", "exclusive_minimum", "step"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.enum is not None:
            payload["enum"] = deepcopy(list(self.enum))
        if self.nullable:
            payload["nullable"] = True
        return payload


@dataclass(frozen=True)
class SimulationCatalogTask:
    id: str
    display_name: str
    task_type: str
    robot: str
    supports_training: bool
    supports_video: bool
    parameters: Sequence[SimulationCatalogParameter]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", validate_identifier("id", self.id))
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name is required")
        object.__setattr__(self, "robot", validate_identifier("robot", self.robot))
        if self.task_type not in {"training", "demo"}:
            raise ValueError("task_type must be training or demo")
        if not isinstance(self.supports_training, bool) or not isinstance(
            self.supports_video, bool
        ):
            raise ValueError("task capabilities must be boolean")
        if self.supports_training != (self.task_type == "training"):
            raise ValueError("supports_training must match task_type")
        if not isinstance(self.parameters, Sequence) or isinstance(
            self.parameters, (str, bytes)
        ):
            raise ValueError("parameters must be an array")
        parsed = tuple(
            parameter
            if isinstance(parameter, SimulationCatalogParameter)
            else SimulationCatalogParameter.from_dict(parameter)
            for parameter in self.parameters
        )
        if not parsed or len({parameter.name for parameter in parsed}) != len(parsed):
            raise ValueError("catalog task parameters must be non-empty and unique")
        object.__setattr__(self, "parameters", parsed)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SimulationCatalogTask":
        payload = _mapping("catalog task", values)
        required = {
            "id", "display_name", "task_type", "robot", "supports_training",
            "supports_video", "parameters",
        }
        metadata = {
            "composition", "display_names", "environment_type",
            "expected_artifacts", "expected_metrics", "goal_success",
            "intent_tags", "parameter_schema", "parameter_schema_digest",
            "parameter_schema_ref", "planning_role", "preconditions",
            "resource_profile", "run_kind", "safety", "summary", "task_id",
        }
        if required - set(payload) or set(payload) - required - metadata:
            raise ValueError("catalog task fields do not match the contract")
        return cls(**{name: payload[name] for name in required})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "robot": self.robot,
            "supports_training": self.supports_training,
            "supports_video": self.supports_video,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


@dataclass(frozen=True)
class SimulationTaskCatalog:
    tasks: Sequence[SimulationCatalogTask]
    catalog_version: Optional[str] = None
    catalog_digest: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tasks, Sequence) or isinstance(self.tasks, (str, bytes)):
            raise ValueError("tasks must be an array")
        parsed = tuple(
            task
            if isinstance(task, SimulationCatalogTask)
            else SimulationCatalogTask.from_dict(task)
            for task in self.tasks
        )
        if not parsed or len({task.id for task in parsed}) != len(parsed):
            raise ValueError("catalog tasks must be non-empty and unique")
        object.__setattr__(self, "tasks", parsed)
        if self.catalog_version is not None and (
            not isinstance(self.catalog_version, str) or not self.catalog_version
        ):
            raise ValueError("catalog_version must be a non-empty string")
        if self.catalog_digest is not None and (
            not isinstance(self.catalog_digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.catalog_digest)
        ):
            raise ValueError("catalog_digest must be a sha256 digest")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SimulationTaskCatalog":
        payload = _mapping("task catalog", values)
        if "tasks" not in payload or set(payload) - {
            "tasks", "catalog_version", "catalog_digest"
        }:
            raise ValueError("task catalog fields do not match the contract")
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"tasks": [task.to_dict() for task in self.tasks]}
        if self.catalog_version is not None:
            payload["catalog_version"] = self.catalog_version
        if self.catalog_digest is not None:
            payload["catalog_digest"] = self.catalog_digest
        return payload


def resolve_catalog_task_parameters(
    catalog: SimulationTaskCatalog,
    environment_type: str,
    task_type: str,
    supplied: Mapping[str, Any],
    *,
    baseline: Optional[Mapping[str, Any]] = None,
) -> Tuple[SimulationCatalogTask, Dict[str, Any]]:
    """Resolve defaults and validate one create request solely from catalog metadata."""

    if not isinstance(catalog, SimulationTaskCatalog):
        raise ValueError("catalog must be a parsed SimulationTaskCatalog")
    validate_identifier("environment_type", environment_type)
    if task_type not in {"training", "demo"}:
        raise ValueError("task_type must be training or demo")
    supplied_values = _mapping("parameters", supplied)
    baseline_values = _mapping("baseline", baseline or {})
    try:
        task = next(item for item in catalog.tasks if item.id == environment_type)
    except StopIteration as exc:
        raise ValueError("environment_type is not present in the Simulation catalog") from exc
    if task.task_type != task_type:
        raise ValueError("environment_type does not match run_kind")

    definitions = {parameter.name: parameter for parameter in task.parameters}
    unknown = sorted((set(supplied_values) | set(baseline_values)) - set(definitions))
    if unknown:
        raise ValueError("parameters contain unknown fields: %s" % ", ".join(unknown))
    resolved = {
        name: deepcopy(parameter.default) for name, parameter in definitions.items()
    }
    resolved.update(deepcopy(baseline_values))
    resolved.update(deepcopy(supplied_values))
    for name, parameter in definitions.items():
        resolved[name] = _validate_catalog_parameter_value(parameter, resolved[name])
    return task, resolved


def _validate_catalog_parameter_value(
    parameter: SimulationCatalogParameter, value: Any
) -> Any:
    kind = parameter.type
    if value is None and parameter.nullable:
        return None
    if kind is SimulationCatalogParameterType.INTEGER:
        valid = isinstance(value, int) and not isinstance(value, bool)
        numbers = (value,) if valid else ()
    elif kind is SimulationCatalogParameterType.NUMBER:
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
        numbers = (value,) if valid else ()
    elif kind is SimulationCatalogParameterType.BOOLEAN:
        valid = isinstance(value, bool)
        numbers = ()
    elif kind in {
        SimulationCatalogParameterType.NUMBER_PAIR,
        SimulationCatalogParameterType.NUMBER_TRIPLET,
    }:
        expected_length = 2 if kind is SimulationCatalogParameterType.NUMBER_PAIR else 3
        valid = (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == expected_length
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(item)
                for item in value
            )
        )
        numbers = tuple(value) if valid else ()
    elif kind is SimulationCatalogParameterType.STRING:
        valid = isinstance(value, str)
        numbers = ()
    else:
        valid = parameter.enum is not None and any(
            _same_catalog_enum_value(value, item) for item in parameter.enum
        )
        numbers = ()
    if not valid:
        raise ValueError("parameter %s does not match catalog type" % parameter.name)
    if parameter.minimum is not None and any(
        item < parameter.minimum for item in numbers
    ):
        raise ValueError("parameter %s is below minimum" % parameter.name)
    if parameter.exclusive_minimum is not None and any(
        item <= parameter.exclusive_minimum for item in numbers
    ):
        raise ValueError("parameter %s is below exclusive minimum" % parameter.name)
    if parameter.maximum is not None and any(
        item > parameter.maximum for item in numbers
    ):
        raise ValueError("parameter %s exceeds maximum" % parameter.name)
    if parameter.step is not None:
        base = parameter.minimum if parameter.minimum is not None else 0.0
        if any(
            not math.isclose((item - base) / parameter.step, round((item - base) / parameter.step), abs_tol=1e-9)
            for item in numbers
        ):
            raise ValueError("parameter %s does not match step" % parameter.name)
    if kind in {
        SimulationCatalogParameterType.NUMBER_PAIR,
        SimulationCatalogParameterType.NUMBER_TRIPLET,
    }:
        return list(value)
    return deepcopy(value)


@dataclass(frozen=True)
class SimulationTaskResponse:
    """Complete task snapshot returned by Simulation HTTP schema 1.0."""

    task_id: str
    run_id: str
    project_id: str
    status: SimulationTaskStatus
    created_at: datetime
    updated_at: datetime
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", validate_identifier("task_id", self.task_id))
        object.__setattr__(self, "run_id", validate_identifier("run_id", self.run_id))
        object.__setattr__(
            self, "project_id", validate_identifier("project_id", self.project_id)
        )
        if not isinstance(self.status, SimulationTaskStatus):
            try:
                object.__setattr__(self, "status", SimulationTaskStatus(self.status))
            except (TypeError, ValueError) as exc:
                raise ValueError("status is not a recognized simulation task status") from exc
        for name in ("created_at", "updated_at"):
            value = getattr(self, name)
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError("%s must be an ISO 8601 datetime" % name) from exc
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise ValueError("%s must be timezone-aware UTC" % name)
            if value.utcoffset() != timedelta(0):
                raise ValueError("%s must be UTC" % name)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "config", _mapping("config", self.config))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SimulationTaskResponse":
        payload = _mapping("response", values)
        allowed = {
            "task_id", "run_id", "project_id", "status", "created_at",
            "updated_at", "config",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("response contains unknown fields: %s" % ", ".join(unknown))
        required = allowed
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError("response is missing fields: %s" % ", ".join(missing))
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "project_id": self.project_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "config": deepcopy(dict(self.config)),
        }
