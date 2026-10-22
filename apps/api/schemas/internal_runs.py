"""Strict schema 1.0 models for Simulation -> Backend Run Events."""

import math
import re
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Dict, List, Literal, Optional, Union

from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)


NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
MetricNumber = Union[StrictInt, StrictFloat]
STATE_TRANSITIONS = frozenset(
    {
        ("queued", "initializing"),
        ("initializing", "running"),
        ("running", "finalizing"),
        ("running", "stopping"),
    }
)
SENSITIVE_MESSAGE_PATTERN = re.compile(
    r"traceback|authorization\s*:|(?:token|password|secret)\s*=|"
    r"(?:postgres(?:ql)?|redis)://",
    re.IGNORECASE,
)


class _StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class StateEventPayload(_StrictPayload):
    previous_status: Literal[
        "queued", "initializing", "running", "finalizing", "stopping"
    ]
    status: Literal["initializing", "running", "finalizing", "stopping"]

    @model_validator(mode="after")
    def validate_transition(self) -> "StateEventPayload":
        if (self.previous_status, self.status) not in STATE_TRANSITIONS:
            raise ValueError("state payload transition is not allowed")
        return self


class MetricEventPayload(_StrictPayload):
    step: NonNegativeInt
    metrics: Dict[str, MetricNumber] = Field(..., min_length=1)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(
        cls, value: Dict[str, MetricNumber]
    ) -> Dict[str, MetricNumber]:
        for name, number in value.items():
            if not name.strip():
                raise ValueError("metric names must not be blank")
            if isinstance(number, bool) or not math.isfinite(number):
                raise ValueError("metrics must contain finite numbers")
            if name == "update_index" and (
                not isinstance(number, int) or number < 1
            ):
                raise ValueError(
                    "update_index must be an integer greater than or equal to 1"
                )
        return value


class HeartbeatEventPayload(_StrictPayload):
    status: Literal["initializing", "running", "finalizing", "stopping"]
    training_steps: NonNegativeInt


class ErrorEventPayload(_StrictPayload):
    code: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1, max_length=1000)
    retryable: bool

    @field_validator("code", "message")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("message")
    @classmethod
    def require_redacted_message(cls, value: str) -> str:
        if any(character in value for character in ("\r", "\n", "\x00")):
            raise ValueError("error message must be a single-line summary")
        if SENSITIVE_MESSAGE_PATTERN.search(value):
            raise ValueError("error message must be redacted")
        return value


Payload = Union[
    StateEventPayload,
    MetricEventPayload,
    HeartbeatEventPayload,
    ErrorEventPayload,
]


class InternalRunEventRequest(BaseModel):
    # FastAPI decodes JSON before model validation, so datetime parsing must
    # remain enabled; integer and payload scalar fields are strict explicitly.
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    event_id: str = Field(..., min_length=1, max_length=255)
    sequence: Annotated[StrictInt, Field(ge=1)]
    type: Literal["state", "metric", "heartbeat", "error"]
    run_id: str = Field(..., min_length=1, max_length=255)
    task_id: str = Field(..., min_length=1, max_length=255)
    occurred_at: datetime
    payload: Payload

    @field_validator("event_id", "run_id", "task_id")
    @classmethod
    def reject_blank_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("occurred_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_payload_type(self) -> "InternalRunEventRequest":
        expected = {
            "state": StateEventPayload,
            "metric": MetricEventPayload,
            "heartbeat": HeartbeatEventPayload,
            "error": ErrorEventPayload,
        }[self.type]
        if type(self.payload) is not expected:
            raise ValueError("payload does not match Event type")
        return self


class InternalRunEventResponse(BaseModel):
    accepted: Literal[True] = True
    duplicate: bool
    late: bool


class TerminalArtifactReference(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, populate_by_name=True
    )

    artifact_type: str = Field(..., alias="type", min_length=1, max_length=255)
    path: str = Field(..., min_length=1, max_length=1024)
    size_bytes: Optional[NonNegativeInt] = None
    sha256: Optional[str] = None

    @field_validator("artifact_type")
    @classmethod
    def reject_blank_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact type must not be blank")
        return value

    @field_validator("path")
    @classmethod
    def require_safe_relative_posix_path(cls, value: str) -> str:
        if "\\" in value or "\x00" in value:
            raise ValueError("artifact path must use safe POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path must be Run-relative")
        if value in {"", "."} or str(path) != value:
            raise ValueError("artifact path must be normalized")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError("sha256 must be 64 hexadecimal characters")
        return value.lower() if value is not None else None


class TerminalEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["passed", "failed", "unavailable"]
    deterministic: Literal[True]
    policy_passed: bool
    episode_count: Optional[NonNegativeInt] = None
    success_count: Optional[NonNegativeInt] = None
    success_rate: Optional[MetricNumber] = None
    wilson_95_lower_bound: Optional[MetricNumber] = None
    wilson_95_upper_bound: Optional[MetricNumber] = None
    mean_episode_length: Optional[MetricNumber] = None
    mean_episode_return: Optional[MetricNumber] = None
    mean_final_distance: Optional[MetricNumber] = None
    mean_minimum_distance: Optional[MetricNumber] = None
    mean_object_displacement: Optional[MetricNumber] = None
    timeout_rate: Optional[MetricNumber] = None
    failure_rate: Optional[MetricNumber] = None
    failure_reason: Optional[str] = Field(None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def validate_result(self) -> "TerminalEvaluation":
        if self.policy_passed != (self.status == "passed"):
            raise ValueError("evaluation status and policy_passed disagree")
        metric_names = (
            "episode_count", "success_count", "success_rate",
            "wilson_95_lower_bound", "wilson_95_upper_bound",
            "mean_episode_length", "mean_episode_return", "mean_final_distance",
            "mean_minimum_distance", "mean_object_displacement", "timeout_rate",
            "failure_rate",
        )
        if self.status == "unavailable":
            if self.failure_reason is None or not self.failure_reason.strip():
                raise ValueError("unavailable evaluation requires failure_reason")
            if any(getattr(self, name) is not None for name in metric_names):
                raise ValueError("unavailable evaluation must not claim metrics")
            return self
        if self.failure_reason is not None:
            raise ValueError("available evaluation must not have failure_reason")
        if any(getattr(self, name) is None for name in metric_names):
            raise ValueError("available evaluation requires result metrics")
        if self.success_count > self.episode_count:
            raise ValueError("evaluation success_count exceeds episode_count")
        if self.episode_count < 100:
            raise ValueError("available evaluation requires at least 100 episodes")
        for name in (
            "success_rate", "wilson_95_lower_bound", "wilson_95_upper_bound",
        ):
            number = getattr(self, name)
            if isinstance(number, bool) or not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError("evaluation rates must be finite within [0, 1]")
        expected_rate = self.success_count / self.episode_count
        if not math.isclose(self.success_rate, expected_rate, abs_tol=1e-12):
            raise ValueError("evaluation success counts are inconsistent")
        for name in (
            "mean_episode_length", "mean_episode_return", "mean_final_distance",
            "mean_minimum_distance", "mean_object_displacement", "timeout_rate",
            "failure_rate",
        ):
            number = getattr(self, name)
            if number is not None and (
                isinstance(number, bool) or not math.isfinite(number)
            ):
                raise ValueError("evaluation detail metrics must be finite")
        for name in (
            "mean_episode_length", "mean_final_distance",
            "mean_minimum_distance", "mean_object_displacement",
        ):
            number = getattr(self, name)
            if number is not None and number < 0:
                raise ValueError("evaluation lengths and distances must be non-negative")
        for name in ("timeout_rate", "failure_rate"):
            number = getattr(self, name)
            if number is not None and not 0 <= number <= 1:
                raise ValueError("evaluation rates must be finite within [0, 1]")
        z = 1.959963984540054
        z_squared = z * z
        denominator = 1.0 + z_squared / self.episode_count
        center = (
            expected_rate + z_squared / (2.0 * self.episode_count)
        ) / denominator
        half_width = z / denominator * math.sqrt(
            expected_rate * (1.0 - expected_rate) / self.episode_count
            + z_squared / (4.0 * self.episode_count * self.episode_count)
        )
        expected_lower = (
            0.0 if self.success_count == 0 else max(0.0, center - half_width)
        )
        expected_upper = (
            1.0 if self.success_count == self.episode_count
            else min(1.0, center + half_width)
        )
        if not math.isclose(
            self.wilson_95_lower_bound, expected_lower, abs_tol=1e-12
        ) or not math.isclose(
            self.wilson_95_upper_bound, expected_upper, abs_tol=1e-12
        ):
            raise ValueError("evaluation Wilson 95% interval is inconsistent")
        expected_passed = self.success_rate >= 0.80 and expected_lower >= 0.70
        if self.policy_passed != expected_passed:
            raise ValueError("evaluation policy gate is inconsistent")
        return self


class TerminalCallbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    event_id: str = Field(..., min_length=1, max_length=255)
    sequence: Annotated[StrictInt, Field(ge=1)]
    run_id: str = Field(..., min_length=1, max_length=255)
    task_id: str = Field(..., min_length=1, max_length=255)
    status: Literal["completed", "failed", "stopped"]
    started_at: datetime
    ended_at: datetime
    training_steps: NonNegativeInt
    update_steps: NonNegativeInt
    stop_reason: Optional[str] = Field(..., max_length=255)
    metrics: Dict[str, MetricNumber]
    model: Optional[TerminalArtifactReference]
    artifacts: List[TerminalArtifactReference]
    error: Optional[ErrorEventPayload]
    evaluation: Optional[TerminalEvaluation] = None

    @field_validator("event_id", "run_id", "task_id")
    @classmethod
    def reject_blank_callback_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value

    @field_validator("started_at", "ended_at")
    @classmethod
    def require_callback_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("Callback times must be timezone-aware UTC")
        return value

    @field_validator("metrics")
    @classmethod
    def validate_final_metrics(
        cls, value: Dict[str, MetricNumber]
    ) -> Dict[str, MetricNumber]:
        for name, number in value.items():
            if (
                not name.strip()
                or isinstance(number, bool)
                or not math.isfinite(number)
            ):
                raise ValueError("metrics must contain named finite numbers")
            if name == "update_index" and (
                not isinstance(number, int) or number < 1
            ):
                raise ValueError(
                    "update_index must be an integer greater than or equal to 1"
                )
        return value

    @model_validator(mode="after")
    def validate_terminal_contract(self) -> "TerminalCallbackRequest":
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not be before started_at")
        if self.status == "completed":
            if self.error is not None:
                raise ValueError("completed Callback error must be null")
            if not any(item.artifact_type == "manifest" for item in self.artifacts):
                raise ValueError("completed Callback requires manifest Artifact")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed Callback requires error")
        if self.status == "stopped" and (
            self.stop_reason is None or not self.stop_reason.strip()
        ):
            raise ValueError("stopped Callback requires stop_reason")
        self._validate_duplicate_artifact_metadata()
        artifact_types = [item.artifact_type for item in self.artifacts]
        if len(artifact_types) != len(set(artifact_types)):
            raise ValueError("Callback Artifact types must be unique")
        return self

    def _validate_duplicate_artifact_metadata(self) -> None:
        by_path = {}
        for item in self.artifacts + ([self.model] if self.model is not None else []):
            existing = by_path.get(item.path)
            if existing is None:
                by_path[item.path] = item
                continue
            for field in ("size_bytes", "sha256"):
                left = getattr(existing, field)
                right = getattr(item, field)
                if left is not None and right is not None and left != right:
                    raise ValueError("duplicate artifact path has conflicting metadata")


class TerminalCallbackResponse(BaseModel):
    accepted: Literal[True] = True
    duplicate: bool

