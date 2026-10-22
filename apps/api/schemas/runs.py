from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from robotics.runs import ExecutionMode, RunEventType, RunKind
from robotics.runs import RunStatus as CanonicalRunStatus
from robotics.simulation import StreamSessionStatus


class RunCreate(BaseModel):
    environment_id: str
    strategy_id: str
    seed: int = 42
    max_steps: int = Field(120, ge=1, le=2000)
    config: Dict[str, Any] = Field(default_factory=dict)


class V1RunParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: StrictInt = Field(..., ge=0, le=2147483647)


class V1RunSceneReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(..., min_length=1, max_length=255)
    revision: StrictInt = Field(..., ge=1)


class V1RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(..., min_length=1, max_length=255)
    class_id: str = Field(..., min_length=1, max_length=255)
    project_config_version: StrictInt = Field(..., ge=1)
    execution_mode: Literal[ExecutionMode.SIMULATION]
    run_kind: Literal[RunKind.TRAIN, RunKind.PLAY]
    environment_type: Optional[str] = Field(None, min_length=1, max_length=128)
    parameters: Dict[str, Any]
    scene_ref: Optional[V1RunSceneReference] = None
    code: str = Field("", max_length=524288)
    reward_extension: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def preserve_legacy_training_contract(self) -> "V1RunCreateRequest":
        if self.environment_type is None:
            seed = self.parameters.get("seed")
            if self.run_kind is not RunKind.TRAIN:
                raise ValueError("Demo Runs require environment_type")
            if (
                set(self.parameters) != {"seed"}
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= 2147483647
            ):
                raise ValueError("legacy Training parameters must contain only seed in range")
        return self


class CompareRequest(BaseModel):
    run_ids: List[str] = Field(..., min_length=2, max_length=4)


class ExportRequest(BaseModel):
    format: str = "manifest"
    evaluation_report_id: Optional[str] = None


class V1RunBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_ids: List[str] = Field(..., min_length=2, max_length=10)

    @field_validator("run_ids")
    @classmethod
    def run_ids_are_unique_and_nonempty(cls, value: List[str]) -> List[str]:
        if any(not isinstance(run_id, str) or not run_id.strip() for run_id in value):
            raise ValueError("run_ids must contain non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError("run_ids must not contain duplicates")
        return value


class V1RunExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_ids: List[str] = Field(..., min_length=1, max_length=10)
    format: Literal["json"] = "json"

    @field_validator("run_ids")
    @classmethod
    def run_ids_are_unique_and_nonempty(cls, value: List[str]) -> List[str]:
        if any(not isinstance(run_id, str) or not run_id.strip() for run_id in value):
            raise ValueError("run_ids must contain non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError("run_ids must not contain duplicates")
        return value


class V1RunDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_ids: List[str] = Field(..., min_length=1, max_length=10)

    @field_validator("run_ids")
    @classmethod
    def run_ids_are_unique_and_nonempty(cls, value: List[str]) -> List[str]:
        if any(not run_id.strip() for run_id in value) or len(set(value)) != len(value):
            raise ValueError("run_ids must contain unique, non-empty strings")
        return value


class V1RunComparisonResponse(BaseModel):
    schema_version: Literal["1.0"]
    run_ids: List[str]
    cross_project: bool
    project_ids: List[str]
    runs: List[Dict[str, Any]]
    differences: Dict[str, Any]
    limitations: List[str]


class V1RunErrorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    code: str
    message: str
    retryable: bool


class V1RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    user_id: str
    project_id: str
    class_id: Optional[str]
    status: CanonicalRunStatus
    execution_mode: ExecutionMode
    run_kind: RunKind
    parameters: Dict[str, Any]
    project_config_snapshot: Dict[str, Any]
    code_sha256: str
    simulation_task_id: Optional[str]
    progress: float
    metrics_summary: Dict[str, Any]
    error: Optional[V1RunErrorResponse]
    last_sequence: int
    training_steps: int
    update_steps: int
    stop_reason: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    updated_at: datetime


class V1RunEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    sequence: int
    type: RunEventType
    run_id: str
    task_id: Optional[str]
    occurred_at: datetime
    schema_version: str
    payload: Dict[str, Any]


class V1RunArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    artifact_id: str
    run_id: str
    artifact_type: str
    path: str
    size_bytes: Optional[int]
    sha256: Optional[str]
    is_primary_model: bool
    created_at: datetime


class V1StreamDescriptorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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

