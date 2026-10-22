"""Strict canonical teaching Project request and response schemas."""

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from robotics.projects import TrainingProfile


class TeachingProjectCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field("", max_length=4000)
    instructions: str = Field("", max_length=16000)
    starter_code: str = Field("", max_length=524288)
    training_profile: TrainingProfile = TrainingProfile.SHORT
    video_enabled: bool = True
    enabled: bool = True
    environment_type: Literal[
        "s1_push", "s1_reach", "s1_grasp", "s1_lift", "s1_lower_release", "s1_place",
    ] = "s1_push"


class TeachingProjectPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_version: int = Field(..., ge=1)
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=4000)
    instructions: Optional[str] = Field(None, max_length=16000)
    starter_code: Optional[str] = Field(None, max_length=524288)
    training_profile: Optional[TrainingProfile] = None
    video_enabled: Optional[bool] = None
    enabled: Optional[bool] = None


class TeachingProjectPermissionsResponse(BaseModel):
    can_view: bool
    can_use: bool
    can_modify: bool


class TeachingProjectResponse(BaseModel):
    project_id: str
    course_id: str
    title: str
    description: str
    instructions: str
    starter_code: str
    starter_code_sha256: str
    task_type: str
    environment_type: str
    algorithm: str
    success_condition: Dict[str, Any]
    training_profile: TrainingProfile
    expected_training_steps: int
    video_enabled: bool
    parameter_schema: Dict[str, Any]
    evaluation_rubric: Dict[str, Any]
    config_version: int
    enabled: bool
    created_by_user_id: str
    created_at: datetime
    updated_at: datetime
    permissions: TeachingProjectPermissionsResponse
