"""Canonical Evaluation API schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from robotics.evaluations import EvaluationStatus


class V1EvaluationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1, max_length=255)


class V1EvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    evaluation_id: str
    run_id: str
    requested_by_user_id: str
    status: EvaluationStatus
    score: Optional[float]
    grade: str
    rubric_version: str
    rubric_snapshot: Dict[str, Any]
    evidence: Dict[str, Any]
    evidence_sha256: str
    summary: Dict[str, Any]
    limitations: List[str]
    evaluator_version: str
    created_at: datetime
