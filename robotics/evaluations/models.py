"""Immutable canonical Evaluation report model."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class EvaluationStatus(str, Enum):
    ASSESSED = "assessed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value


@dataclass(frozen=True)
class EvaluationReport:
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

    def __post_init__(self) -> None:
        for value, name in (
            (self.evaluation_id, "evaluation_id"),
            (self.run_id, "run_id"),
            (self.requested_by_user_id, "requested_by_user_id"),
            (self.grade, "grade"),
            (self.rubric_version, "rubric_version"),
            (self.evidence_sha256, "evidence_sha256"),
            (self.evaluator_version, "evaluator_version"),
        ):
            _text(value, name)
        if not isinstance(self.status, EvaluationStatus):
            raise ValueError("status must be an EvaluationStatus")
        if self.score is not None and (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not 0 <= self.score <= 100
        ):
            raise ValueError("score must be between 0 and 100")
        if self.status == EvaluationStatus.ASSESSED and self.score is None:
            raise ValueError("assessed report requires score")
        if (
            self.status == EvaluationStatus.INSUFFICIENT_EVIDENCE
            and self.score is not None
        ):
            raise ValueError("insufficient report cannot have score")
        if len(self.evidence_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.evidence_sha256
        ):
            raise ValueError("evidence_sha256 must be lowercase hexadecimal")
        for value, name in (
            (self.rubric_snapshot, "rubric_snapshot"),
            (self.evidence, "evidence"),
            (self.summary, "summary"),
        ):
            if not isinstance(value, dict):
                raise ValueError("%s must be an object" % name)
            object.__setattr__(self, name, deepcopy(value))
        if not isinstance(self.limitations, list) or not all(
            isinstance(item, str) and item.strip() for item in self.limitations
        ):
            raise ValueError("limitations must contain non-empty strings")
        object.__setattr__(self, "limitations", deepcopy(self.limitations))
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("created_at must be timezone-aware UTC")
