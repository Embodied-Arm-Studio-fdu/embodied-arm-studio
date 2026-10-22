"""Canonical persistent Evaluation reports."""

from robotics.evaluations.models import EvaluationReport, EvaluationStatus
from robotics.evaluations.comparison import (
    CanonicalJSONExport,
    CanonicalRunComparisonExportService,
    CanonicalRunFactsNotFound,
    CanonicalRunFactsUnavailable,
)
from robotics.evaluations.repository import (
    EvaluationRepository,
    EvaluationRepositoryError,
    EvaluationReportNotFound,
    PostgreSQLEvaluationRepository,
)
from robotics.evaluations.service import (
    EVALUATOR_VERSION,
    EvaluationCreateResult,
    EvaluationNotAllowed,
    EvaluationNotFound,
    EvaluationService,
    EvaluationUnavailable,
    canonical_evidence_json,
    evidence_sha256,
)

__all__ = [
    "EVALUATOR_VERSION",
    "CanonicalJSONExport",
    "CanonicalRunComparisonExportService",
    "CanonicalRunFactsNotFound",
    "CanonicalRunFactsUnavailable",
    "EvaluationCreateResult",
    "EvaluationNotAllowed",
    "EvaluationNotFound",
    "EvaluationReport",
    "EvaluationReportNotFound",
    "EvaluationRepository",
    "EvaluationRepositoryError",
    "EvaluationService",
    "EvaluationStatus",
    "EvaluationUnavailable",
    "PostgreSQLEvaluationRepository",
    "canonical_evidence_json",
    "evidence_sha256",
]
