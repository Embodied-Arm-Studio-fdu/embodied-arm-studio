"""Deterministic Evaluation from frozen Run and Artifact metadata evidence."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from robotics.evaluations.models import EvaluationReport, EvaluationStatus
from robotics.evaluations.repository import (
    EvaluationReportNotFound,
    EvaluationRepository,
    EvaluationRepositoryError,
)
from robotics.runs.domain import TERMINAL_STATUSES
from robotics.runs.repository import RepositoryError, RunNotFound
from robotics.run_access import (
    RunAccessForbidden,
    RunAccessNotFound,
    RunAccessUnavailable,
)


EVALUATOR_VERSION = "canonical-rule-evaluator/1.0"
SUPPORTED_OPERATORS = {"gte", "lte", "eq"}


class EvaluationNotFound(LookupError):
    """Unknown or inaccessible Run/Evaluation."""


class EvaluationNotAllowed(ValueError):
    """The Run is not eligible for Evaluation."""


class EvaluationUnavailable(RuntimeError):
    """A redacted persistence or evidence lookup failure."""


@dataclass(frozen=True)
class EvaluationCreateResult:
    report: EvaluationReport
    replayed: bool


def canonical_evidence_json(evidence: Dict[str, Any]) -> str:
    return json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def evidence_sha256(evidence: Dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_evidence_json(evidence).encode("utf-8")
    ).hexdigest()


class EvaluationService:
    """Create immutable reports without reading Artifact file contents."""

    def __init__(
        self,
        run_service: Any,
        artifact_metadata_store: Any,
        repository: EvaluationRepository,
        run_access_service: Optional[Any] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        evaluation_id_factory: Callable[[], str] = lambda: "evaluation-%s"
        % uuid4().hex,
    ) -> None:
        self._run_service = run_service
        self._artifact_metadata_store = artifact_metadata_store
        self._repository = repository
        self._run_access_service = run_access_service
        self._clock = clock
        self._evaluation_id_factory = evaluation_id_factory

    def create(self, run_id: str, trusted_user_id: str) -> EvaluationCreateResult:
        run = self._authorized_run(run_id, trusted_user_id, create=True)
        if run.status not in TERMINAL_STATUSES:
            raise EvaluationNotAllowed("Run must be terminal before evaluation")
        try:
            artifacts = self._artifact_metadata_store.list_artifacts(run_id)
        except (RepositoryError, EvaluationRepositoryError) as exc:
            raise EvaluationUnavailable(
                "Evaluation evidence is temporarily unavailable"
            ) from exc
        evidence = self._build_evidence(run, artifacts)
        try:
            fingerprint = evidence_sha256(evidence)
        except (TypeError, ValueError) as exc:
            raise EvaluationUnavailable("Evaluation evidence is invalid") from exc
        rubric = deepcopy(run.project_config_snapshot.get("evaluation_rubric", {}))
        report = self._evaluate(
            run_id=run.run_id,
            requested_by_user_id=trusted_user_id,
            rubric=rubric,
            evidence=evidence,
            fingerprint=fingerprint,
        )
        try:
            persisted, replayed = self._repository.create_or_get(report)
        except EvaluationRepositoryError as exc:
            raise EvaluationUnavailable(
                "Evaluation persistence is temporarily unavailable"
            ) from exc
        return EvaluationCreateResult(report=persisted, replayed=replayed)

    def get(self, evaluation_id: str, trusted_user_id: str) -> EvaluationReport:
        try:
            report = self._repository.get(evaluation_id)
        except EvaluationReportNotFound as exc:
            raise EvaluationNotFound("Evaluation not found") from exc
        except EvaluationRepositoryError as exc:
            raise EvaluationUnavailable(
                "Evaluation lookup is temporarily unavailable"
            ) from exc
        self._authorized_run(report.run_id, trusted_user_id, create=False)
        return report

    def list_for_run(
        self, run_id: str, trusted_user_id: str
    ) -> List[EvaluationReport]:
        """Return persisted history without evaluating or changing the Run."""
        self._authorized_run(run_id, trusted_user_id, create=False)
        method = getattr(self._repository, "list_for_run", None)
        if not callable(method):
            raise EvaluationUnavailable(
                "Evaluation history is not configured"
            )
        try:
            return method(run_id)
        except EvaluationRepositoryError as exc:
            raise EvaluationUnavailable(
                "Evaluation history is temporarily unavailable"
            ) from exc

    def _authorized_run(
        self, run_id: str, trusted_user_id: str, *, create: bool
    ):
        if self._run_access_service is not None:
            try:
                method = (
                    self._run_access_service.create_evaluation
                    if create else self._run_access_service.read
                )
                return method(run_id, trusted_user_id)
            except Exception as exc:
                if isinstance(exc, RunAccessForbidden):
                    if create:
                        raise EvaluationNotAllowed(
                            "Evaluation creation is forbidden"
                        ) from exc
                    raise EvaluationNotFound("Evaluation not found") from exc
                if isinstance(exc, RunAccessNotFound):
                    raise EvaluationNotFound("Evaluation not found") from exc
                if isinstance(exc, RunAccessUnavailable):
                    raise EvaluationUnavailable(
                        "Evaluation Run lookup is temporarily unavailable"
                    ) from exc
                raise
        try:
            run = self._run_service.get_run(run_id)
        except RunNotFound as exc:
            raise EvaluationNotFound("Evaluation not found") from exc
        except RepositoryError as exc:
            raise EvaluationUnavailable(
                "Evaluation Run lookup is temporarily unavailable"
            ) from exc
        if run.user_id != trusted_user_id:
            raise EvaluationNotFound("Evaluation not found")
        return run

    @staticmethod
    def _build_evidence(run, artifacts) -> Dict[str, Any]:
        error = None
        if run.error is not None:
            error = {
                "code": run.error.code,
                "message": run.error.message,
                "retryable": run.error.retryable,
            }
        artifact_items = sorted(
            (
                {
                    "artifact_id": item.artifact_id,
                    "type": item.artifact_type,
                    "path": item.path,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in artifacts
            ),
            key=lambda item: (item["path"], item["artifact_id"]),
        )
        return deepcopy(
            {
                "run": {
                    "run_id": run.run_id,
                    "project_id": run.project_id,
                    "project_config_snapshot": run.project_config_snapshot,
                    "parameters": run.parameters,
                    "seed": run.parameters.get("seed"),
                    "code_sha256": run.code_sha256,
                    "status": run.status.value,
                    "metrics_summary": run.metrics_summary,
                    "training_steps": run.training_steps,
                    "update_steps": run.update_steps,
                    "error": error,
                    "stop_reason": run.stop_reason,
                },
                "artifacts": artifact_items,
            }
        )

    def _evaluate(
        self,
        *,
        run_id: str,
        requested_by_user_id: str,
        rubric: Dict[str, Any],
        evidence: Dict[str, Any],
        fingerprint: str,
    ) -> EvaluationReport:
        limitations: List[str] = []
        if not evidence["artifacts"]:
            limitations.append("No registered Artifact metadata is available.")
        valid, rubric_version, validation_limitations = self._validate_rubric(rubric)
        limitations.extend(validation_limitations)
        criteria_summary: List[Dict[str, Any]] = []
        score: Optional[float] = None
        grade = "insufficient_evidence"
        status = EvaluationStatus.INSUFFICIENT_EVIDENCE
        if valid:
            earned = 0.0
            total = 0.0
            for criterion in rubric["criteria"]:
                found, actual = self._evidence_value(
                    evidence, criterion["evidence_path"]
                )
                if not found:
                    limitations.append(
                        "Missing evidence: %s" % criterion["evidence_path"]
                    )
                    criteria_summary.append(
                        {
                            "criterion_id": criterion["criterion_id"],
                            "status": "insufficient_evidence",
                            "evidence_path": criterion["evidence_path"],
                        }
                    )
                    continue
                met = self._compare(
                    actual, criterion["operator"], criterion["target"]
                )
                if met is None:
                    limitations.append(
                        "Incompatible evidence type: %s"
                        % criterion["evidence_path"]
                    )
                    criteria_summary.append(
                        {
                            "criterion_id": criterion["criterion_id"],
                            "status": "insufficient_evidence",
                            "evidence_path": criterion["evidence_path"],
                        }
                    )
                    continue
                weight = float(criterion["weight"])
                total += weight
                if met:
                    earned += weight
                criteria_summary.append(
                    {
                        "criterion_id": criterion["criterion_id"],
                        "status": "assessed",
                        "evidence_path": criterion["evidence_path"],
                        "operator": criterion["operator"],
                        "target": deepcopy(criterion["target"]),
                        "actual": deepcopy(actual),
                        "met": met,
                        "weight": weight,
                    }
                )
            if total > 0 and all(
                item["status"] == "assessed" for item in criteria_summary
            ):
                score = round(100.0 * earned / total, 2)
                grade = self._grade(score, rubric["grade_bands"])
                status = EvaluationStatus.ASSESSED
        summary = {
            "result": status.value,
            "criteria": criteria_summary,
            "explanation": (
                "Score is the weighted percentage of explicitly satisfied criteria."
                if status == EvaluationStatus.ASSESSED
                else "No score was produced because the frozen rubric or evidence is insufficient."
            ),
        }
        return EvaluationReport(
            evaluation_id=self._evaluation_id_factory(),
            run_id=run_id,
            requested_by_user_id=requested_by_user_id,
            status=status,
            score=score,
            grade=grade,
            rubric_version=rubric_version,
            rubric_snapshot=rubric,
            evidence=evidence,
            evidence_sha256=fingerprint,
            summary=summary,
            limitations=limitations,
            evaluator_version=EVALUATOR_VERSION,
            created_at=self._clock(),
        )

    @staticmethod
    def _validate_rubric(
        rubric: Any,
    ) -> Tuple[bool, str, List[str]]:
        if not isinstance(rubric, dict) or not rubric:
            return False, "missing", ["Frozen evaluation_rubric is missing."]
        version = rubric.get("version")
        if not isinstance(version, str) or not version.strip():
            return False, "unsupported", ["Rubric version is missing."]
        criteria = rubric.get("criteria")
        bands = rubric.get("grade_bands")
        if not isinstance(criteria, list) or not criteria:
            return False, version, ["Rubric criteria are missing."]
        if not isinstance(bands, list) or not bands:
            return False, version, ["Rubric grade bands are missing."]
        seen = set()
        for criterion in criteria:
            if not isinstance(criterion, dict):
                return False, version, ["Rubric criterion is invalid."]
            criterion_id = criterion.get("criterion_id")
            path = criterion.get("evidence_path")
            operator = criterion.get("operator")
            weight = criterion.get("weight")
            if (
                not isinstance(criterion_id, str)
                or not criterion_id.strip()
                or criterion_id in seen
                or not isinstance(path, str)
                or not path.strip()
                or operator not in SUPPORTED_OPERATORS
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight <= 0
                or "target" not in criterion
            ):
                detail = (
                    "Unknown rubric operator: %s" % operator
                    if operator not in SUPPORTED_OPERATORS
                    else "Rubric criterion is invalid."
                )
                return False, version, [detail]
            seen.add(criterion_id)
        valid_bands = []
        for band in bands:
            if not isinstance(band, dict):
                return False, version, ["Rubric grade band is invalid."]
            minimum = band.get("minimum_score")
            grade = band.get("grade")
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, (int, float))
                or not math.isfinite(minimum)
                or not 0 <= minimum <= 100
                or not isinstance(grade, str)
                or not grade.strip()
            ):
                return False, version, ["Rubric grade band is invalid."]
            valid_bands.append(float(minimum))
        if min(valid_bands) != 0:
            return False, version, ["Rubric grade bands must cover score zero."]
        return True, version, []

    @staticmethod
    def _evidence_value(evidence: Dict[str, Any], path: str):
        value: Any = evidence
        for component in path.split("."):
            if not component or not isinstance(value, dict) or component not in value:
                return False, None
            value = value[component]
        return True, value

    @staticmethod
    def _compare(actual: Any, operator: str, target: Any) -> Optional[bool]:
        if operator == "eq":
            return actual == target
        if (
            isinstance(actual, bool)
            or isinstance(target, bool)
            or not isinstance(actual, (int, float))
            or not isinstance(target, (int, float))
            or not math.isfinite(actual)
            or not math.isfinite(target)
        ):
            return None
        return actual >= target if operator == "gte" else actual <= target

    @staticmethod
    def _grade(score: float, bands: List[Dict[str, Any]]) -> str:
        ordered = sorted(
            bands, key=lambda item: float(item["minimum_score"]), reverse=True
        )
        return next(
            item["grade"] for item in ordered if score >= item["minimum_score"]
        )
