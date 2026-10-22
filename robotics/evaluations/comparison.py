"""Read-only canonical Run comparison and JSON export projections."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qsl, quote, urlsplit

from robotics.evaluations.models import EvaluationReport
from robotics.evaluations.repository import EvaluationRepositoryError
from robotics.runs.models import Run, RunEvent
from robotics.runs.repository import RepositoryError, RunNotFound
from robotics.run_access import RunAccessNotFound, RunAccessUnavailable


SCHEMA_VERSION = "1.0"
_SENSITIVE_KEY = re.compile(
    r"(^|_)(password|passwd|secret|token|api_key|database_url|dsn)(_|$)",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_DATABASE_SCHEMES = {
    "postgres",
    "postgresql",
    "postgresql+psycopg",
    "redis",
    "rediss",
    "file",
}


class CanonicalRunFactsNotFound(LookupError):
    """One or more requested Runs are unknown or inaccessible."""


class CanonicalRunFactsUnavailable(RuntimeError):
    """Canonical comparison/export persistence is unavailable."""


@dataclass(frozen=True)
class CanonicalJSONExport:
    content: bytes
    etag: str
    filename: str = "runs-export.json"


@dataclass(frozen=True)
class _RunFacts:
    run: Run
    events: Sequence[RunEvent]
    artifacts: Sequence[Any]
    evaluations: Sequence[EvaluationReport]


class CanonicalRunComparisonExportService:
    """Aggregate PostgreSQL canonical facts without changing business data."""

    def __init__(
        self,
        run_service: Any,
        artifact_metadata_store: Any,
        evaluation_repository: Any,
        run_access_service: Optional[Any] = None,
    ) -> None:
        self._run_service = run_service
        self._artifact_metadata_store = artifact_metadata_store
        self._evaluation_repository = evaluation_repository
        self._run_access_service = run_access_service

    def compare(self, run_ids: Sequence[str], trusted_user_id: str) -> Dict[str, Any]:
        facts = self._load_facts(run_ids, trusted_user_id, include_events=False)
        project_ids = list(dict.fromkeys(item.run.project_id for item in facts))
        run_items = [self._comparison_item(item) for item in facts]
        rubric_versions = [
            item["evaluation"]["rubric_version"]
            if item["evaluation"] is not None else None
            for item in run_items
        ]
        available_rubrics = [value for value in rubric_versions if value is not None]
        rubric_versions_comparable = (
            len(available_rubrics) == len(rubric_versions)
            and len(set(available_rubrics)) == 1
        )
        limitations = [
            "Scores and grades do not by themselves establish algorithm superiority.",
            "Completed, failed, and stopped statuses are reported neutrally and are not quality rankings.",
            "Missing metrics and Evaluation values remain null; they are never replaced with zero.",
            "Artifact comparison covers registered metadata completeness only; Artifact files are not read.",
        ]
        if len(project_ids) > 1:
            limitations.append(
                "Cross-project comparison is allowed, but project configuration differences limit direct attribution."
            )
        if not rubric_versions_comparable:
            limitations.append(
                "Evaluation results are not directly comparable because rubric versions differ or an Evaluation is missing."
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "run_ids": list(run_ids),
            "cross_project": len(project_ids) > 1,
            "project_ids": project_ids,
            "runs": run_items,
            "differences": {
                "parameters": self._mapping_differences(
                    [item["parameters"] for item in run_items]
                ),
                "config_versions": {
                    "values": [
                        item["project_config_snapshot_summary"]["config_version"]
                        for item in run_items
                    ],
                    "different": self._different(
                        [
                            item["project_config_snapshot_summary"]["config_version"]
                            for item in run_items
                        ]
                    ),
                },
                "metrics": self._mapping_differences(
                    [item["metrics_summary"] for item in run_items]
                ),
                "statuses": {
                    "values": [item["status"] for item in run_items],
                    "different": self._different(
                        [item["status"] for item in run_items]
                    ),
                },
                "training_steps": {
                    "values": [item["training_steps"] for item in run_items],
                    "different": self._different(
                        [item["training_steps"] for item in run_items]
                    ),
                },
                "update_steps": {
                    "values": [item["update_steps"] for item in run_items],
                    "different": self._different(
                        [item["update_steps"] for item in run_items]
                    ),
                },
                "artifacts": {
                    "values": [item["artifacts"] for item in run_items],
                    "different": self._different(
                        [item["artifacts"] for item in run_items]
                    ),
                },
                "evaluations": {
                    "values": [item["evaluation"] for item in run_items],
                    "different": self._different(
                        [item["evaluation"] for item in run_items]
                    ),
                    "rubric_versions_comparable": rubric_versions_comparable,
                },
            },
            "limitations": limitations,
        }

    def export_json(
        self, run_ids: Sequence[str], trusted_user_id: str
    ) -> CanonicalJSONExport:
        facts = self._load_facts(run_ids, trusted_user_id, include_events=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "runs": [self._export_item(item) for item in facts],
            "limitations": [
                "This export contains canonical metadata and structured facts only.",
                "Artifact files, Checkpoints, logs, and full metrics files are not embedded.",
                "Event sequence gaps are preserved and are not synthesized.",
                "Evaluation results reflect their persisted rubric and evidence versions; no Evaluation was rerun.",
            ],
        }
        content = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return CanonicalJSONExport(
            content=content,
            etag=hashlib.sha256(content).hexdigest(),
        )

    def _load_facts(
        self,
        run_ids: Sequence[str],
        trusted_user_id: str,
        *,
        include_events: bool,
    ) -> List[_RunFacts]:
        loaded: List[_RunFacts] = []
        try:
            for run_id in run_ids:
                if self._run_access_service is None:
                    run = self._run_service.get_run(run_id)
                    if run.user_id != trusted_user_id:
                        raise CanonicalRunFactsNotFound()
                else:
                    try:
                        run = self._run_access_service.read(
                            run_id, trusted_user_id
                        )
                    except Exception as exc:
                        if isinstance(exc, RunAccessNotFound):
                            raise CanonicalRunFactsNotFound() from exc
                        if isinstance(exc, RunAccessUnavailable):
                            raise CanonicalRunFactsUnavailable(
                                "Canonical Run facts are temporarily unavailable"
                            ) from exc
                        raise
                events = (
                    self._run_service.list_events(run_id)
                    if include_events else ()
                )
                artifacts = self._artifact_metadata_store.list_artifacts(run_id)
                evaluations = self._evaluation_repository.list_for_run(run_id)
                loaded.append(
                    _RunFacts(
                        run=run,
                        events=tuple(events),
                        artifacts=tuple(artifacts),
                        evaluations=tuple(evaluations),
                    )
                )
        except CanonicalRunFactsNotFound:
            raise
        except RunNotFound as exc:
            raise CanonicalRunFactsNotFound() from exc
        except (RepositoryError, EvaluationRepositoryError) as exc:
            raise CanonicalRunFactsUnavailable(
                "Canonical Run facts are temporarily unavailable"
            ) from exc
        return loaded

    def _comparison_item(self, facts: _RunFacts) -> Dict[str, Any]:
        run = facts.run
        return {
            "run_id": run.run_id,
            "project_id": run.project_id,
            "class_id": run.class_id,
            "status": run.status.value,
            "parameters": _sanitize(run.parameters),
            "project_config_snapshot_summary": self._config_summary(
                run.project_config_snapshot
            ),
            "code_sha256": run.code_sha256,
            "progress": run.progress,
            "metrics_summary": _sanitize(run.metrics_summary),
            "training_steps": run.training_steps,
            "update_steps": run.update_steps,
            "error": _safe_error(run.error),
            "stop_reason": run.stop_reason,
            "artifacts": self._artifact_summary(facts.artifacts),
            "evaluation": (
                self._evaluation_summary(facts.evaluations[0])
                if facts.evaluations else None
            ),
        }

    def _export_item(self, facts: _RunFacts) -> Dict[str, Any]:
        return {
            "run": _run_snapshot(facts.run),
            "events": [
                _event_snapshot(event)
                for event in sorted(facts.events, key=lambda item: item.sequence)
            ],
            "artifacts": [
                _artifact_export(artifact)
                for artifact in sorted(
                    facts.artifacts,
                    key=lambda item: (item.artifact_type, item.artifact_id),
                )
            ],
            "evaluations": [
                _evaluation_export(report)
                for report in reversed(facts.evaluations)
            ],
        }

    @staticmethod
    def _config_summary(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        safe_snapshot = _sanitize(snapshot)
        encoded = json.dumps(
            safe_snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        version = snapshot.get("config_version", snapshot.get("version"))
        if not isinstance(version, (str, int, float)) or isinstance(version, bool):
            version = None
        return {
            "config_version": version,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "keys": sorted(str(key) for key in snapshot),
        }

    @staticmethod
    def _artifact_summary(artifacts: Sequence[Any]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for artifact in artifacts:
            counts[artifact.artifact_type] = counts.get(artifact.artifact_type, 0) + 1
        return {
            "total_count": len(artifacts),
            "counts_by_type": {key: counts[key] for key in sorted(counts)},
            "missing_size_count": sum(
                artifact.size_bytes is None for artifact in artifacts
            ),
            "missing_sha256_count": sum(
                artifact.sha256 is None for artifact in artifacts
            ),
            "primary_model_count": sum(
                artifact.is_primary_model for artifact in artifacts
            ),
        }

    @staticmethod
    def _evaluation_summary(report: EvaluationReport) -> Dict[str, Any]:
        return {
            "evaluation_id": report.evaluation_id,
            "status": report.status.value,
            "score": report.score,
            "grade": report.grade,
            "rubric_version": report.rubric_version,
            "evidence_sha256": report.evidence_sha256,
            "limitations": deepcopy(report.limitations),
            "created_at": _utc_text(report.created_at),
        }

    @staticmethod
    def _mapping_differences(values: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        keys = sorted({str(key) for value in values for key in value})
        return {
            key: {
                "values": [deepcopy(value.get(key)) for value in values],
                "different": CanonicalRunComparisonExportService._different(
                    [value.get(key) for value in values]
                ),
            }
            for key in keys
        }

    @staticmethod
    def _different(values: Sequence[Any]) -> bool:
        normalized = [
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for value in values
        ]
        return len(set(normalized)) > 1


def _run_snapshot(run: Run) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "user_id": run.user_id,
        "project_id": run.project_id,
        "class_id": run.class_id,
        "status": run.status.value,
        "execution_mode": run.execution_mode.value,
        "run_kind": run.run_kind.value,
        "parameters": _sanitize(run.parameters),
        "project_config_snapshot": _sanitize(run.project_config_snapshot),
        "code_sha256": run.code_sha256,
        "simulation_task_id": run.simulation_task_id,
        "progress": run.progress,
        "metrics_summary": _sanitize(run.metrics_summary),
        "error": _safe_error(run.error),
        "last_sequence": run.last_sequence,
        "training_steps": run.training_steps,
        "update_steps": run.update_steps,
        "stop_reason": run.stop_reason,
        "created_at": _utc_text(run.created_at),
        "started_at": _utc_text(run.started_at),
        "ended_at": _utc_text(run.ended_at),
        "updated_at": _utc_text(run.updated_at),
    }


def _event_snapshot(event: RunEvent) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "sequence": event.sequence,
        "type": event.type.value,
        "run_id": event.run_id,
        "task_id": event.task_id,
        "occurred_at": _utc_text(event.occurred_at),
        "schema_version": event.schema_version,
        "payload": _sanitize(event.payload),
    }


def _artifact_export(artifact: Any) -> Dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "run_id": artifact.run_id,
        "artifact_type": artifact.artifact_type,
        "logical_path": artifact.path,
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
        "is_primary_model": artifact.is_primary_model,
        "created_at": _utc_text(artifact.created_at),
        "download_path": "/api/v1/runs/%s/artifacts/%s/download"
        % (quote(artifact.run_id, safe=""), quote(artifact.artifact_id, safe="")),
    }


def _evaluation_export(report: EvaluationReport) -> Dict[str, Any]:
    return {
        "evaluation_id": report.evaluation_id,
        "run_id": report.run_id,
        "status": report.status.value,
        "score": report.score,
        "grade": report.grade,
        "rubric_version": report.rubric_version,
        "rubric_snapshot": _sanitize(report.rubric_snapshot),
        "evidence": _sanitize(report.evidence),
        "evidence_sha256": report.evidence_sha256,
        "summary": _sanitize(report.summary),
        "limitations": deepcopy(report.limitations),
        "evaluator_version": report.evaluator_version,
        "created_at": _utc_text(report.created_at),
    }


def _safe_error(error: Any) -> Optional[Dict[str, Any]]:
    if error is None:
        return None
    return {
        "code": error.code,
        "message": "Run reported an error",
        "retryable": error.retryable,
    }


def _sanitize(value: Any, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[redacted]"
    if key.lower() == "message" and isinstance(value, str):
        return "Structured message redacted"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(item_value, str(item_key))
            for item_key, item_value in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, key) for item in value]
    if isinstance(value, str) and _unsafe_string(value):
        return "[redacted]"
    return deepcopy(value)


def _unsafe_string(value: str) -> bool:
    lowered = value.lower()
    if "bearer " in lowered or "token=" in lowered:
        return True
    if value.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(value):
        return True
    try:
        parsed = urlsplit(value)
    except ValueError:
        return True
    if parsed.scheme.lower() in _DATABASE_SCHEMES:
        return True
    if parsed.username is not None or parsed.password is not None:
        return True
    if any(
        _SENSITIVE_KEY.search(name)
        for name, unused_value in parse_qsl(parsed.query, keep_blank_values=True)
    ):
        return True
    return False


def _utc_text(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
