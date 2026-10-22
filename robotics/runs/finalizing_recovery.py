"""Fail-closed evidence checks for controlled stale-finalizing recovery."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from robotics.runs.domain import RunStatus


class FinalizingRecoveryError(RuntimeError):
    """Recovery evidence is incomplete or conflicts with persisted Run facts."""


@dataclass(frozen=True)
class FinalizingRecoveryPlan:
    run_id: str
    task_id: str
    callback: Dict[str, Any]
    verified_artifacts: Tuple[str, ...]


def plan_finalizing_recovery(
    run,
    task: Mapping[str, Any],
    outbox_record: Mapping[str, Any],
    artifact_directory: Path,
    *,
    now: Optional[datetime] = None,
    minimum_age: timedelta = timedelta(minutes=10),
) -> FinalizingRecoveryPlan:
    """Verify immutable terminal evidence without changing PostgreSQL or Redis."""

    current = now or datetime.now(timezone.utc)
    if run.status not in {
        RunStatus.FINALIZING, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.STOPPED
    }:
        raise FinalizingRecoveryError("Run is neither finalizing nor terminal")
    if run.status is RunStatus.FINALIZING and current - run.updated_at < minimum_age:
        raise FinalizingRecoveryError("Run is inside the finalizing timeout window")
    if not run.simulation_task_id or task.get("task_id") != run.simulation_task_id:
        raise FinalizingRecoveryError("Simulation task identity does not match Run")
    callback = deepcopy(outbox_record.get("callback"))
    if not isinstance(callback, dict):
        raise FinalizingRecoveryError("Callback outbox payload is missing")
    if (
        callback.get("run_id") != run.run_id
        or callback.get("task_id") != run.simulation_task_id
        or callback.get("event_id") != outbox_record.get("event_id")
    ):
        raise FinalizingRecoveryError("Callback identity does not match Run and outbox")
    terminal_status = task.get("status")
    if terminal_status not in {"completed", "failed", "stopped"}:
        raise FinalizingRecoveryError("Simulation task is not terminal")
    if callback.get("status") != terminal_status:
        raise FinalizingRecoveryError("Callback status does not match Simulation task")
    if run.status is not RunStatus.FINALIZING and run.status.value != callback.get("status"):
        raise FinalizingRecoveryError("terminal Run status conflicts with Callback")

    root = artifact_directory.resolve()
    verified = []
    for item in callback.get("artifacts", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise FinalizingRecoveryError("Callback Artifact metadata is malformed")
        candidate = (root / item["path"]).resolve()
        if root not in candidate.parents or not candidate.is_file():
            raise FinalizingRecoveryError("Callback Artifact is missing: %s" % item["path"])
        if item.get("size_bytes") is not None and candidate.stat().st_size != item["size_bytes"]:
            raise FinalizingRecoveryError("Callback Artifact size differs: %s" % item["path"])
        if item.get("sha256") is not None:
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise FinalizingRecoveryError("Callback Artifact digest differs: %s" % item["path"])
        verified.append(item["path"])

    if terminal_status == "completed" and callback.get("evaluation") is None:
        summary = next(
            (item for item in callback.get("artifacts", []) if item.get("type") == "evaluation_summary"),
            None,
        )
        if summary is None:
            raise FinalizingRecoveryError("Completed training evidence has no evaluation summary")
        values = json.loads((root / summary["path"]).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise FinalizingRecoveryError("Evaluation summary is malformed")
        interval = values.get("confidence_interval_95")
        if not isinstance(interval, dict):
            raise FinalizingRecoveryError("Evaluation confidence interval is missing")
        passed = (
            values.get("episode_count", 0) >= 100
            and values.get("success_rate", 0) >= 0.80
            and interval.get("lower", 0) >= 0.70
        )
        evaluation = {
            "status": "passed" if passed else "failed",
            "deterministic": True,
            "policy_passed": passed,
            "episode_count": values["episode_count"],
            "success_count": values["success_count"],
            "success_rate": values["success_rate"],
            "wilson_95_lower_bound": interval["lower"],
            "wilson_95_upper_bound": interval["upper"],
        }
        for name in (
            "mean_episode_length", "mean_episode_return", "mean_final_distance",
            "mean_minimum_distance", "mean_object_displacement", "timeout_rate",
            "failure_rate",
        ):
            if name in values:
                evaluation[name] = values[name]
        callback["evaluation"] = evaluation

    return FinalizingRecoveryPlan(
        run_id=run.run_id,
        task_id=run.simulation_task_id,
        callback=callback,
        verified_artifacts=tuple(sorted(verified)),
    )

