"""Compatibility projections between canonical Runs and legacy contracts."""

from copy import deepcopy
from typing import Any, Dict, Optional

from robotics.contracts import (
    RunEvent as LegacyRunEvent,
    RunRecord as LegacyRunRecord,
    RunRequest as LegacyRunRequest,
    RunStatus as LegacyRunStatus,
)
from robotics.runs.domain import RunStatus
from robotics.runs.models import ExecutionMode, Run, RunEvent, RunEventType, RunKind, SCHEMA_VERSION


class LegacyCompatibilityError(ValueError):
    """Raised when canonical and legacy contracts cannot be converted safely."""


_LEGACY_STATUS_BY_CANONICAL = {
    RunStatus.QUEUED: LegacyRunStatus.QUEUED,
    RunStatus.INITIALIZING: LegacyRunStatus.QUEUED,
    RunStatus.RUNNING: LegacyRunStatus.RUNNING,
    RunStatus.FINALIZING: LegacyRunStatus.RUNNING,
    RunStatus.STOPPING: LegacyRunStatus.RUNNING,
    RunStatus.COMPLETED: LegacyRunStatus.COMPLETED,
    RunStatus.FAILED: LegacyRunStatus.FAILED,
    RunStatus.STOPPED: LegacyRunStatus.STOPPED,
}


def to_legacy_status(status: Any) -> LegacyRunStatus:
    """Project one canonical eight-state status onto the legacy five states."""

    try:
        canonical_status = RunStatus(status)
        return _LEGACY_STATUS_BY_CANONICAL[canonical_status]
    except (KeyError, TypeError, ValueError) as exc:
        raise LegacyCompatibilityError(
            "unknown canonical Run status: %r" % (status,)
        ) from exc


def legacy_request_to_create_command(
    request: LegacyRunRequest,
    *,
    user_id: str,
    project_id: str,
    execution_mode: ExecutionMode,
    run_kind: RunKind = RunKind.TRAIN,
    code: str = "",
) -> Dict[str, Any]:
    """Convert a legacy request into keyword arguments for RunService.create_run."""

    if not isinstance(request, LegacyRunRequest):
        raise LegacyCompatibilityError("request must be a legacy RunRequest")
    if not isinstance(user_id, str) or not user_id.strip():
        raise LegacyCompatibilityError("user_id must be provided by the caller")
    if not isinstance(project_id, str) or not project_id.strip():
        raise LegacyCompatibilityError("project_id must be provided by the caller")
    if not isinstance(execution_mode, ExecutionMode):
        raise LegacyCompatibilityError(
            "execution_mode must be provided as an ExecutionMode"
        )
    if not isinstance(run_kind, RunKind):
        raise LegacyCompatibilityError("run_kind must be provided as a RunKind")
    if not isinstance(code, str):
        raise LegacyCompatibilityError("code must be a string")
    if not isinstance(request.environment_id, str) or not request.environment_id:
        raise LegacyCompatibilityError("legacy request is missing environment_id")
    if not isinstance(request.strategy_id, str) or not request.strategy_id:
        raise LegacyCompatibilityError("legacy request is missing strategy_id")
    if not isinstance(request.config, dict):
        raise LegacyCompatibilityError("legacy request config must be a dict")

    return {
        "user_id": user_id,
        "project_id": project_id,
        "execution_mode": execution_mode,
        "run_kind": run_kind,
        "parameters": {
            "seed": request.seed,
            "max_steps": request.max_steps,
            "config": deepcopy(request.config),
        },
        "project_config_snapshot": {
            "environment_id": request.environment_id,
            "strategy_id": request.strategy_id,
            "config": deepcopy(request.config),
        },
        "code": code,
    }


def to_legacy_run_record(run: Run) -> LegacyRunRecord:
    """Project a canonical Run into the exact legacy RunRecord shape."""

    if not isinstance(run, Run):
        raise LegacyCompatibilityError("run must be a canonical Run")
    seed = _required_field(run.parameters, "seed", "parameters")
    snapshot = run.project_config_snapshot
    if any(name in snapshot for name in ("environment_id", "strategy_id", "config")):
        max_steps = _required_field(run.parameters, "max_steps", "parameters")
        environment_id = _required_field(
            snapshot, "environment_id", "project_config_snapshot"
        )
        strategy_id = _required_field(
            snapshot, "strategy_id", "project_config_snapshot"
        )
        config = _required_field(snapshot, "config", "project_config_snapshot")
    else:
        max_steps = _required_field(
            snapshot, "expected_training_steps", "project_config_snapshot"
        )
        environment_id = _required_field(
            snapshot, "environment_type", "project_config_snapshot"
        )
        strategy_id = _required_field(
            snapshot, "algorithm", "project_config_snapshot"
        )
        config = _required_field(
            snapshot, "training_config", "project_config_snapshot"
        )
    if not isinstance(config, dict):
        raise LegacyCompatibilityError(
            "project_config_snapshot.config must be a dict"
        )

    error = run.error.message if run.error is not None else None
    exit_code = None
    if run.status in {RunStatus.COMPLETED, RunStatus.STOPPED}:
        exit_code = 0
    elif run.status == RunStatus.FAILED:
        exit_code = 1
    return LegacyRunRecord(
        run_id=run.run_id,
        mode=run.run_kind.value,
        environment_id=environment_id,
        strategy_id=strategy_id,
        seed=seed,
        max_steps=max_steps,
        config=deepcopy(config),
        status=to_legacy_status(run.status),
        started_at=_legacy_timestamp(run.started_at),
        ended_at=_legacy_timestamp(run.ended_at),
        exit_code=exit_code,
        error=error,
    )


def to_legacy_run_event(event: RunEvent) -> Optional[LegacyRunEvent]:
    """Convert a canonical Event, returning None for legacy-hidden heartbeats."""

    if not isinstance(event, RunEvent):
        raise LegacyCompatibilityError("event must be a canonical RunEvent")
    if event.schema_version != SCHEMA_VERSION:
        raise LegacyCompatibilityError(
            "unsupported canonical Event schema_version: %s" % event.schema_version
        )
    if event.type == RunEventType.HEARTBEAT:
        return None

    payload = deepcopy(event.payload)
    if event.type == RunEventType.STATE:
        status = _required_field(payload, "status", "state payload")
        previous_status = _required_field(
            payload, "previous_status", "state payload"
        )
        payload["status"] = to_legacy_status(status).value
        payload["previous_status"] = to_legacy_status(previous_status).value
        legacy_type = "status"
    elif event.type == RunEventType.METRIC:
        legacy_type = "metric"
    elif event.type == RunEventType.ERROR:
        legacy_type = "error"
    else:
        raise LegacyCompatibilityError(
            "cannot convert canonical Event type: %r" % event.type
        )

    return LegacyRunEvent(
        run_id=event.run_id,
        type=legacy_type,
        payload=payload,
        event_id=event.event_id,
        timestamp=event.occurred_at.isoformat(),
        version=1,
    )


def _required_field(values: Dict[str, Any], name: str, owner: str) -> Any:
    if name not in values:
        raise LegacyCompatibilityError(
            "%s is missing required legacy field %s" % (owner, name)
        )
    return values[name]


def _legacy_timestamp(value: Any) -> Optional[str]:
    return value.isoformat() if value is not None else None
