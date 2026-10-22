"""Internal Simulation Event ingestion router."""

import hmac
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from apps.api.schemas.internal_runs import (
    InternalRunEventRequest,
    InternalRunEventResponse,
    TerminalCallbackRequest,
    TerminalCallbackResponse,
)
from robotics.runs import (
    ApplyEventResult,
    EventConflict,
    EventSequenceConflict,
    InvalidRunEvent,
    InvalidRunStateTransition,
    RepositoryDataError,
    RepositoryIntegrityError,
    RunEvent,
    RunEventType,
    RunKind,
    RunNotFound,
    RunService,
    RunStatus as CanonicalRunStatus,
    RunTaskMismatch,
    TerminalRunStatusConflict,
)
from robotics.runs.artifacts import ArtifactRegistration
from robotics.runs.models import RunError
from robotics.runs.postgres_callback import (
    ArtifactMetadataConflict,
    PostgreSQLTerminalCallbackService,
    TerminalCallbackCommand,
    TerminalCallbackConflict,
)


def create_internal_runs_router(
    run_service: RunService,
    service_token: Optional[str],
    callback_service: Optional[PostgreSQLTerminalCallbackService] = None,
) -> APIRouter:
    if service_token is not None and not service_token.strip():
        raise ValueError("internal Run Event service token must not be blank")
    router = APIRouter(prefix="/api/internal/v1")

    @router.post(
        "/runs/{run_id}/events",
        response_model=InternalRunEventResponse,
        status_code=202,
    )
    def ingest_run_event(
        run_id: str,
        payload: InternalRunEventRequest,
        authorization: Optional[str] = Header(None, alias="Authorization"),
    ) -> InternalRunEventResponse:
        _authorize(service_token, authorization)
        if run_id != payload.run_id:
            _error(400, "run_id_mismatch", "path and body run_id must match")

        event = RunEvent(
            schema_version=payload.schema_version,
            event_id=payload.event_id,
            sequence=payload.sequence,
            type=RunEventType(payload.type),
            run_id=payload.run_id,
            task_id=payload.task_id,
            occurred_at=payload.occurred_at,
            payload=payload.payload.model_dump(),
        )
        try:
            result = run_service.apply_simulation_event(event)
        except RunNotFound:
            _error(404, "run_not_found", "Run not found")
        except RunTaskMismatch:
            _error(409, "run_task_conflict", "Run and simulation task are not compatible")
        except EventConflict:
            _error(409, "event_id_conflict", "event_id has different content")
        except EventSequenceConflict:
            _error(409, "sequence_conflict", "Run sequence is already occupied")
        except (InvalidRunEvent, InvalidRunStateTransition):
            _error(409, "event_state_conflict", "Event conflicts with the current Run snapshot")
        except (RepositoryDataError, RepositoryIntegrityError, SQLAlchemyError):
            _error(503, "event_store_unavailable", "Run Event storage is temporarily unavailable")

        return InternalRunEventResponse(
            duplicate=result == ApplyEventResult.DUPLICATE,
            late=result == ApplyEventResult.LATE,
        )

    @router.post(
        "/runs/{run_id}/callback",
        response_model=TerminalCallbackResponse,
        status_code=202,
    )
    def ingest_terminal_callback(
        run_id: str,
        payload: TerminalCallbackRequest,
        authorization: Optional[str] = Header(None, alias="Authorization"),
    ) -> TerminalCallbackResponse:
        _authorize(service_token, authorization)
        if run_id != payload.run_id:
            _error(400, "run_id_mismatch", "path and body run_id must match")
        if callback_service is None:
            _error(
                503,
                "callback_store_unavailable",
                "Terminal Callback storage is not configured",
            )
        try:
            callback_run = run_service.get_run(run_id)
        except RunNotFound:
            _error(404, "run_not_found", "Run not found")
        except (RepositoryDataError, RepositoryIntegrityError, SQLAlchemyError):
            _error(
                503,
                "callback_store_unavailable",
                "Terminal Callback storage is temporarily unavailable",
            )
        try:
            validate_terminal_callback_contract(callback_run, payload)
        except ValueError as exc:
            _error(422, "terminal_contract_invalid", str(exc))

        command = terminal_callback_command(payload)
        try:
            result = callback_service.apply(command)
        except RunNotFound:
            _error(404, "run_not_found", "Run not found")
        except RunTaskMismatch:
            _error(409, "run_task_conflict", "Run and simulation task are not compatible")
        except EventConflict:
            _error(409, "event_id_conflict", "event_id has different content")
        except EventSequenceConflict:
            _error(409, "sequence_conflict", "Run sequence is already occupied")
        except (
            ArtifactMetadataConflict,
            TerminalCallbackConflict,
            TerminalRunStatusConflict,
        ):
            _error(409, "terminal_conflict", "Callback conflicts with terminal Run facts")
        except (RepositoryDataError, RepositoryIntegrityError, SQLAlchemyError):
            _error(
                503,
                "callback_store_unavailable",
                "Terminal Callback storage is temporarily unavailable",
            )
        return TerminalCallbackResponse(duplicate=result.duplicate)

    return router


def terminal_callback_command(payload: TerminalCallbackRequest) -> TerminalCallbackCommand:
    """Map a validated wire Callback to the canonical idempotent command."""

    event_payload = payload.model_dump(
        mode="json",
        by_alias=True,
        exclude={"schema_version", "event_id", "sequence", "run_id", "task_id"},
    )
    event = RunEvent(
        schema_version=payload.schema_version,
        event_id=payload.event_id,
        sequence=payload.sequence,
        type=RunEventType.STATE,
        run_id=payload.run_id,
        task_id=payload.task_id,
        occurred_at=payload.ended_at,
        payload=event_payload,
    )
    registrations = [
        ArtifactRegistration(
            artifact_type=item.artifact_type,
            path=item.path,
            size_bytes=item.size_bytes,
            sha256=item.sha256,
        )
        for item in payload.artifacts
    ]
    if payload.model is not None:
        registrations.append(
            ArtifactRegistration(
                artifact_type=payload.model.artifact_type,
                path=payload.model.path,
                size_bytes=payload.model.size_bytes,
                sha256=payload.model.sha256,
                is_primary_model=True,
            )
        )
    error = (
        RunError(**payload.error.model_dump())
        if payload.error is not None
        else None
    )
    return TerminalCallbackCommand(
        event=event,
        status=CanonicalRunStatus(payload.status),
        started_at=payload.started_at,
        ended_at=payload.ended_at,
        training_steps=payload.training_steps,
        update_steps=payload.update_steps,
        stop_reason=payload.stop_reason,
        metrics=payload.metrics,
        evaluation=(
            payload.evaluation.model_dump(mode="json", exclude_none=True)
            if payload.evaluation is not None else None
        ),
        error=error,
        artifacts=tuple(registrations),
    )


def _authorize(expected: Optional[str], authorization: Optional[str]) -> None:
    if expected is None:
        return
    if authorization is None:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            detail={"code": "unauthorized", "message": "Bearer token is required"},
        )
    scheme, separator, supplied = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not supplied:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            detail={"code": "unauthorized", "message": "Bearer token is required"},
        )
    if not hmac.compare_digest(supplied, expected):
        _error(403, "forbidden", "Bearer token is not valid")


def validate_terminal_callback_contract(run, payload: TerminalCallbackRequest) -> None:
    """Apply Run-subtype conditions after structural request validation."""

    snapshot = run.project_config_snapshot
    snapshot_type = snapshot.get("task_type")
    artifact_types = {item.artifact_type for item in payload.artifacts}
    model_type = payload.model.artifact_type if payload.model is not None else None
    if run.run_kind is RunKind.PLAY:
        if snapshot_type != "demo":
            raise ValueError("Demo Run snapshot task_type must be demo")
        catalog_task = snapshot.get("catalog_task")
        if (
            not isinstance(catalog_task, dict)
            or catalog_task.get("task_type") != "demo"
            or catalog_task.get("id") != snapshot.get("environment_type")
        ):
            raise ValueError("Demo Run catalog identity is invalid")
        if payload.model is not None:
            raise ValueError("Demo Callback model must be null")
        if payload.training_steps != 0 or payload.update_steps != 0:
            raise ValueError("Demo Callback step counters must be zero")
        if payload.evaluation is not None:
            raise ValueError("Demo Callback evaluation must be null")
        allowed = {
            "demo_config", "demo_metrics", "stdout_log", "stderr_log",
            "task_log", "manifest", "demo_video",
        }
        unknown = sorted(artifact_types - allowed)
        if unknown:
            raise ValueError(
                "Demo Callback contains forbidden Artifact types: %s"
                % ", ".join(unknown)
            )
        if payload.status in {"completed", "stopped"}:
            if "manifest" not in artifact_types:
                raise ValueError("Demo Callback requires manifest Artifact")
        return

    if run.run_kind is not RunKind.TRAIN or snapshot_type != "training":
        raise ValueError("Training Run snapshot task_type must be training")
    if payload.status != "completed":
        return
    if payload.model is None:
        raise ValueError("Training completed Callback requires model")
    if model_type not in {"checkpoint_best", "checkpoint_latest"}:
        raise ValueError("Training model must be a checkpoint Artifact")
    if "manifest" not in artifact_types:
        raise ValueError("Training completed Callback requires manifest Artifact")
    # Catalog-created Runs use the current Simulation artifact policy.  Legacy
    # snapshots retain their historical smoke callback compatibility.
    if "catalog_task" not in snapshot:
        return
    allowed = {
        "checkpoint_best", "checkpoint_latest", "training_config",
        "training_metrics", "training_episode_metrics", "stdout_log",
        "stderr_log", "task_log", "manifest", "training_video",
        "evaluation_summary", "evaluation_episodes", "evaluation_video",
    }
    unknown = sorted(artifact_types - allowed)
    if unknown:
        raise ValueError(
            "Training Callback contains invalid Artifact types: %s"
            % ", ".join(unknown)
        )
    required = {
        "checkpoint_best", "checkpoint_latest", "training_config",
        "stdout_log", "stderr_log", "task_log", "manifest",
    }
    present = artifact_types | {model_type}
    missing = sorted(required - present)
    if missing:
        raise ValueError(
            "Training completed Callback is missing Artifacts: %s"
            % ", ".join(missing)
        )
    if not {"training_metrics", "training_episode_metrics"} & artifact_types:
        raise ValueError(
            "Training completed Callback requires training_metrics or "
            "training_episode_metrics Artifact"
        )
    task_config = snapshot.get("task_config")
    if isinstance(task_config, dict) and task_config.get("run_mode") == "training":
        if payload.evaluation is None:
            raise ValueError("Production Training Callback requires evaluation")
        evaluation_artifacts = {"evaluation_summary", "evaluation_episodes"}
        missing_evaluation = sorted(evaluation_artifacts - artifact_types)
        if missing_evaluation:
            raise ValueError(
                "Production Training Callback is missing evaluation Artifacts: %s"
                % ", ".join(missing_evaluation)
            )


def _error(status_code: int, code: str, message: str) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )

