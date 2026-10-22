"""Synchronous SQLAlchemy repository for canonical Runs in PostgreSQL."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional

from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from robotics.runs.db_models import RunEventRow, RunRow, RunArtifactRow, RunIdempotencyKeyRow, EvaluationReportRow
from robotics.runs.domain import RunStatus
from robotics.runs.models import (
    ExecutionMode,
    Run,
    RunError,
    RunEvent,
    RunEventType,
    RunKind,
)
from robotics.runs.repository import (
    AppendEventResult,
    EventConflict,
    EventSequenceConflict,
    RepositoryDataError,
    RepositoryIntegrityError,
    RunAlreadyExists,
    RunNotFound,
    TaskIdConflict,
    validate_run_deletion,
    dispatch_error_update,
)


def run_to_row(run: Run) -> RunRow:
    """Copy a canonical Run into a new SQLAlchemy row."""

    return RunRow(
        run_id=run.run_id,
        user_id=run.user_id,
        project_id=run.project_id,
        class_id=run.class_id,
        status=run.status.value,
        execution_mode=run.execution_mode.value,
        run_kind=run.run_kind.value,
        parameters=deepcopy(run.parameters),
        project_config_snapshot=deepcopy(run.project_config_snapshot),
        code_sha256=run.code_sha256,
        simulation_task_id=run.simulation_task_id,
        progress=run.progress,
        metrics_summary=deepcopy(run.metrics_summary),
        error=_error_to_json(run.error),
        last_sequence=run.last_sequence,
        training_steps=run.training_steps,
        update_steps=run.update_steps,
        stop_reason=run.stop_reason,
        created_at=run.created_at,
        started_at=run.started_at,
        ended_at=run.ended_at,
        updated_at=run.updated_at,
    )


def run_from_row(row: RunRow) -> Run:
    """Restore a canonical Run, rejecting malformed persisted values."""

    run_id = getattr(row, "run_id", "<unknown>")
    try:
        raw_error = _copy_optional_json_object(row.error, "error")
        error = RunError(**raw_error) if raw_error is not None else None
        return Run(
            run_id=run_id,
            user_id=row.user_id,
            project_id=row.project_id,
            class_id=getattr(row, "class_id", None),
            status=RunStatus(row.status),
            execution_mode=ExecutionMode(row.execution_mode),
            run_kind=RunKind(row.run_kind),
            parameters=_copy_json_object(row.parameters, "parameters"),
            project_config_snapshot=_copy_json_object(
                row.project_config_snapshot, "project_config_snapshot"
            ),
            code_sha256=row.code_sha256,
            simulation_task_id=row.simulation_task_id,
            progress=row.progress,
            metrics_summary=_copy_json_object(
                row.metrics_summary, "metrics_summary"
            ),
            error=error,
            last_sequence=row.last_sequence,
            training_steps=row.training_steps,
            update_steps=row.update_steps,
            stop_reason=row.stop_reason,
            created_at=_utc_datetime(row.created_at, "created_at"),
            started_at=_optional_utc_datetime(row.started_at, "started_at"),
            ended_at=_optional_utc_datetime(row.ended_at, "ended_at"),
            updated_at=_utc_datetime(row.updated_at, "updated_at"),
        )
    except RepositoryDataError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise RepositoryDataError(
            "Invalid persisted Run %s: %s" % (run_id, exc)
        ) from exc


def event_to_row(event: RunEvent) -> RunEventRow:
    """Copy a canonical Event into a new SQLAlchemy row."""

    return RunEventRow(
        event_id=event.event_id,
        run_id=event.run_id,
        sequence=event.sequence,
        type=event.type.value,
        task_id=event.task_id,
        occurred_at=event.occurred_at,
        schema_version=event.schema_version,
        payload=deepcopy(event.payload),
    )


def event_from_row(row: RunEventRow) -> RunEvent:
    """Restore a canonical Event, rejecting malformed persisted values."""

    event_id = getattr(row, "event_id", "<unknown>")
    try:
        return RunEvent(
            event_id=event_id,
            run_id=row.run_id,
            sequence=row.sequence,
            type=RunEventType(row.type),
            task_id=row.task_id,
            occurred_at=_utc_datetime(row.occurred_at, "occurred_at"),
            schema_version=row.schema_version,
            payload=_copy_json_object(row.payload, "payload"),
        )
    except RepositoryDataError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise RepositoryDataError(
            "Invalid persisted RunEvent %s: %s" % (event_id, exc)
        ) from exc


class PostgreSQLRunRepository:
    """RunRepository implementation with one synchronous Session per operation."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory

    @property
    def session_factory(self) -> Callable[[], Session]:
        """Expose the factory for services that must share one transaction."""

        return self._session_factory

    @contextmanager
    def _session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def create_run(self, run: Run) -> Run:
        try:
            with self._session() as session:
                row = run_to_row(run)
                session.add(row)
                session.flush()
                return run_from_row(row)
        except IntegrityError as exc:
            self._raise_run_write_conflict(
                run, exc, operation="create_run", duplicate_is_conflict=True
            )
            raise AssertionError("unreachable")

    def get_run(self, run_id: str) -> Run:
        with self._session() as session:
            row = session.get(RunRow, run_id)
            if row is None:
                raise RunNotFound(run_id)
            return run_from_row(row)

    def list_runs(self) -> List[Run]:
        with self._session() as session:
            rows = session.execute(
                select(RunRow).order_by(RunRow.created_at.desc(), RunRow.run_id.asc())
            ).scalars().all()
            return [run_from_row(row) for row in rows]

    def delete_owned_terminal_runs(self, run_ids: List[str], actor_user_id: str) -> int:
        # Lock in a consistent order; validate the whole batch before any mutation.
        with self._session() as session:
            rows = session.execute(select(RunRow).where(RunRow.run_id.in_(run_ids))
                                   .order_by(RunRow.run_id).with_for_update()).scalars().all()
            validate_run_deletion(rows, run_ids, actor_user_id)
            for model in (EvaluationReportRow, RunArtifactRow, RunEventRow, RunIdempotencyKeyRow):
                session.execute(delete(model).where(model.run_id.in_(run_ids)))
            session.execute(delete(RunRow).where(RunRow.run_id.in_(run_ids)))
            return len(run_ids)

    def save_run(self, run: Run) -> Run:
        try:
            with self._session() as session:
                row = session.execute(
                    select(RunRow)
                    .where(RunRow.run_id == run.run_id)
                    .with_for_update()
                ).scalar_one_or_none()
                if row is None:
                    raise RunNotFound(run.run_id)
                if run.last_sequence < row.last_sequence:
                    return run_from_row(row)
                copy_run_to_existing_row(run, row)
                session.flush()
                return run_from_row(row)
        except IntegrityError as exc:
            self._raise_run_write_conflict(
                run, exc, operation="save_run", duplicate_is_conflict=False
            )
            raise AssertionError("unreachable")

    def record_dispatch_error(self, run_id, error, now, definitive):
        # Status, error and audit event commit or roll back together.
        with self._session() as session:
            row = session.execute(select(RunRow).where(RunRow.run_id == run_id)
                                  .with_for_update()).scalar_one_or_none()
            if row is None:
                raise RunNotFound(run_id)
            current = run_from_row(row)
            if session.execute(select(RunEventRow.event_id).where(
                    RunEventRow.run_id == run_id).limit(1)).first() is not None:
                return current
            updated, event = dispatch_error_update(current, error, now, definitive)
            copy_run_to_existing_row(updated, row)
            if event is not None:
                session.add(event_to_row(event))
            session.flush()
            return run_from_row(row)

    def bind_task_id(self, run_id: str, task_id: str) -> Run:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        try:
            with self._session() as session:
                row = session.execute(
                    select(RunRow)
                    .where(RunRow.run_id == run_id)
                    .with_for_update()
                ).scalar_one_or_none()
                if row is None:
                    raise RunNotFound(run_id)
                if row.simulation_task_id is not None:
                    if row.simulation_task_id != task_id:
                        raise TaskIdConflict(
                            run_id, row.simulation_task_id, task_id
                        )
                    return run_from_row(row)
                row.simulation_task_id = task_id
                if row.error and row.error.get("code", "").startswith("DISPATCH_"):
                    row.error = None
                session.flush()
                return run_from_row(row)
        except IntegrityError as exc:
            self._raise_task_owner_conflict(
                run_id, task_id, exc, operation="bind_task_id"
            )
            raise AssertionError("unreachable")

    def append_event(self, event: RunEvent) -> AppendEventResult:
        try:
            with self._session() as session:
                if session.get(RunRow, event.run_id) is None:
                    raise RunNotFound(event.run_id)
                session.add(event_to_row(event))
                session.flush()
                return AppendEventResult.STORED
        except IntegrityError as exc:
            return self._classify_event_conflict(event, exc)

    def list_events(
        self,
        run_id: str,
        after_sequence: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[RunEvent]:
        threshold = after_sequence if after_sequence is not None else 0
        with self._session() as session:
            if session.get(RunRow, run_id) is None:
                raise RunNotFound(run_id)
            statement = (
                select(RunEventRow)
                .where(
                    RunEventRow.run_id == run_id,
                    RunEventRow.sequence > threshold,
                )
                .order_by(RunEventRow.sequence.asc())
            )
            if limit is not None:
                statement = statement.limit(limit)
            rows = session.execute(statement).scalars().all()
            return [event_from_row(row) for row in rows]

    def _raise_run_write_conflict(
        self,
        run: Run,
        cause: IntegrityError,
        *,
        operation: str,
        duplicate_is_conflict: bool,
    ) -> None:
        with self._session() as session:
            if duplicate_is_conflict and session.get(RunRow, run.run_id) is not None:
                raise RunAlreadyExists(run.run_id) from cause
            if run.simulation_task_id is not None:
                owner_run_id = self._task_owner(
                    session, run.simulation_task_id
                )
                if owner_run_id is not None and owner_run_id != run.run_id:
                    raise TaskIdConflict(
                        run.run_id,
                        None,
                        run.simulation_task_id,
                        owner_run_id=owner_run_id,
                    ) from cause
            raise RepositoryIntegrityError(operation) from cause

    def _raise_task_owner_conflict(
        self,
        run_id: str,
        task_id: str,
        cause: IntegrityError,
        *,
        operation: str,
    ) -> None:
        with self._session() as session:
            owner_run_id = self._task_owner(session, task_id)
            if owner_run_id is not None and owner_run_id != run_id:
                raise TaskIdConflict(
                    run_id,
                    None,
                    task_id,
                    owner_run_id=owner_run_id,
                ) from cause
            raise RepositoryIntegrityError(operation) from cause

    def _classify_event_conflict(
        self, event: RunEvent, cause: IntegrityError
    ) -> AppendEventResult:
        with self._session() as session:
            if session.get(RunRow, event.run_id) is None:
                raise RunNotFound(event.run_id) from cause
            existing = session.get(RunEventRow, event.event_id)
            if existing is not None:
                if event_from_row(existing) == event:
                    return AppendEventResult.DUPLICATE
                raise EventConflict(event.event_id) from cause
            occupied = session.execute(
                select(RunEventRow).where(
                    RunEventRow.run_id == event.run_id,
                    RunEventRow.sequence == event.sequence,
                )
            ).scalar_one_or_none()
            if occupied is not None:
                raise EventSequenceConflict(event.run_id, event.sequence) from cause
            raise RepositoryIntegrityError("append_event") from cause

    @staticmethod
    def _task_owner(session: Session, task_id: str) -> Optional[str]:
        return session.execute(
            select(RunRow.run_id).where(RunRow.simulation_task_id == task_id)
        ).scalar_one_or_none()


def copy_run_to_existing_row(run: Run, row: RunRow) -> None:
    row.user_id = run.user_id
    row.project_id = run.project_id
    row.class_id = run.class_id
    row.status = run.status.value
    row.execution_mode = run.execution_mode.value
    row.run_kind = run.run_kind.value
    row.parameters = deepcopy(run.parameters)
    row.project_config_snapshot = deepcopy(run.project_config_snapshot)
    row.code_sha256 = run.code_sha256
    row.simulation_task_id = run.simulation_task_id
    row.progress = run.progress
    row.metrics_summary = deepcopy(run.metrics_summary)
    row.error = _error_to_json(run.error)
    row.last_sequence = run.last_sequence
    row.training_steps = run.training_steps
    row.update_steps = run.update_steps
    row.stop_reason = run.stop_reason
    row.created_at = run.created_at
    row.started_at = run.started_at
    row.ended_at = run.ended_at
    row.updated_at = run.updated_at


def _error_to_json(error: Optional[RunError]) -> Optional[Dict[str, Any]]:
    if error is None:
        return None
    return {
        "code": error.code,
        "message": error.message,
        "retryable": error.retryable,
    }


def _copy_json_object(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RepositoryDataError("Persisted %s must be a JSON object" % field)
    return deepcopy(value)


def _copy_optional_json_object(
    value: Any, field: str
) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return _copy_json_object(value, field)


def _utc_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise RepositoryDataError("Persisted %s must be a datetime" % field)
    if value.tzinfo is None or value.utcoffset() is None:
        raise RepositoryDataError("Persisted %s must be timezone-aware UTC" % field)
    if value.utcoffset() != timedelta(0):
        raise RepositoryDataError("Persisted %s must be UTC" % field)
    return value.astimezone(timezone.utc)


def _optional_utc_datetime(value: Any, field: str) -> Optional[datetime]:
    return None if value is None else _utc_datetime(value, field)

