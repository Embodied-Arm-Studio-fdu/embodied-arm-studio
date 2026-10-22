"""Atomic PostgreSQL application of trusted terminal Simulation Callbacks."""

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from robotics.runs.artifacts import ArtifactRegistration, RunArtifact
from robotics.runs.db_models import RunArtifactRow, RunEventRow, RunRow
from robotics.runs.domain import (
    RunStatus,
    is_terminal,
    validate_terminal_convergence,
)
from robotics.runs.models import ExecutionMode, RunError, RunEvent
from robotics.runs.postgres_repository import (
    copy_run_to_existing_row,
    event_from_row,
    event_to_row,
    run_from_row,
)
from robotics.runs.repository import (
    EventConflict,
    EventSequenceConflict,
    RepositoryIntegrityError,
    RunNotFound,
)
from robotics.runs.service import (
    RunTaskMismatch,
    calculate_run_progress,
    validate_update_index,
)


class TerminalCallbackConflict(ValueError):
    """Raised when a Callback cannot reconcile with the locked Run."""


class ArtifactMetadataConflict(ValueError):
    """Raised when one path carries incompatible metadata."""


@dataclass(frozen=True)
class TerminalCallbackCommand:
    event: RunEvent
    status: RunStatus
    started_at: datetime
    ended_at: datetime
    training_steps: int
    update_steps: int
    stop_reason: Optional[str]
    metrics: Dict[str, object]
    error: Optional[RunError]
    artifacts: Tuple[ArtifactRegistration, ...]
    evaluation: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class TerminalCallbackResult:
    duplicate: bool


class PostgreSQLTerminalCallbackService:
    """Commit Run, terminal Event, and Artifact metadata in one transaction."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        before_commit: Optional[Callable[[Session], None]] = None,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory
        self._clock = clock
        self._before_commit = before_commit

    def apply(self, command: TerminalCallbackCommand) -> TerminalCallbackResult:
        if not isinstance(command, TerminalCallbackCommand):
            raise TypeError("command must be a TerminalCallbackCommand")
        session = self._session_factory()
        try:
            with session.begin():
                row = session.execute(
                    select(RunRow)
                    .where(RunRow.run_id == command.event.run_id)
                    .with_for_update()
                ).scalar_one_or_none()
                if row is None:
                    raise RunNotFound(command.event.run_id)
                run = run_from_row(row)
                self._validate_run(run, command)

                existing = session.get(RunEventRow, command.event.event_id)
                if existing is not None:
                    if event_from_row(existing) == command.event:
                        if run.status != command.status:
                            raise TerminalCallbackConflict(
                                "Callback status does not match terminal Run"
                            )
                        return TerminalCallbackResult(duplicate=True)
                    raise EventConflict(command.event.event_id)

                occupied = session.execute(
                    select(RunEventRow).where(
                        RunEventRow.run_id == command.event.run_id,
                        RunEventRow.sequence == command.event.sequence,
                    )
                ).scalar_one_or_none()
                if occupied is not None:
                    raise EventSequenceConflict(
                        command.event.run_id, command.event.sequence
                    )

                validate_terminal_convergence(run.status, command.status)
                if is_terminal(run.status):
                    raise TerminalCallbackConflict(
                        "Run already has a different terminal fact"
                    )
                if command.event.sequence <= run.last_sequence:
                    raise TerminalCallbackConflict(
                        "Callback sequence is not newer than the Run snapshot"
                    )

                metrics = deepcopy(command.metrics)
                if command.evaluation is not None:
                    metrics["evaluation"] = deepcopy(command.evaluation)
                if "update_index" in metrics:
                    validate_update_index(
                        metrics["update_index"], "Callback metrics.update_index"
                    )
                training_steps = max(run.training_steps, command.training_steps)
                metrics["training_steps"] = training_steps
                metrics["update_steps"] = command.update_steps
                updated = replace(
                    run,
                    status=command.status,
                    progress=calculate_run_progress(
                        run,
                        status=command.status,
                        training_steps=training_steps,
                    ),
                    metrics_summary=metrics,
                    error=command.error,
                    last_sequence=command.event.sequence,
                    training_steps=training_steps,
                    update_steps=command.update_steps,
                    stop_reason=command.stop_reason,
                    started_at=command.started_at,
                    ended_at=command.ended_at,
                    updated_at=self._clock(),
                )
                copy_run_to_existing_row(updated, row)
                session.add(event_to_row(command.event))
                for artifact in self._merge_artifacts(command.artifacts):
                    session.add(
                        RunArtifactRow(
                            artifact_id=self._artifact_id(
                                command.event.run_id, artifact.path
                            ),
                            run_id=command.event.run_id,
                            artifact_type=artifact.artifact_type,
                            path=artifact.path,
                            size_bytes=artifact.size_bytes,
                            sha256=artifact.sha256,
                            is_primary_model=artifact.is_primary_model,
                            created_at=command.ended_at,
                        )
                    )
                session.flush()
                if self._before_commit is not None:
                    self._before_commit(session)
            return TerminalCallbackResult(duplicate=False)
        except IntegrityError as exc:
            raise RepositoryIntegrityError("terminal_callback") from exc
        finally:
            session.close()

    def list_artifacts(self, run_id: str) -> List[RunArtifact]:
        session = self._session_factory()
        try:
            if session.get(RunRow, run_id) is None:
                raise RunNotFound(run_id)
            rows = session.execute(
                select(RunArtifactRow)
                .where(RunArtifactRow.run_id == run_id)
                .order_by(RunArtifactRow.path.asc())
            ).scalars().all()
            return [self._artifact_from_row(row) for row in rows]
        except (SQLAlchemyError, ValueError) as exc:
            raise RepositoryIntegrityError("artifact_list") from exc
        finally:
            session.close()

    def get_artifact(self, run_id: str, artifact_id: str) -> RunArtifact:
        """Return one registered Artifact only when it belongs to the Run."""
        session = self._session_factory()
        try:
            row = session.execute(
                select(RunArtifactRow).where(
                    RunArtifactRow.artifact_id == artifact_id,
                    RunArtifactRow.run_id == run_id,
                )
            ).scalar_one_or_none()
            if row is None:
                # Keep unknown Run, unknown Artifact, and cross-Run IDs opaque.
                raise RunNotFound(run_id)
            return self._artifact_from_row(row)
        except (SQLAlchemyError, ValueError) as exc:
            raise RepositoryIntegrityError("artifact_lookup") from exc
        finally:
            session.close()

    @staticmethod
    def _validate_run(run, command: TerminalCallbackCommand) -> None:
        if run.execution_mode != ExecutionMode.SIMULATION:
            raise RunTaskMismatch("Run execution_mode must be simulation")
        if run.simulation_task_id is None:
            raise RunTaskMismatch("simulation Run has no bound task_id")
        if command.event.task_id != run.simulation_task_id:
            raise RunTaskMismatch("Callback task_id does not match bound task")

    @staticmethod
    def _merge_artifacts(
        artifacts: Tuple[ArtifactRegistration, ...]
    ) -> List[ArtifactRegistration]:
        merged: Dict[str, ArtifactRegistration] = {}
        paths_by_type: Dict[str, str] = {}
        for item in artifacts:
            existing_path = paths_by_type.get(item.artifact_type)
            if existing_path is not None and existing_path != item.path:
                raise ArtifactMetadataConflict(
                    "Artifact type has multiple paths: %s" % item.artifact_type
                )
            paths_by_type[item.artifact_type] = item.path
            current = merged.get(item.path)
            if current is None:
                merged[item.path] = item
                continue
            size_bytes = PostgreSQLTerminalCallbackService._merge_optional(
                current.size_bytes, item.size_bytes, item.path
            )
            sha256 = PostgreSQLTerminalCallbackService._merge_optional(
                current.sha256, item.sha256, item.path
            )
            artifact_type = (
                item.artifact_type
                if item.is_primary_model
                else current.artifact_type
            )
            merged[item.path] = ArtifactRegistration(
                artifact_type=artifact_type,
                path=item.path,
                size_bytes=size_bytes,
                sha256=sha256,
                is_primary_model=(
                    current.is_primary_model or item.is_primary_model
                ),
            )
        return [merged[path] for path in sorted(merged)]

    @staticmethod
    def _merge_optional(left, right, path: str):
        if left is not None and right is not None and left != right:
            raise ArtifactMetadataConflict(
                "Artifact path has conflicting metadata: %s" % path
            )
        return left if left is not None else right

    @staticmethod
    def _artifact_id(run_id: str, path: str) -> str:
        return str(uuid5(NAMESPACE_URL, "run-artifact:%s:%s" % (run_id, path)))

    @staticmethod
    def _artifact_from_row(row: RunArtifactRow) -> RunArtifact:
        created_at = row.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = created_at.astimezone(timezone.utc)
        return RunArtifact(
            artifact_id=row.artifact_id,
            run_id=row.run_id,
            artifact_type=row.artifact_type,
            path=row.path,
            size_bytes=row.size_bytes,
            sha256=row.sha256,
            is_primary_model=row.is_primary_model,
            created_at=created_at,
        )
