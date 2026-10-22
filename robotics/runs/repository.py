"""Repository contract and thread-safe in-memory implementation for Runs."""

from copy import deepcopy
from dataclasses import replace
from enum import Enum
from datetime import datetime
from uuid import uuid4
from threading import RLock
from typing import Dict, List, Optional, Protocol

from robotics.runs.models import Run, RunError, RunEvent, RunEventType, ExecutionMode
from robotics.runs.domain import RunStatus


class RepositoryError(Exception):
    """Base class for Run repository errors."""


class RunDeletionNotAllowed(RepositoryError):
    """Only terminal, owned Runs can be deleted."""


def validate_run_deletion(runs, run_ids, actor_user_id):
    by_id = {run.run_id: run for run in runs}
    for run_id in run_ids:
        run = by_id.get(run_id)
        if run is None or run.user_id != actor_user_id:
            raise RunNotFound(run_id)
    if any(str(getattr(run.status, "value", run.status)) not in {"completed", "failed", "stopped"} for run in runs):
        raise RunDeletionNotAllowed("Only completed, failed or stopped runs can be deleted")


class RunNotFound(RepositoryError, LookupError):
    def __init__(self, run_id: str) -> None:
        super().__init__("Run not found: %s" % run_id)
        self.run_id = run_id


class RunAlreadyExists(RepositoryError):
    def __init__(self, run_id: str) -> None:
        super().__init__("Run already exists: %s" % run_id)
        self.run_id = run_id


class TaskIdConflict(RepositoryError):
    def __init__(
        self,
        run_id: str,
        current: Optional[str],
        target: str,
        owner_run_id: Optional[str] = None,
    ) -> None:
        if owner_run_id is None:
            message = "Run %s is already bound to task %s, not %s" % (
                run_id,
                current,
                target,
            )
        else:
            message = "Task %s is already bound to Run %s, not Run %s" % (
                target,
                owner_run_id,
                run_id,
            )
        super().__init__(message)
        self.run_id = run_id
        self.current = current
        self.target = target
        self.owner_run_id = owner_run_id


class TaskIdAlreadyBound(RepositoryError):
    def __init__(self, task_id: str) -> None:
        super().__init__("Simulation task is already bound: %s" % task_id)
        self.task_id = task_id


class RepositoryDataError(RepositoryError):
    """Raised when persisted repository data cannot be decoded safely."""


class RepositoryIntegrityError(RepositoryError):
    """Raised when a database integrity failure has no domain conflict mapping."""

    def __init__(self, operation: str) -> None:
        super().__init__("Repository integrity failure during %s" % operation)
        self.operation = operation


class EventConflict(RepositoryError):
    def __init__(self, event_id: str) -> None:
        super().__init__("event_id has different content: %s" % event_id)
        self.event_id = event_id


class EventSequenceConflict(RepositoryError):
    def __init__(self, run_id: str, sequence: int) -> None:
        super().__init__(
            "Run %s sequence is already occupied: %s" % (run_id, sequence)
        )
        self.run_id = run_id
        self.sequence = sequence


class AppendEventResult(str, Enum):
    STORED = "stored"
    DUPLICATE = "duplicate"


def dispatch_error_update(run, error, now, definitive):
    """Only a pristine, unbound simulation Run can fail before dispatch.

    Temporary errors do not consume the simulation event sequence: an upstream
    task may already exist and its first event must still be accepted.
    """
    if (run.execution_mode != ExecutionMode.SIMULATION
            or run.status != RunStatus.QUEUED
            or run.simulation_task_id is not None
            or run.last_sequence != 0 or run.started_at is not None
            or run.training_steps != 0 or run.update_steps != 0):
        return run, None
    if not definitive:
        return replace(run, error=error, updated_at=now), None
    updated = replace(run, status=RunStatus.FAILED, error=error,
                      ended_at=now, updated_at=now, last_sequence=1)
    event = RunEvent(
        event_id="dispatch-failed-" + str(uuid4()), run_id=run.run_id,
        sequence=1, type=RunEventType.STATE, occurred_at=now,
        payload={"previous_status": "queued", "status": "failed",
                 "source": "dispatch_validation", "error_code": error.code},
    )
    return updated, event


class RunRepository(Protocol):
    def create_run(self, run: Run) -> Run:
        ...

    def get_run(self, run_id: str) -> Run:
        ...

    def list_runs(self) -> List[Run]:
        ...

    def save_run(self, run: Run) -> Run:
        ...

    def record_dispatch_error(self, run_id: str, error: RunError, now: datetime, definitive: bool) -> Run:
        ...

    def bind_task_id(self, run_id: str, task_id: str) -> Run:
        ...

    def append_event(self, event: RunEvent) -> AppendEventResult:
        ...

    def list_events(
        self,
        run_id: str,
        after_sequence: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[RunEvent]:
        ...


class InMemoryRunRepository:
    """Process-local repository that never exposes its mutable stored values."""

    def __init__(self) -> None:
        self._runs: Dict[str, Run] = {}
        self._events_by_run: Dict[str, Dict[int, RunEvent]] = {}
        self._events_by_id: Dict[str, RunEvent] = {}
        self._lock = RLock()

    def create_run(self, run: Run) -> Run:
        with self._lock:
            if run.run_id in self._runs:
                raise RunAlreadyExists(run.run_id)
            stored = deepcopy(run)
            self._runs[run.run_id] = stored
            self._events_by_run[run.run_id] = {}
            return deepcopy(stored)

    def get_run(self, run_id: str) -> Run:
        with self._lock:
            return deepcopy(self._require_run(run_id))

    def list_runs(self) -> List[Run]:
        with self._lock:
            ordered = sorted(
                self._runs.values(), key=lambda run: run.created_at, reverse=True
            )
            return deepcopy(ordered)

    def delete_owned_terminal_runs(self, run_ids: List[str], actor_user_id: str) -> int:
        with self._lock:
            runs = [self._runs[run_id] for run_id in run_ids if run_id in self._runs]
            validate_run_deletion(runs, run_ids, actor_user_id)
            for run_id in run_ids:
                for event in self._events_by_run.pop(run_id, {}).values():
                    self._events_by_id.pop(event.event_id, None)
                del self._runs[run_id]
            return len(run_ids)

    def save_run(self, run: Run) -> Run:
        with self._lock:
            current = self._require_run(run.run_id)
            if run.last_sequence < current.last_sequence:
                return deepcopy(current)
            stored = deepcopy(run)
            self._runs[run.run_id] = stored
            return deepcopy(stored)

    def record_dispatch_error(self, run_id, error, now, definitive):
        with self._lock:
            current = self._require_run(run_id)
            # Unexpected persisted events are execution evidence; preserve them.
            if self._events_by_run.get(run_id):
                return deepcopy(current)
            updated, event = dispatch_error_update(current, error, now, definitive)
            if event is not None:
                self.append_event(event)
            self._runs[run_id] = deepcopy(updated)
            return deepcopy(updated)

    def bind_task_id(self, run_id: str, task_id: str) -> Run:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        with self._lock:
            run = self._require_run(run_id)
            if run.simulation_task_id is not None:
                if run.simulation_task_id != task_id:
                    raise TaskIdConflict(run_id, run.simulation_task_id, task_id)
                return deepcopy(run)
            stored = replace(run, simulation_task_id=task_id,
                             error=None if run.error and run.error.code.startswith("DISPATCH_") else run.error)
            self._runs[run_id] = stored
            return deepcopy(stored)

    def append_event(self, event: RunEvent) -> AppendEventResult:
        with self._lock:
            self._require_run(event.run_id)
            existing_by_id = self._events_by_id.get(event.event_id)
            if existing_by_id is not None:
                if existing_by_id == event:
                    return AppendEventResult.DUPLICATE
                raise EventConflict(event.event_id)
            events = self._events_by_run[event.run_id]
            if event.sequence in events:
                raise EventSequenceConflict(event.run_id, event.sequence)
            stored = deepcopy(event)
            events[event.sequence] = stored
            self._events_by_id[event.event_id] = stored
            return AppendEventResult.STORED

    def list_events(
        self,
        run_id: str,
        after_sequence: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[RunEvent]:
        with self._lock:
            self._require_run(run_id)
            threshold = after_sequence if after_sequence is not None else 0
            events = self._events_by_run[run_id]
            ordered = [events[key] for key in sorted(events) if key > threshold]
            if limit is not None:
                ordered = ordered[:limit]
            return deepcopy(ordered)

    def _require_run(self, run_id: str) -> Run:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise RunNotFound(run_id) from exc

