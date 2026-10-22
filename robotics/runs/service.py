"""Business orchestration for canonical platform Runs."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
from typing import Any, Callable, Dict, List, Mapping, Optional
from uuid import uuid4

from robotics.runs.domain import RunStatus, is_terminal, validate_transition
from robotics.runs.models import ExecutionMode, Run, RunError, RunEvent, RunEventType, RunKind
from robotics.runs.repository import (
    AppendEventResult,
    EventConflict,
    EventSequenceConflict,
    RunRepository,
)


MAX_ERROR_MESSAGE_LENGTH = 1000
MAX_ACTIVE_PROGRESS = 0.99
PROGRESSING_STATUSES = frozenset({
    RunStatus.RUNNING,
    RunStatus.FINALIZING,
    RunStatus.STOPPING,
    RunStatus.FAILED,
    RunStatus.STOPPED,
})


class InvalidRunCreateRequest(ValueError):
    """Raised when Run creation input is not valid service input."""


class RunTaskMismatch(ValueError):
    """Raised when task identity is incompatible with a Run."""


class InvalidRunEvent(ValueError):
    """Raised when an Event does not satisfy the canonical contract."""


class RunStopNotAllowed(ValueError):
    """Raised when the frozen stop policy rejects the current Run state."""

    def __init__(self, run_id: str, status: RunStatus) -> None:
        super().__init__(
            "Run %s cannot be stopped while %s" % (run_id, status.value)
        )
        self.run_id = run_id
        self.status = status


class ApplyEventResult(str, Enum):
    """Observable outcomes of accepting a Run Event."""

    APPLIED = "applied"
    DUPLICATE = "duplicate"
    LATE = "late"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    return str(uuid4())


def expected_training_steps(
    project_config_snapshot: Mapping[str, Any]
) -> Optional[int]:
    """Return only a frozen, positive, non-boolean training-step target."""

    value = project_config_snapshot.get("expected_training_steps")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def calculate_run_progress(
    run: Run,
    *,
    status: Optional[RunStatus] = None,
    training_steps: Optional[int] = None,
) -> float:
    """Calculate monotonic estimated progress from the immutable Run snapshot."""

    target_status = run.status if status is None else RunStatus(status)
    steps = run.training_steps if training_steps is None else training_steps
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("training_steps must be a non-negative integer")
    if target_status == RunStatus.COMPLETED:
        return 1.0
    if target_status == RunStatus.QUEUED:
        return 0.0
    if target_status == RunStatus.INITIALIZING:
        return run.progress

    denominator = expected_training_steps(run.project_config_snapshot)
    if denominator is None and run.execution_mode == ExecutionMode.MOCK:
        legacy_max_steps = run.parameters.get("max_steps")
        if (
            not isinstance(legacy_max_steps, bool)
            and isinstance(legacy_max_steps, int)
            and legacy_max_steps > 0
        ):
            denominator = legacy_max_steps
    if denominator is None or target_status not in PROGRESSING_STATUSES:
        return run.progress
    estimate = min(steps / denominator, MAX_ACTIVE_PROGRESS)
    return min(max(run.progress, estimate), MAX_ACTIVE_PROGRESS)


def validate_update_index(value: Any, name: str = "update_index") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidRunEvent(
            "%s must be an integer greater than or equal to 1" % name
        )
    return value


class RunService:
    """Coordinates Run models, state rules, and repository persistence."""

    def __init__(
        self,
        repository: RunRepository,
        clock: Callable[[], datetime] = utc_now,
        run_id_factory: Callable[[], str] = new_run_id,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._run_id_factory = run_id_factory

    @property
    def repository(self) -> RunRepository:
        return self._repository

    def create_run(
        self,
        *,
        user_id: str,
        project_id: str,
        execution_mode: ExecutionMode,
        parameters: Dict[str, Any],
        project_config_snapshot: Dict[str, Any],
        code: str,
        run_kind: RunKind = RunKind.TRAIN,
        class_id: Optional[str] = None,
    ) -> Run:
        """Create and persist a new queued canonical Run."""

        run = self.build_run(
            user_id=user_id,
            project_id=project_id,
            execution_mode=execution_mode,
            parameters=parameters,
            project_config_snapshot=project_config_snapshot,
            code=code,
            run_kind=run_kind,
            class_id=class_id,
        )
        return self._repository.create_run(run)

    def build_run(
        self,
        *,
        user_id: str,
        project_id: str,
        execution_mode: ExecutionMode,
        parameters: Dict[str, Any],
        project_config_snapshot: Dict[str, Any],
        code: str,
        run_kind: RunKind = RunKind.TRAIN,
        class_id: Optional[str] = None,
    ) -> Run:
        """Build an unpersisted Run using the canonical creation rules."""

        try:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("user_id must be a non-empty string")
            if not isinstance(project_id, str) or not project_id.strip():
                raise ValueError("project_id must be a non-empty string")
            if not isinstance(execution_mode, ExecutionMode):
                raise ValueError("execution_mode must be an ExecutionMode")
            if not isinstance(run_kind, RunKind):
                raise ValueError("run_kind must be a RunKind")
            if not isinstance(parameters, dict):
                raise ValueError("parameters must be a dict")
            if not isinstance(project_config_snapshot, dict):
                raise ValueError("project_config_snapshot must be a dict")
            if not isinstance(code, str):
                raise ValueError("code must be a string")
            if class_id is not None and (
                not isinstance(class_id, str) or not class_id.strip()
            ):
                raise ValueError("class_id must be a non-empty string or None")
            run_id = self._run_id_factory()
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError("run_id_factory must return a non-empty string")
            now = self._clock()
            code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
            run = Run(
                run_id=run_id,
                user_id=user_id,
                project_id=project_id,
                status=RunStatus.QUEUED,
                execution_mode=execution_mode,
                code_sha256=code_sha256,
                class_id=class_id,
                run_kind=run_kind,
                parameters=deepcopy(parameters),
                project_config_snapshot=deepcopy(project_config_snapshot),
                simulation_task_id=None,
                progress=0,
                metrics_summary={},
                error=None,
                last_sequence=0,
                created_at=now,
                started_at=None,
                ended_at=None,
                updated_at=now,
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise InvalidRunCreateRequest(str(exc)) from exc
        return run

    def get_run(self, run_id: str) -> Run:
        return self._repository.get_run(run_id)

    def list_runs(self) -> List[Run]:
        return self._repository.list_runs()

    def list_events(
        self,
        run_id: str,
        after_sequence: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[RunEvent]:
        return self._repository.list_events(run_id, after_sequence, limit)

    def record_dispatch_error(self, run_id: str, code: str, message: str, *, definitive: bool) -> Run:
        error = RunError(code=code, message=message, retryable=not definitive)
        return self._repository.record_dispatch_error(run_id, error, self._clock(), definitive)

    def bind_simulation_task(self, run_id: str, task_id: str) -> Run:
        run = self._repository.get_run(run_id)
        if run.execution_mode != ExecutionMode.SIMULATION:
            raise RunTaskMismatch("Mock Run cannot bind a simulation task")
        if not isinstance(task_id, str) or not task_id.strip():
            raise RunTaskMismatch("task_id must be a non-empty string")
        if run.simulation_task_id == task_id:
            return run
        bound = self._repository.bind_task_id(run_id, task_id)
        return self._repository.save_run(replace(bound, updated_at=self._clock()))

    def transition_status(self, run_id: str, target_status: RunStatus) -> Run:
        run = self._repository.get_run(run_id)
        if not isinstance(target_status, RunStatus):
            raise ValueError("target_status must be a RunStatus")
        if run.status == target_status:
            return run
        updated = self._transition(run, target_status, self._clock())
        return self._repository.save_run(updated)

    def apply_event(self, event: RunEvent) -> ApplyEventResult:
        if not isinstance(event, RunEvent):
            raise InvalidRunEvent("event must be a RunEvent")
        run = self._repository.get_run(event.run_id)
        self._validate_event_task(run, event)

        return self._apply_event_to_run(run, event)

    def apply_simulation_event(self, event: RunEvent) -> ApplyEventResult:
        """Apply an Event only when it belongs to the bound simulation task."""

        if not isinstance(event, RunEvent):
            raise InvalidRunEvent("event must be a RunEvent")
        run = self._repository.get_run(event.run_id)
        if run.execution_mode != ExecutionMode.SIMULATION:
            raise RunTaskMismatch("Run execution_mode must be simulation")
        if run.simulation_task_id is None:
            raise RunTaskMismatch("simulation Run has no bound task_id")
        self._validate_event_task(run, event)
        return self._apply_event_to_run(run, event)

    def _apply_event_to_run(
        self, run: Run, event: RunEvent
    ) -> ApplyEventResult:
        # Check this Run's persisted history before aggregation so a replay of
        # an older state Event remains idempotent, while a newly invalid Event
        # is not written before its state/payload rules have passed.
        event_already_stored = False
        for existing in self._repository.list_events(run.run_id):
            if existing.event_id == event.event_id:
                if existing == event:
                    if event.sequence <= run.last_sequence or is_terminal(run.status):
                        return ApplyEventResult.DUPLICATE
                    # A prior append may have committed before a transient
                    # snapshot-save failure. Reapply below to repair it.
                    event_already_stored = True
                    break
                raise EventConflict(event.event_id)
            if existing.sequence == event.sequence:
                raise EventSequenceConflict(event.run_id, event.sequence)
        if event.sequence <= run.last_sequence or is_terminal(run.status):
            if not event_already_stored:
                append_result = self._repository.append_event(event)
                if append_result == AppendEventResult.DUPLICATE:
                    return ApplyEventResult.DUPLICATE
            return ApplyEventResult.LATE

        now = self._clock()
        if event.type == RunEventType.STATE:
            updated = self._apply_state_event(run, event, now)
        elif event.type == RunEventType.METRIC:
            updated = self._apply_metric_event(run, event, now)
        elif event.type == RunEventType.HEARTBEAT:
            updated = self._apply_heartbeat_event(run, event, now)
        elif event.type == RunEventType.ERROR:
            updated = self._apply_error_event(run, event, now)
        else:  # RunEvent validates this, but keep the service boundary explicit.
            raise InvalidRunEvent("unsupported event type: %s" % event.type)

        if not event_already_stored:
            append_result = self._repository.append_event(event)
            if append_result == AppendEventResult.DUPLICATE:
                return ApplyEventResult.DUPLICATE
        self._repository.save_run(replace(updated, last_sequence=event.sequence))
        if event_already_stored:
            return ApplyEventResult.DUPLICATE
        return ApplyEventResult.APPLIED

    def request_stop(self, run_id: str) -> Run:
        run = self._repository.get_run(run_id)
        if run.status == RunStatus.QUEUED:
            if (
                run.execution_mode == ExecutionMode.SIMULATION
                and run.simulation_task_id is not None
            ):
                return self.transition_status(run_id, RunStatus.STOPPING)
            return self.transition_status(run_id, RunStatus.STOPPED)
        if run.status == RunStatus.RUNNING:
            return self.transition_status(run_id, RunStatus.STOPPING)
        if run.status in {
            RunStatus.STOPPING,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.STOPPED,
        }:
            return run
        if run.status in {RunStatus.INITIALIZING, RunStatus.FINALIZING}:
            raise RunStopNotAllowed(run_id, run.status)
        raise RunStopNotAllowed(run_id, run.status)

    @staticmethod
    def _transition(run: Run, target: RunStatus, now: datetime) -> Run:
        validate_transition(run.status, target)
        started_at = run.started_at
        ended_at = run.ended_at
        if started_at is None and target in {
            RunStatus.INITIALIZING,
            RunStatus.RUNNING,
        }:
            started_at = now
        if ended_at is None and is_terminal(target):
            ended_at = now
        progress = calculate_run_progress(
            run, status=target, training_steps=run.training_steps
        )
        return replace(
            run,
            status=target,
            progress=progress,
            started_at=started_at,
            ended_at=ended_at,
            updated_at=now,
        )

    @staticmethod
    def _validate_event_task(run: Run, event: RunEvent) -> None:
        if run.execution_mode != ExecutionMode.SIMULATION:
            return
        if event.task_id is None or not event.task_id.strip():
            raise RunTaskMismatch("simulation Event requires task_id")
        if event.task_id != run.simulation_task_id:
            raise RunTaskMismatch(
                "simulation Event task_id %s does not match bound task %s"
                % (event.task_id, run.simulation_task_id)
            )

    def _apply_state_event(
        self, run: Run, event: RunEvent, now: datetime
    ) -> Run:
        payload = event.payload
        previous_value = payload.get("previous_status")
        target_value = payload.get("status")
        try:
            previous = RunStatus(previous_value)
            target = RunStatus(target_value)
        except (TypeError, ValueError) as exc:
            raise InvalidRunEvent(
                "state payload previous_status and status must be known Run statuses"
            ) from exc
        if previous != run.status:
            raise InvalidRunEvent(
                "state payload previous_status %s does not match Run status %s"
                % (previous.value, run.status.value)
            )
        return self._transition(run, target, now)

    @staticmethod
    def _apply_metric_event(run: Run, event: RunEvent, now: datetime) -> Run:
        payload = event.payload
        step = payload.get("step")
        metrics = payload.get("metrics")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise InvalidRunEvent("metric payload.step must be a non-negative integer")
        if not isinstance(metrics, Mapping) or not metrics:
            raise InvalidRunEvent("metric payload.metrics must be a non-empty object")

        summary = deepcopy(run.metrics_summary)
        for key, value in metrics.items():
            if not isinstance(key, str) or not key.strip():
                raise InvalidRunEvent("metric names must be non-empty strings")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise InvalidRunEvent("metric %s must be a finite number" % key)
            if key == "update_index":
                update_index = validate_update_index(
                    value, "metric payload.metrics.update_index"
                )
                previous = summary.get("update_index")
                if (
                    not isinstance(previous, bool)
                    and isinstance(previous, int)
                    and previous >= 1
                ):
                    update_index = max(previous, update_index)
                summary[key] = update_index
            else:
                summary[key] = value
        previous_step = summary.get("step")
        if (
            isinstance(previous_step, bool)
            or not isinstance(previous_step, int)
            or previous_step < 0
        ):
            previous_step = 0
        summary["step"] = max(previous_step, step)

        training_steps = max(run.training_steps, step)
        progress = calculate_run_progress(run, training_steps=training_steps)
        if run.execution_mode == ExecutionMode.MOCK and "progress" in payload:
            candidate = payload["progress"]
            if (
                isinstance(candidate, bool)
                or not isinstance(candidate, (int, float))
                or not math.isfinite(candidate)
                or not 0 <= candidate <= 1
            ):
                raise InvalidRunEvent("metric payload.progress must be between 0 and 1")
            progress = min(
                max(run.progress, progress, float(candidate)),
                MAX_ACTIVE_PROGRESS,
            )
        return replace(
            run,
            metrics_summary=summary,
            progress=progress,
            training_steps=training_steps,
            updated_at=now,
        )

    @staticmethod
    def _apply_heartbeat_event(run: Run, event: RunEvent, now: datetime) -> Run:
        payload = event.payload
        try:
            status = RunStatus(payload.get("status"))
        except (TypeError, ValueError) as exc:
            raise InvalidRunEvent("heartbeat payload.status is invalid") from exc
        training_steps = payload.get("training_steps")
        if status not in {
            RunStatus.INITIALIZING,
            RunStatus.RUNNING,
            RunStatus.FINALIZING,
            RunStatus.STOPPING,
        }:
            raise InvalidRunEvent("heartbeat status must be active")
        if (
            isinstance(training_steps, bool)
            or not isinstance(training_steps, int)
            or training_steps < 0
        ):
            raise InvalidRunEvent(
                "heartbeat payload.training_steps must be a non-negative integer"
            )
        training_steps = max(run.training_steps, training_steps)
        return replace(
            run,
            training_steps=training_steps,
            progress=calculate_run_progress(run, training_steps=training_steps),
            updated_at=now,
        )

    @staticmethod
    def _apply_error_event(run: Run, event: RunEvent, now: datetime) -> Run:
        payload = event.payload
        code = payload.get("code")
        message = payload.get("message")
        retryable = payload.get("retryable")
        if not isinstance(code, str) or not code.strip():
            raise InvalidRunEvent("error payload.code must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise InvalidRunEvent("error payload.message must be a non-empty string")
        if not isinstance(retryable, bool):
            raise InvalidRunEvent("error payload.retryable must be a boolean")
        error = RunError(
            code=code,
            message=message[:MAX_ERROR_MESSAGE_LENGTH],
            retryable=retryable,
        )
        return replace(run, error=error, updated_at=now)

