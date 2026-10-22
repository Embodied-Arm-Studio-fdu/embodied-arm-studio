"""Application service coordinating domain rules and replaceable ports."""

from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from uuid import uuid4

from .domain import (
    InvalidStateTransition,
    SimulationTask,
    TaskConfig,
    TaskNotFound,
    TaskStatus,
    utc_now,
)
from .backend_events import BackendMessageFactory, RedisRunSequenceGenerator
from .callback_outbox import (
    CallbackDispatcher,
    InMemoryCallbackOutbox,
    RedisCallbackOutbox,
)
from .ports import BackendClient, Scheduler, TaskQueue, TaskRepository


class StopNotAllowed(ValueError):
    """Raised when Phase 1 has no specified stop transition for a task state."""


class SimulationService:
    def __init__(
        self,
        repository: TaskRepository,
        scheduler: Scheduler,
        backend: BackendClient,
        task_id_factory: Callable[[], str] = None,
        cancellation_store=None,
        artifacts_root: Optional[Path] = None,
        message_factory=None,
        callback_outbox=None,
        callback_dispatcher=None,
        task_lifecycle=None,
    ) -> None:
        self._repository = repository
        self._scheduler = scheduler
        self._backend = backend
        self._task_id_factory = task_id_factory or (lambda: str(uuid4()))
        self._cancellation_store = cancellation_store
        self._artifacts_root = Path(artifacts_root) if artifacts_root is not None else None
        if message_factory is None and hasattr(repository, "client") and hasattr(repository, "namespace"):
            message_factory = BackendMessageFactory(
                RedisRunSequenceGenerator(repository.client, repository.namespace)
            )
        self._messages = message_factory or BackendMessageFactory()
        if callback_outbox is None and hasattr(repository, "client") and hasattr(repository, "namespace"):
            callback_outbox = RedisCallbackOutbox(repository.client, repository.namespace)
        self._callback_outbox = callback_outbox or InMemoryCallbackOutbox()
        self._callback_dispatcher = callback_dispatcher or CallbackDispatcher(
            self._callback_outbox, backend
        )
        self._task_lifecycle = task_lifecycle

    @property
    def callback_outbox(self):
        return self._callback_outbox

    @property
    def callback_dispatcher(self):
        return self._callback_dispatcher

    def create_task(
        self,
        run_id: str,
        project_id: str,
        config: Mapping[str, Any] = None,
    ) -> SimulationTask:
        existing = self._repository.get_by_run_id(run_id)
        if existing is not None:
            # Queue enqueue is idempotent by task_id. Retrying a create call can
            # therefore repair a prior Redis/RQ enqueue failure after the Task
            # JSON was already committed. Legacy Scheduler behavior is kept.
            if existing.status == TaskStatus.QUEUED and isinstance(
                self._scheduler, TaskQueue
            ):
                self._scheduler.submit(existing.task_id)
            return existing

        now = utc_now()
        task = SimulationTask(
            task_id=self._task_id_factory(),
            run_id=run_id,
            project_id=project_id,
            status=TaskStatus.QUEUED,
            created_at=now,
            updated_at=now,
            config=TaskConfig.from_mapping(config or {}),
        )
        try:
            self._repository.save(task)
        except ValueError:
            # A Redis-backed repository can observe another process winning the
            # same run_id between the initial read and atomic save.
            concurrent = self._repository.get_by_run_id(run_id)
            if concurrent is None:
                raise
            return concurrent
        self._scheduler.submit(task.task_id)
        self._backend.report_status(task)
        return task

    def get_task(self, task_id: str) -> SimulationTask:
        task = self._repository.get_by_task_id(task_id)
        if task is None:
            raise TaskNotFound("task not found: %s" % task_id)
        return task

    def transition_task_status(
        self, task_id: str, target: TaskStatus
    ) -> SimulationTask:
        previous = self.get_task(task_id)
        updated = self._repository.transition(task_id, target)
        self._backend.report_status(updated)
        if self._task_lifecycle is not None:
            try:
                self._task_lifecycle.on_task_state(updated)
            except Exception:
                # Optional adjunct lifecycle can never fail the business task.
                pass
        if (previous.status, updated.status) in {
            (TaskStatus.QUEUED, TaskStatus.INITIALIZING),
            (TaskStatus.INITIALIZING, TaskStatus.RUNNING),
            (TaskStatus.RUNNING, TaskStatus.FINALIZING),
            (TaskStatus.RUNNING, TaskStatus.STOPPING),
        }:
            self._create_and_publish_safely(
                "state", updated,
                {"previous_status": previous.status.value, "status": updated.status.value},
            )
        return updated

    def publish_metric_record(self, task_id: str, record: Mapping[str, Any]) -> None:
        task = self.get_task(task_id)
        allowed = (
            "update_index", "average_reward", "policy_loss", "value_loss", "entropy",
            "episode_length", "episodes", "approx_kl", "clip_fraction",
            "explained_variance", "tick", "elapsed_seconds",
        )
        metrics = {name: record[name] for name in allowed if name in record}
        self._create_and_publish_safely(
            "metric", task,
            {
                "step": int(record.get("training_steps", record.get("tick", 0))),
                "metrics": metrics,
            },
        )

    def publish_heartbeat(self, task_id: str, training_steps: int = 0) -> None:
        task = self.get_task(task_id)
        if task.status not in {
            TaskStatus.INITIALIZING, TaskStatus.RUNNING, TaskStatus.FINALIZING,
            TaskStatus.STOPPING,
        }:
            return
        self._create_and_publish_safely(
            "heartbeat", task,
            {"status": task.status.value, "training_steps": int(training_steps)},
        )

    def publish_error(self, task_id: str, code: str, message: str) -> None:
        task = self.get_task(task_id)
        self._create_and_publish_safely(
            "error", task,
            {"code": code, "message": message, "retryable": False},
        )

    def _create_and_publish_safely(self, event_type, task, payload) -> None:
        try:
            event = self._messages.event(
                event_type, task.run_id, task.task_id, payload
            )
            self._backend.publish_event(event)
        except Exception:
            # Phase 5.2 reporting is best-effort and must not fail GPU training.
            return

    def request_stop(self, task_id: str) -> SimulationTask:
        task = self.get_task(task_id)
        if task.status in (
            TaskStatus.STOPPING,
            TaskStatus.STOPPED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        ):
            return task
        if task.status == TaskStatus.QUEUED:
            if self._cancellation_store is not None:
                self._cancellation_store.request(task_id)
            cancel = getattr(self._scheduler, "cancel", None)
            if cancel is None:
                raise StopNotAllowed("scheduler does not support queued cancellation")
            cancel(task_id)
            current = self.get_task(task_id)
            if current.status == TaskStatus.QUEUED:
                try:
                    stopped = self.transition_task_status(
                        task_id, TaskStatus.STOPPED
                    )
                except InvalidStateTransition:
                    # The RQ Worker may have atomically claimed the task after
                    # the read. Its child will still observe the cancel flag.
                    current = self.get_task(task_id)
                    if current.status != TaskStatus.QUEUED:
                        return current
                    raise
                self._finalize_queued_stop(stopped)
                return stopped
            task = current
            if task.status in (
                TaskStatus.STOPPING,
                TaskStatus.STOPPED,
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
            ):
                return task
        if task.status == TaskStatus.RUNNING:
            if self._cancellation_store is not None:
                self._cancellation_store.request(task_id)
            try:
                return self.transition_task_status(task_id, TaskStatus.STOPPING)
            except InvalidStateTransition:
                # Another process can reconcile the same shared cancel flag
                # first. Treat its stopping/stopped commit as idempotent.
                current = self.get_task(task_id)
                if current.status in (
                    TaskStatus.STOPPING,
                    TaskStatus.STOPPED,
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                ):
                    return current
                raise

        # Queue cancellation and initialization/finalization interruption require
        # a real queue/Worker contract, so Phase 1 rejects them explicitly.
        raise StopNotAllowed(
            "stop is not defined for task status %s" % task.status.value
        )

    def _finalize_queued_stop(self, task: SimulationTask) -> None:
        from .task_runner import ExecutionTaskSpec, TaskExecutionResult
        if self._artifacts_root is None:
            result = TaskExecutionResult(
                task_id=task.task_id, run_id=task.run_id,
                status=TaskStatus.STOPPED, training_steps=0, update_steps=0,
                metrics_path="", checkpoint_path="", final_metrics_summary={},
            )
            self.enqueue_terminal_callback(
                task.task_id, result, stop_reason="requested_by_user"
            )
            return
        from .artifact_finalizer import ArtifactFinalizer
        from .task_registry import PRODUCTION_TASK_REGISTRY

        task_spec = ExecutionTaskSpec.from_task(task)
        definition = PRODUCTION_TASK_REGISTRY.get(task_spec.definition_id)
        finalizer = ArtifactFinalizer(
            self._artifacts_root, policy=definition.artifact_policy
        )
        finalizer.prepare(
            task_spec, initial_status=TaskStatus.QUEUED
        )
        finalizer.log_event("queued_task_cancelled")
        summary = finalizer.finalize(TaskStatus.STOPPED)
        result = TaskExecutionResult(
            task_id=task.task_id, run_id=task.run_id, status=TaskStatus.STOPPED,
            training_steps=summary.training_steps, update_steps=summary.update_steps,
            metrics_path=summary.metrics_path,
            checkpoint_path=summary.latest_checkpoint_path,
            final_metrics_summary=summary.final_metrics_summary,
        ).with_artifacts(summary)
        self.enqueue_terminal_callback(
            task.task_id, result, stop_reason="requested_by_user"
        )

    def enqueue_terminal_callback(
        self, task_id: str, result, stop_reason=None, error_code=None
    ):
        """Persist one logical terminal callback, then attempt delivery."""
        task = self.get_task(task_id)
        if not task.is_terminal or result.status != task.status:
            raise ValueError("terminal callback requires the committed matching status")
        existing = self._callback_outbox.get_by_task_id(task_id)
        if existing is not None:
            return existing

        artifacts = [dict(item) for item in result.artifacts]
        by_type = {item.get("type"): item.get("path") for item in artifacts}
        model_path = by_type.get("checkpoint_best") or by_type.get("checkpoint_latest")
        model = None
        if model_path:
            model = {
                "type": "checkpoint_best" if by_type.get("checkpoint_best") else "checkpoint_latest",
                "path": model_path,
            }
        error = None
        if task.status == TaskStatus.FAILED:
            source = dict(result.error or {})
            error = {
                "code": error_code or "TRAINING_FAILED",
                "message": str(source.get("message", "training failed")),
                "retryable": False,
            }
        callback = self._messages.terminal_callback(
            task.run_id, task.task_id, status=task.status.value,
            started_at=result.started_at or task.created_at.isoformat(),
            ended_at=result.ended_at or task.updated_at.isoformat(),
            training_steps=int(result.training_steps),
            update_steps=int(result.update_steps),
            stop_reason=stop_reason,
            metrics={
                key: value
                for key, value in result.final_metrics_summary.items()
                if not isinstance(value, bool) and isinstance(value, (int, float))
            }, model=model,
            artifacts=artifacts, error=error, evaluation=result.evaluation,
        )
        # The dispatcher clock is also the retry schedule's time source. Using
        # it for initial persistence keeps injected/manual clocks coherent.
        record = self._callback_outbox.put_if_absent(
            callback, now=self._callback_dispatcher.clock()
        )
        # Persistence always precedes this best-effort attempt.
        return self._callback_dispatcher.dispatch_event(record.event_id)
