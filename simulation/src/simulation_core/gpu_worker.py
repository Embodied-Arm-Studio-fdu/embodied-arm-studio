"""Single-task GPU Worker and isolated training-process lifecycle."""

from dataclasses import replace
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import signal
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Optional

from .artifact_finalizer import ArtifactFinalizer
from .cancellation import PollingCancellationChecker, RedisCancellationStore
from .domain import TaskStatus
from .service import SimulationService
from .task_registry import PRODUCTION_TASK_REGISTRY
from .task_runner import ExecutionTaskSpec, TaskExecutionResult, TaskRunner


class WorkerBusyError(RuntimeError):
    """Raised when the single-task Worker already owns a training process."""


class TaskNotExecutableError(RuntimeError):
    """Raised when a task is not queued and cannot be started by this Worker."""


class ActiveTaskMismatchError(RuntimeError):
    """Raised when stop targets a task other than the Worker's active task."""


class TrainingProcessError(RuntimeError):
    """Raised when the isolated training process fails or exits abnormally."""

    def __init__(
        self,
        message: str,
        error: Optional[Mapping[str, Any]] = None,
        result: Optional[TaskExecutionResult] = None,
    ) -> None:
        super().__init__(message)
        self.error = dict(error or {})
        self.result = result


def classify_process_exit(exit_code: int) -> Mapping[str, Any]:
    """Return a stable, JSON-safe classification for an abnormal child exit."""
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise TypeError("exit_code must be an integer")
    if exit_code < 0:
        signal_number = -exit_code
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = "SIGNAL_%d" % signal_number
        return {
            "type": "NativeProcessSignal",
            "message": "training process terminated by %s" % signal_name,
            "exit_code": exit_code,
            "signal": signal_name,
            "signal_number": signal_number,
            "retryable": False,
        }
    return {
        "type": "AbnormalProcessExit",
        "message": "training process exited with code %d" % exit_code,
        "exit_code": exit_code,
        "retryable": False,
    }


def _training_process_main(
    task_payload: Mapping[str, Any],
    artifacts_root: str,
    connection: Connection,
    cancel_event,
    shared_cancel_config=None,
) -> None:
    """Child entrypoint. All real simulator/trainer construction happens here."""
    task = None
    finalizer = None
    runner = None
    try:
        task = ExecutionTaskSpec.from_dict(task_payload)
        definition = PRODUCTION_TASK_REGISTRY.get(task.definition_id)
        finalizer = ArtifactFinalizer(
            Path(artifacts_root), policy=definition.artifact_policy
        )
        finalizer.prepare(task, child_pid=os.getpid())
        cancel_logged = False
        shared_cancel_store = (
            RedisCancellationStore.from_config(shared_cancel_config)
            if shared_cancel_config is not None
            else None
        )
        stream_gateway = None
        if shared_cancel_store is not None:
            from .stream_session import RedisRecordingStreamGateway
            stream_gateway = RedisRecordingStreamGateway(
                shared_cancel_store.client,
                namespace=shared_cancel_config.get("namespace", "simulation"),
            )
        cancellation_checker = PollingCancellationChecker(
            task.task_id,
            local_checker=cancel_event.is_set,
            shared_store=shared_cancel_store,
        )

        def report_progress(status: TaskStatus) -> None:
            finalizer.log_event("status_changed", status=status.value)
            if status == TaskStatus.RUNNING:
                finalizer.log_event("training_started")
            connection.send({"type": "status", "status": status.value})

        def report_metric(record: Mapping[str, Any]) -> None:
            # Only a JSON-compatible completed-update record crosses the Pipe.
            connection.send({"type": "metric", "record": dict(record)})

        def cancel_requested() -> bool:
            nonlocal cancel_logged
            requested = cancellation_checker()
            if requested and not cancel_logged:
                finalizer.log_event("stop_requested")
                cancel_logged = True
            return requested

        failure_injector = None
        if (os.environ.get("SIMULATION_TEST_FAILURE_INJECTION") ==
                "training_after_running"):
            def failure_injector():
                raise RuntimeError("controlled training failure after RUNNING")

        runner = TaskRunner(
            Path(artifacts_root),
            progress_callback=report_progress,
            cancel_requested=cancel_requested,
            metric_callback=report_metric,
            stream_gateway=stream_gateway,
            failure_injector=failure_injector,
        )
        result = None
        child_error = None
        with finalizer.capture_process_output():
            try:
                result = runner.run(task)
            except BaseException as exc:
                child_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                traceback.print_exc()

        if child_error is not None:
            finalizer.log_event(
                "error",
                error_type=child_error["type"],
                message=child_error["message"],
            )
            artifact_summary = finalizer.finalize(
                TaskStatus.FAILED,
                training_steps=runner.training_steps,
                update_steps=runner.update_steps,
                error={
                    "type": child_error["type"],
                    "message": child_error["message"],
                },
            )
            failed_result = TaskExecutionResult(
                task_id=task.task_id,
                run_id=task.run_id,
                status=TaskStatus.FAILED,
                training_steps=artifact_summary.training_steps,
                update_steps=artifact_summary.update_steps,
                metrics_path=artifact_summary.metrics_path,
                checkpoint_path=artifact_summary.latest_checkpoint_path,
                final_metrics_summary=artifact_summary.final_metrics_summary,
                error={
                    "type": child_error["type"],
                    "message": child_error["message"],
                },
            ).with_artifacts(artifact_summary, error=child_error)
            connection.send(
                {
                    "type": "error",
                    "error": child_error,
                    "result": failed_result.to_dict(),
                }
            )
            return

        if result.status == TaskStatus.STOPPED:
            finalizer.log_event("stop_observed")
        artifact_summary = finalizer.finalize(
            result.status,
            training_steps=result.training_steps,
            update_steps=result.update_steps,
            final_metrics_summary=result.final_metrics_summary,
            evaluation=result.evaluation,
        )
        result = result.with_artifacts(artifact_summary)
        connection.send({"type": "result", "result": result.to_dict()})
    except BaseException as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            connection.send({"type": "error", "error": error})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class GPUWorker:
    """Run at most one heavy task and reconcile child events in the parent."""

    def __init__(
        self,
        service: SimulationService,
        artifacts_root: Path,
        multiprocessing_context=None,
        process_target: Optional[Callable[..., None]] = None,
        stop_timeout: float = 10.0,
        shared_cancel_config: Optional[Mapping[str, Any]] = None,
        heartbeat_interval: float = 10.0,
        monotonic_clock: Callable[[], float] = None,
    ) -> None:
        if stop_timeout <= 0:
            raise ValueError("stop_timeout must be positive")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        self._service = service
        self._artifacts_root = Path(artifacts_root)
        self._context = multiprocessing_context or multiprocessing.get_context("spawn")
        self._process_target = process_target or _training_process_main
        self._stop_timeout = float(stop_timeout)
        self._shared_cancel_config = (
            dict(shared_cancel_config) if shared_cancel_config is not None else None
        )
        self._heartbeat_interval = float(heartbeat_interval)
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._busy_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._is_busy = False
        self._active_task_id = None
        self._active_child_pid = None
        self._active_process = None
        self._active_cancel_event = None
        self._active_completion = None
        self._force_terminated = False
        self._last_child_pid = None
        self._last_exit_code = None
        self._last_stop_forced = False
        self._last_result = None
        self._latest_training_steps = 0

    @property
    def is_busy(self) -> bool:
        with self._state_lock:
            return self._is_busy

    @property
    def active_task_id(self) -> Optional[str]:
        with self._state_lock:
            return self._active_task_id

    @property
    def active_child_pid(self) -> Optional[int]:
        with self._state_lock:
            return self._active_child_pid

    @property
    def last_child_pid(self) -> Optional[int]:
        return self._last_child_pid

    @property
    def last_exit_code(self) -> Optional[int]:
        return self._last_exit_code

    @property
    def last_stop_forced(self) -> bool:
        """Whether the most recent stop needed the timeout terminate fallback."""
        return self._last_stop_forced

    @property
    def last_result(self) -> Optional[TaskExecutionResult]:
        return self._last_result

    def run(self, task_id: str) -> TaskExecutionResult:
        if not self._busy_lock.acquire(blocking=False):
            raise WorkerBusyError("GPU Worker is already running a training task")

        completion = threading.Event()
        with self._state_lock:
            self._is_busy = True
            self._active_task_id = task_id
            self._active_completion = completion
            self._force_terminated = False
            self._last_stop_forced = False
            self._last_result = None
            self._latest_training_steps = 0
        process = None
        process_started = False
        receive_connection = None
        send_connection = None
        try:
            task = self._service.get_task(task_id)
            if task.status != TaskStatus.QUEUED:
                raise TaskNotExecutableError(
                    "task %s is not executable from status %s"
                    % (task_id, task.status.value)
                )
            task_spec = ExecutionTaskSpec.from_task(task)
            self._service.transition_task_status(task_id, TaskStatus.INITIALIZING)

            receive_connection, send_connection = self._context.Pipe(duplex=False)
            cancel_event = self._context.Event()
            with self._state_lock:
                self._active_cancel_event = cancel_event
            process = self._context.Process(
                target=self._process_target,
                args=(
                    task_spec.to_dict(),
                    str(self._artifacts_root),
                    send_connection,
                    cancel_event,
                    self._shared_cancel_config,
                ),
                name="simulation-training-%s" % task_id,
            )
            process.start()
            process_started = True
            send_connection.close()
            send_connection = None
            with self._state_lock:
                self._active_process = process
                self._active_child_pid = process.pid
            self._last_child_pid = process.pid

            result = self._monitor_process(task_id, process, receive_connection)
            if result.task_id != task.task_id or result.run_id != task.run_id:
                raise TrainingProcessError(
                    "child returned a result that does not match the task"
                )
            current = self._service.get_task(task_id)
            if (
                result.status == TaskStatus.STOPPED
                and current.status == TaskStatus.RUNNING
            ):
                # A shared cancellation flag can be observed by the child in
                # the small window between the caller writing the flag and
                # committing running -> stopping in Redis. Reconcile through
                # the normal domain path instead of misclassifying the clean
                # cooperative result as a training failure.
                self._service.transition_task_status(task_id, TaskStatus.STOPPING)
                current = self._service.get_task(task_id)
            if current.status == TaskStatus.STOPPING:
                if result.status not in (TaskStatus.STOPPED, TaskStatus.COMPLETED):
                    raise TrainingProcessError("child returned an invalid stop result")
                stopped_result = (
                    result
                    if result.status == TaskStatus.STOPPED
                    else replace(result, status=TaskStatus.STOPPED)
                )
                self._service.transition_task_status(task_id, TaskStatus.STOPPED)
                self._service.enqueue_terminal_callback(
                    task_id, stopped_result,
                    stop_reason=(
                        "stop_timeout" if self._last_stop_forced
                        else "requested_by_user"
                    ),
                )
                self._last_result = stopped_result
                return stopped_result
            if result.status != TaskStatus.COMPLETED:
                raise TrainingProcessError("child returned an unexpected stopped result")
            self._service.transition_task_status(task_id, TaskStatus.COMPLETED)
            # A real child always returns the finalized manifest metadata.
            # Legacy fake process targets without artifacts remain supported.
            if result.manifest_path:
                self._service.enqueue_terminal_callback(task_id, result)
            self._last_result = result
            return result
        except Exception as exc:
            if isinstance(exc, TrainingProcessError) and exc.result is not None:
                self._last_result = exc.result
            elif self._is_failure_finalizable(task_id):
                try:
                    self._last_result = self._finalize_parent_failure(task_id, exc)
                    if isinstance(exc, TrainingProcessError):
                        exc.result = self._last_result
                except Exception:
                    # Preserve the original process/training error. Artifact
                    # finalization failure is diagnostic and must not mask it.
                    pass
            if self._is_failure_finalizable(task_id):
                self._service.publish_error(
                    task_id, self._error_code(task_id), self._error_message(exc)
                )
            error_code = None
            if self._is_failure_finalizable(task_id):
                error_code = self._error_code(task_id)
            self._transition_to_failed(task_id)
            try:
                current = self._service.get_task(task_id)
                if current.status == TaskStatus.FAILED:
                    failure_result = self._last_result or self._bare_failure_result(
                        task_id, exc
                    )
                    self._service.enqueue_terminal_callback(
                        task_id, failure_result,
                        error_code=error_code or "WORKER_FAILED",
                    )
            except Exception:
                # Preserve the original training exception if callback setup fails.
                pass
            raise
        finally:
            if send_connection is not None:
                send_connection.close()
            if receive_connection is not None:
                receive_connection.close()
            if process is not None and process_started:
                if process.is_alive():
                    process.terminate()
                process.join()
                self._last_exit_code = process.exitcode
                process.close()
            elif process is not None:
                process.close()
            with self._state_lock:
                self._active_child_pid = None
                self._active_process = None
                self._active_cancel_event = None
                self._active_task_id = None
                self._is_busy = False
                self._active_completion = None
            completion.set()
            self._busy_lock.release()

    def stop(self, task_id: str):
        """Cooperatively stop the matching active task, escalating only on timeout."""
        with self._state_lock:
            active_task_id = self._active_task_id
            completion = self._active_completion
            cancel_event = self._active_cancel_event

        if active_task_id is not None and active_task_id != task_id:
            raise ActiveTaskMismatchError(
                "task %s does not match active task %s" % (task_id, active_task_id)
            )

        task = self._service.get_task(task_id)
        if task.status in (
            TaskStatus.STOPPED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        ):
            return task
        if active_task_id is None:
            raise TaskNotExecutableError(
                "task %s is not active in this GPU Worker" % task_id
            )

        # The current phase intentionally supports running cancellation only.
        task = self._service.request_stop(task_id)
        if cancel_event is not None:
            cancel_event.set()
        if completion is None or completion.wait(self._stop_timeout):
            return self._service.get_task(task_id)

        # Normal stop never comes through here. terminate() is strictly the
        # bounded fallback for a child that ignored cooperative cancellation.
        process = None
        with self._state_lock:
            if self._active_task_id == task_id and not self._force_terminated:
                self._force_terminated = True
                self._last_stop_forced = True
                process = self._active_process
        if process is not None and process.is_alive():
            process.terminate()
            process.join()

        # The monitor observes the exit and owns state reconciliation/cleanup.
        completion.wait(max(1.0, self._stop_timeout))
        final_task = self._service.get_task(task_id)
        if final_task.status != TaskStatus.STOPPED:
            raise TrainingProcessError(
                "training process cleanup did not finish after forced termination"
            )
        return final_task

    def _monitor_process(
        self, task_id: str, process, connection: Connection
    ) -> TaskExecutionResult:
        result = None
        child_error = None
        child_error_result = None
        connection_open = True
        last_heartbeat = self._monotonic_clock()
        while process.is_alive() or connection_open:
            if connection_open and connection.poll(0.05):
                try:
                    message = connection.recv()
                except EOFError:
                    connection_open = False
                else:
                    message_type = message.get("type")
                    if message_type == "status":
                        self._apply_child_status(task_id, message)
                    elif message_type == "metric":
                        record = dict(message["record"])
                        self._latest_training_steps = int(
                            record.get("training_steps", 0)
                        )
                        self._service.publish_metric_record(task_id, record)
                    elif message_type == "result":
                        result = TaskExecutionResult.from_dict(message["result"])
                    elif message_type == "error":
                        child_error = dict(message["error"])
                        if message.get("result") is not None:
                            child_error_result = TaskExecutionResult.from_dict(
                                message["result"]
                            )
                    else:
                        child_error = {
                            "type": "InvalidChildMessage",
                            "message": "unknown child message: %r" % message,
                        }
            elif not process.is_alive():
                connection_open = False
            now = self._monotonic_clock()
            if now - last_heartbeat >= self._heartbeat_interval:
                self._service.publish_heartbeat(task_id, self._latest_training_steps)
                last_heartbeat = now

        process.join()
        self._last_exit_code = process.exitcode
        if self._was_force_terminated(task_id):
            return self._stopped_result_from_artifacts(task_id)
        if process.exitcode != 0:
            classified = classify_process_exit(process.exitcode)
            raise TrainingProcessError(
                classified["message"],
                classified,
            )
        if child_error is not None:
            raise TrainingProcessError(
                "training process failed: %s" % child_error.get("message", ""),
                child_error,
                result=child_error_result,
            )
        if result is None:
            raise TrainingProcessError("training process exited without a result")
        return result

    def _apply_child_status(self, task_id: str, message: Mapping[str, Any]) -> None:
        target = TaskStatus(message["status"])
        if target not in (TaskStatus.RUNNING, TaskStatus.FINALIZING):
            raise TrainingProcessError(
                "child requested unsupported status %s" % target.value
            )
        current = self._service.get_task(task_id)
        if current.status in (TaskStatus.STOPPING, TaskStatus.STOPPED):
            # A finalizing message may already be in the pipe when stop wins.
            return
        self._service.transition_task_status(task_id, target)

    def _was_force_terminated(self, task_id: str) -> bool:
        with self._state_lock:
            return self._active_task_id == task_id and self._force_terminated

    def _stopped_result_from_artifacts(self, task_id: str) -> TaskExecutionResult:
        task = self._service.get_task(task_id)
        task_spec = ExecutionTaskSpec.from_task(task)
        definition = PRODUCTION_TASK_REGISTRY.get(task_spec.definition_id)
        finalizer = ArtifactFinalizer(
            self._artifacts_root, policy=definition.artifact_policy
        )
        finalizer.prepare(task_spec, child_pid=self._active_child_pid)
        finalizer.log_event("stop_timeout", action="terminate")
        summary = finalizer.finalize(TaskStatus.STOPPED)
        return TaskExecutionResult(
            task_id=task.task_id,
            run_id=task.run_id,
            status=TaskStatus.STOPPED,
            training_steps=summary.training_steps,
            update_steps=summary.update_steps,
            metrics_path=summary.metrics_path,
            checkpoint_path=summary.latest_checkpoint_path,
            final_metrics_summary=summary.final_metrics_summary,
        ).with_artifacts(summary)

    def _is_failure_finalizable(self, task_id: str) -> bool:
        try:
            status = self._service.get_task(task_id).status
        except Exception:
            return False
        return status in (
            TaskStatus.INITIALIZING,
            TaskStatus.RUNNING,
            TaskStatus.FINALIZING,
        )

    def _finalize_parent_failure(
        self, task_id: str, exc: Exception
    ) -> TaskExecutionResult:
        task = self._service.get_task(task_id)
        task_spec = ExecutionTaskSpec.from_task(task)
        definition = PRODUCTION_TASK_REGISTRY.get(task_spec.definition_id)
        finalizer = ArtifactFinalizer(
            self._artifacts_root, policy=definition.artifact_policy
        )
        finalizer.prepare(task_spec, child_pid=self._active_child_pid)
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if isinstance(exc, TrainingProcessError) and exc.error:
            error = dict(exc.error)
            error["type"] = str(error.get("type", type(exc).__name__))
            error["message"] = str(error.get("message", str(exc)))
        finalizer.log_event(
            "error", error_type=error["type"], message=error["message"]
        )
        summary = finalizer.finalize(TaskStatus.FAILED, error=error)
        return TaskExecutionResult(
            task_id=task.task_id,
            run_id=task.run_id,
            status=TaskStatus.FAILED,
            training_steps=summary.training_steps,
            update_steps=summary.update_steps,
            metrics_path=summary.metrics_path,
            checkpoint_path=summary.latest_checkpoint_path,
            final_metrics_summary=summary.final_metrics_summary,
            error=error,
        ).with_artifacts(summary, error=error)

    def _transition_to_failed(self, task_id: str) -> None:
        try:
            current = self._service.get_task(task_id)
        except Exception:
            return
        if current.status in (
            TaskStatus.INITIALIZING,
            TaskStatus.RUNNING,
            TaskStatus.FINALIZING,
        ):
            self._service.transition_task_status(task_id, TaskStatus.FAILED)

    def _bare_failure_result(self, task_id, exc):
        task = self._service.get_task(task_id)
        return TaskExecutionResult(
            task_id=task.task_id, run_id=task.run_id, status=TaskStatus.FAILED,
            training_steps=self._latest_training_steps, update_steps=0,
            metrics_path="", checkpoint_path="", final_metrics_summary={},
            error={"type": type(exc).__name__, "message": self._error_message(exc)},
        )

    def _error_code(self, task_id: str) -> str:
        status = self._service.get_task(task_id).status
        return {
            TaskStatus.INITIALIZING: "INITIALIZATION_FAILED",
            TaskStatus.RUNNING: "TRAINING_FAILED",
            TaskStatus.FINALIZING: "FINALIZATION_FAILED",
        }.get(status, "WORKER_FAILED")

    @staticmethod
    def _error_message(exc: Exception) -> str:
        if isinstance(exc, TrainingProcessError) and exc.error.get("message"):
            return str(exc.error["message"])
        return str(exc)
