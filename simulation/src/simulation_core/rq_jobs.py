"""Stable thin RQ entry points; heavy GPU resources stay in the child process."""

from pathlib import Path
import gc
import logging
import os

from .adapters import NoopScheduler, RedisRecordingBackendClient
from .cancellation import RedisCancellationStore
from .domain import TaskStatus
from .gpu_worker import GPUWorker
from .redis_repository import RedisTaskRepository
from .service import SimulationService
from .task_runner import TaskExecutionResult
from .http_backend import HttpBackendClient
from .resource_tracker_cleanup import cleanup_owned_resource_tracker


LOGGER = logging.getLogger(__name__)


def execute_task(
    task_id,
    service,
    worker_factory,
    cancellation_store=None,
):
    """Dependency-injectable task job core used by GPU-free unit tests."""
    task = service.get_task(task_id)
    if task.status in (
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.STOPPED,
    ):
        return {"task_id": task_id, "status": task.status.value, "skipped": True}
    if cancellation_store is not None and cancellation_store.is_requested(task_id):
        stopped = service.request_stop(task_id)
        return {"task_id": task_id, "status": stopped.status.value, "skipped": True}

    try:
        worker = worker_factory(service)
        result = worker.run(task_id)
        return result.to_dict() if hasattr(result, "to_dict") else result
    except Exception as exc:
        current = service.get_task(task_id)
        if current.status == TaskStatus.QUEUED:
            # A wrapper/factory failure happened before GPUWorker could claim
            # the task. Move through the domain-valid path so it cannot remain
            # queued forever after RQ records the job as failed.
            service.transition_task_status(task_id, TaskStatus.INITIALIZING)
            current = service.get_task(task_id)
        if current.status in (
            TaskStatus.INITIALIZING,
            TaskStatus.RUNNING,
            TaskStatus.FINALIZING,
        ):
            running_failure = (
                "DEMO_FAILED"
                if current.config.values.get("task_type") == "demo"
                else "TRAINING_FAILED"
            )
            code = {
                TaskStatus.INITIALIZING: "INITIALIZATION_FAILED",
                TaskStatus.RUNNING: running_failure,
                TaskStatus.FINALIZING: "FINALIZATION_FAILED",
            }[current.status]
            service.publish_error(task_id, code, str(exc))
            service.transition_task_status(task_id, TaskStatus.FAILED)
            failed_task = service.get_task(task_id)
            service.enqueue_terminal_callback(
                task_id,
                TaskExecutionResult(
                    task_id=failed_task.task_id, run_id=failed_task.run_id,
                    status=TaskStatus.FAILED, training_steps=0, update_steps=0,
                    metrics_path="", checkpoint_path="",
                    final_metrics_summary={},
                    error={"type": type(exc).__name__, "message": str(exc)},
                ),
                error_code=code,
            )
        raise
    finally:
        if cancellation_store is not None:
            current = service.get_task(task_id)
            if current.is_terminal:
                cancellation_store.clear(task_id)


def execute_training_task(
    task_id,
    service,
    worker_factory,
    cancellation_store=None,
):
    """Compatibility facade for callers using the original training name."""
    return execute_task(task_id, service, worker_factory, cancellation_store)


def run_task(
    task_id: str,
    redis_url: str,
    artifacts_root: str,
    namespace: str = "simulation",
):
    """Import-stable RQ job: rebuild shared boundaries, then invoke GPUWorker."""
    repository = RedisTaskRepository.from_url(redis_url, namespace=namespace)
    cancellation_store = RedisCancellationStore.from_url(
        redis_url, namespace=namespace
    )
    backend = HttpBackendClient.from_env(required=False) or RedisRecordingBackendClient(
        repository.client, namespace=namespace
    )
    service = SimulationService(
        repository,
        NoopScheduler(),
        backend,
        cancellation_store=cancellation_store,
        artifacts_root=Path(artifacts_root),
    )
    shared_cancel_config = {
        "redis_url": redis_url,
        "namespace": namespace,
        "request_ttl": cancellation_store.request_ttl,
    }
    try:
        return execute_training_task(
            task_id,
            service,
            worker_factory=lambda current_service: GPUWorker(
                current_service,
                Path(artifacts_root),
                shared_cancel_config=shared_cancel_config,
                heartbeat_interval=float(os.environ.get(
                    "SIMULATION_HEARTBEAT_INTERVAL", "10.0"
                )),
            ),
            cancellation_store=cancellation_store,
        )
    finally:
        # execute_training_task/GPUWorker have already closed Pipes and
        # joined/closed the spawn child.  Collect unreachable SemLock wrappers
        # before stopping the tracker that the short-lived RQ work horse owns.
        try:
            gc.collect()
            cleanup_owned_resource_tracker(logger=LOGGER)
        except Exception:
            # Keep this outer guard even though the helper is best-effort: a
            # cleanup regression must never replace task/callback semantics.
            LOGGER.exception("resource_tracker cleanup escaped its isolation")


def run_training_task(
    task_id: str,
    redis_url: str,
    artifacts_root: str,
    namespace: str = "simulation",
):
    """Import-stable compatibility facade retained for existing RQ jobs."""
    return run_task(task_id, redis_url, artifacts_root, namespace)
