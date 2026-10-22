"""In-memory test queue and optional Redis/RQ queue adapter."""

from collections import deque
from typing import Callable, Dict, Optional

from .ports import TaskQueue


class QueueDependencyError(RuntimeError):
    pass


class TaskQueueError(RuntimeError):
    pass


class InMemoryTaskQueue(TaskQueue):
    """Deterministic FIFO fake; execute_next is explicit and GPU-free."""

    def __init__(self) -> None:
        self._order = deque()
        self._statuses: Dict[str, str] = {}

    def enqueue(self, task_id: str) -> str:
        if task_id not in self._statuses:
            self._statuses[task_id] = "queued"
            self._order.append(task_id)
        return task_id

    def cancel(self, task_id: str) -> bool:
        status = self._statuses.get(task_id)
        if status is None:
            return False
        if status == "queued":
            self._statuses[task_id] = "canceled"
            return True
        return status == "canceled"

    def get_job_status(self, task_id: str) -> Optional[str]:
        return self._statuses.get(task_id)

    def execute_next(self, handler: Callable[[str], object]):
        while self._order:
            task_id = self._order.popleft()
            if self._statuses.get(task_id) != "queued":
                continue
            self._statuses[task_id] = "started"
            try:
                result = handler(task_id)
            except Exception:
                self._statuses[task_id] = "failed"
                raise
            self._statuses[task_id] = "finished"
            return task_id, result
        return None


class RQTaskQueue(TaskQueue):
    """One-to-one task_id/RQ job_id adapter with optional dependencies."""

    def __init__(
        self,
        queue,
        redis_url: str,
        artifacts_root: str,
        namespace: str = "simulation",
        job_timeout: int = 86400,
    ) -> None:
        self._queue = queue
        self.redis_url = redis_url
        self.artifacts_root = str(artifacts_root)
        self.namespace = namespace
        self.job_timeout = int(job_timeout)

    @classmethod
    def from_url(
        cls,
        redis_url: str,
        artifacts_root: str,
        queue_name: str = "simulation-gpu",
        namespace: str = "simulation",
        job_timeout: int = 86400,
    ):
        try:
            import redis
            from rq import Queue
        except ImportError as exc:
            raise QueueDependencyError(
                "Python packages 'redis' and 'rq' are required for RQTaskQueue"
            ) from exc
        try:
            connection = redis.Redis.from_url(redis_url, decode_responses=False)
            connection.ping()
            queue = Queue(queue_name, connection=connection)
        except Exception as exc:
            raise TaskQueueError("cannot connect RQTaskQueue to Redis: %s" % exc) from exc
        return cls(
            queue,
            redis_url=redis_url,
            artifacts_root=artifacts_root,
            namespace=namespace,
            job_timeout=job_timeout,
        )

    def enqueue(self, task_id: str) -> str:
        try:
            existing = self._queue.fetch_job(task_id)
            if existing is not None:
                return task_id
            self._queue.enqueue(
                "simulation_core.rq_jobs.run_training_task",
                task_id,
                redis_url=self.redis_url,
                artifacts_root=self.artifacts_root,
                namespace=self.namespace,
                job_id=task_id,
                job_timeout=self.job_timeout,
                result_ttl=86400,
                failure_ttl=86400,
            )
            return task_id
        except Exception as exc:
            raise TaskQueueError("RQ enqueue failed for %s: %s" % (task_id, exc)) from exc

    def cancel(self, task_id: str) -> bool:
        try:
            job = self._queue.fetch_job(task_id)
            if job is None:
                return False
            status = self._status_value(job.get_status(refresh=True))
            if status in ("queued", "deferred", "scheduled"):
                job.cancel()
                return True
            return status == "canceled"
        except Exception as exc:
            raise TaskQueueError("RQ cancel failed for %s: %s" % (task_id, exc)) from exc

    def get_job_status(self, task_id: str) -> Optional[str]:
        try:
            job = self._queue.fetch_job(task_id)
            return (
                self._status_value(job.get_status(refresh=True))
                if job is not None
                else None
            )
        except Exception as exc:
            raise TaskQueueError(
                "RQ status lookup failed for %s: %s" % (task_id, exc)
            ) from exc

    @staticmethod
    def _status_value(status) -> str:
        """Normalize both legacy strings and newer RQ JobStatus enums."""
        return str(status.value if hasattr(status, "value") else status)
