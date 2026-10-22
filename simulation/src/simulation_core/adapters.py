"""In-memory and Redis recording adapters used before the real HTTP backend."""

import json
import threading
from typing import Dict, List, Optional

from .domain import SimulationTask, TaskNotFound
from .ports import BackendClient, CancellationStore, Scheduler, TaskRepository


class InMemoryTaskRepository(TaskRepository):
    def __init__(self) -> None:
        self._by_task_id: Dict[str, SimulationTask] = {}
        self._task_id_by_run_id: Dict[str, str] = {}
        self._lock = threading.RLock()

    def get_by_task_id(self, task_id: str) -> Optional[SimulationTask]:
        with self._lock:
            return self._by_task_id.get(task_id)

    def get_by_run_id(self, run_id: str) -> Optional[SimulationTask]:
        with self._lock:
            task_id = self._task_id_by_run_id.get(run_id)
            return self._by_task_id.get(task_id) if task_id is not None else None

    def has_run_id(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._task_id_by_run_id

    def save(self, task: SimulationTask) -> None:
        with self._lock:
            existing_task = self.get_by_run_id(task.run_id)
            if existing_task is not None and existing_task.task_id != task.task_id:
                raise ValueError("run_id already belongs to another task: %s" % task.run_id)
            if task.task_id in self._by_task_id:
                raise ValueError("task_id already exists: %s" % task.task_id)
            self._by_task_id[task.task_id] = task
            self._task_id_by_run_id[task.run_id] = task.task_id

    def update(self, task: SimulationTask) -> None:
        with self._lock:
            current = self._by_task_id.get(task.task_id)
            if current is None:
                raise TaskNotFound("task not found: %s" % task.task_id)
            if current.run_id != task.run_id:
                raise ValueError("run_id cannot change for an existing task")
            self._by_task_id[task.task_id] = task

    def transition(self, task_id, target):
        with self._lock:
            return super().transition(task_id, target)

    def __len__(self) -> int:
        return len(self._by_task_id)


class FakeScheduler(Scheduler):
    def __init__(self) -> None:
        self.submitted_task_ids: List[str] = []
        self.cancelled_task_ids: List[str] = []

    def submit(self, task_id: str) -> None:
        self.submitted_task_ids.append(task_id)

    def cancel(self, task_id: str) -> bool:
        if task_id not in self.cancelled_task_ids:
            self.cancelled_task_ids.append(task_id)
        return True


class NoopScheduler(Scheduler):
    """RQ job-side scheduler: a dequeued job must never enqueue itself."""

    def submit(self, task_id: str) -> None:
        return None

    def cancel(self, task_id: str) -> bool:
        return True


class FakeBackendClient(BackendClient):
    def __init__(self) -> None:
        self.reported_tasks: List[SimulationTask] = []
        self.events = []
        self.terminal_callbacks = []

    def report_status(self, task: SimulationTask) -> None:
        self.reported_tasks.append(task)

    def publish_event(self, event) -> None:
        self.events.append(event)

    def publish_terminal_callback(self, callback) -> None:
        self.terminal_callbacks.append(callback)

    def events_for_run(self, run_id: str):
        return [item for item in self.events if item.run_id == run_id]

    def terminal_callbacks_for_run(self, run_id: str):
        return [item for item in self.terminal_callbacks if item.run_id == run_id]


class FailingBackendClient(FakeBackendClient):
    """Configurable terminal-delivery failure fake for dispatcher tests."""

    def __init__(self, failures_before_success=1, before_terminal_publish=None):
        super().__init__()
        self.failures_before_success = int(failures_before_success)
        self.before_terminal_publish = before_terminal_publish
        self.terminal_attempts = []

    def publish_terminal_callback(self, callback) -> None:
        self.terminal_attempts.append(callback)
        if self.before_terminal_publish is not None:
            self.before_terminal_publish(callback)
        if len(self.terminal_attempts) <= self.failures_before_success:
            raise RuntimeError("configured backend callback failure")
        super().publish_terminal_callback(callback)


RecordingBackendClient = FakeBackendClient


class RedisRecordingBackendClient(BackendClient):
    """Cross-process test sink. This is deliberately not an HTTP adapter."""

    def __init__(self, client, namespace="simulation") -> None:
        self._client = client
        self.namespace = namespace

    def report_status(self, task: SimulationTask) -> None:
        return None

    def publish_event(self, event) -> None:
        self._client.rpush(
            self._events_key(event.run_id),
            json.dumps(event.to_dict(), separators=(",", ":"), sort_keys=True),
        )

    def publish_terminal_callback(self, callback) -> None:
        self._client.rpush(
            self._callbacks_key(callback.run_id),
            json.dumps(callback.to_dict(), separators=(",", ":"), sort_keys=True),
        )

    def events_for_run(self, run_id: str):
        values = self._client.lrange(self._events_key(run_id), 0, -1)
        return [json.loads(item.decode() if isinstance(item, bytes) else item) for item in values]

    def terminal_callbacks_for_run(self, run_id: str):
        values = self._client.lrange(self._callbacks_key(run_id), 0, -1)
        return [json.loads(item.decode() if isinstance(item, bytes) else item) for item in values]

    def _events_key(self, run_id):
        return "%s:run:%s:recorded_events" % (self.namespace, run_id)

    def _callbacks_key(self, run_id):
        return "%s:run:%s:recorded_callbacks" % (self.namespace, run_id)


class NullBackendClient(BackendClient):
    """RQ-side status sink; RedisTaskRepository remains the shared authority."""

    def report_status(self, task: SimulationTask) -> None:
        return None

    def publish_event(self, event) -> None:
        return None

    def publish_terminal_callback(self, callback) -> None:
        return None


class InMemoryCancellationStore(CancellationStore):
    def __init__(self) -> None:
        self._requested = set()
        self._lock = threading.Lock()

    def request(self, task_id: str) -> None:
        with self._lock:
            self._requested.add(task_id)

    def is_requested(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._requested

    def clear(self, task_id: str) -> None:
        with self._lock:
            self._requested.discard(task_id)
