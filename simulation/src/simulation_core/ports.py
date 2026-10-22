"""Replaceable boundaries used by the Phase 1 application service."""

from abc import ABC, abstractmethod
from typing import Optional

from .domain import SimulationTask, TaskNotFound, TaskStatus


class TaskRepository(ABC):
    @abstractmethod
    def get_by_task_id(self, task_id: str) -> Optional[SimulationTask]:
        raise NotImplementedError

    @abstractmethod
    def get_by_run_id(self, run_id: str) -> Optional[SimulationTask]:
        raise NotImplementedError

    @abstractmethod
    def has_run_id(self, run_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def save(self, task: SimulationTask) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, task: SimulationTask) -> None:
        raise NotImplementedError

    def transition(self, task_id: str, target: TaskStatus) -> SimulationTask:
        """Atomically transition when overridden by a shared repository."""
        task = self.get_by_task_id(task_id)
        if task is None:
            raise TaskNotFound("task not found: %s" % task_id)
        updated = task.transition_to(target)
        self.update(updated)
        return updated


class Scheduler(ABC):
    @abstractmethod
    def submit(self, task_id: str) -> None:
        raise NotImplementedError


class TaskQueue(Scheduler):
    """Queue boundary; RQ status is separate from SimulationTask.status."""

    def submit(self, task_id: str) -> None:
        self.enqueue(task_id)

    @abstractmethod
    def enqueue(self, task_id: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_job_status(self, task_id: str) -> Optional[str]:
        raise NotImplementedError


class CancellationStore(ABC):
    """Cross-process cooperative cancellation state."""

    @abstractmethod
    def request(self, task_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_requested(self, task_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def clear(self, task_id: str) -> None:
        raise NotImplementedError


class BackendClient(ABC):
    """Replaceable simulation-to-backend reporting boundary."""

    @abstractmethod
    def report_status(self, task: SimulationTask) -> None:
        raise NotImplementedError

    @abstractmethod
    def publish_event(self, event) -> None:
        raise NotImplementedError

    @abstractmethod
    def publish_terminal_callback(self, callback) -> None:
        raise NotImplementedError
