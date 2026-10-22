"""JSON Redis-backed SimulationTask repository for cross-process state."""

from datetime import datetime
import json
from typing import Any, Mapping, Optional

from .domain import SimulationTask, TaskConfig, TaskNotFound, TaskStatus
from .ports import TaskRepository


class RedisDependencyError(RuntimeError):
    """Raised when the optional redis Python package is unavailable."""


class RedisRepositoryError(RuntimeError):
    """Raised when a Redis repository operation cannot complete."""


def task_to_json(task: SimulationTask) -> str:
    """Serialize the shared Task protocol as deterministic JSON, never pickle."""
    return json.dumps(
        {
            "task_id": task.task_id,
            "run_id": task.run_id,
            "project_id": task.project_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "config": dict(task.config.values),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def task_from_json(payload: Any) -> SimulationTask:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        values = json.loads(str(payload))
        return SimulationTask(
            task_id=str(values["task_id"]),
            run_id=str(values["run_id"]),
            project_id=str(values["project_id"]),
            status=TaskStatus(values["status"]),
            created_at=datetime.fromisoformat(values["created_at"]),
            updated_at=datetime.fromisoformat(values["updated_at"]),
            config=TaskConfig.from_mapping(values["config"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RedisRepositoryError("invalid SimulationTask JSON in Redis") from exc


class RedisTaskRepository(TaskRepository):
    """Redis hash-free repository using JSON task values and run-id indexes."""

    def __init__(self, client, namespace: str = "simulation") -> None:
        if not namespace or any(character.isspace() for character in namespace):
            raise ValueError("Redis namespace must be non-empty without whitespace")
        self._client = client
        self.namespace = namespace

    @property
    def client(self):
        return self._client

    @classmethod
    def from_url(cls, redis_url: str, namespace: str = "simulation"):
        try:
            import redis
        except ImportError as exc:
            raise RedisDependencyError(
                "Python package 'redis' is required for RedisTaskRepository"
            ) from exc
        try:
            client = redis.Redis.from_url(redis_url, decode_responses=False)
            client.ping()
        except Exception as exc:
            raise RedisRepositoryError(
                "cannot connect to Redis for SimulationTask repository: %s" % exc
            ) from exc
        return cls(client, namespace=namespace)

    def get_by_task_id(self, task_id: str) -> Optional[SimulationTask]:
        try:
            payload = self._client.get(self._task_key(task_id))
        except Exception as exc:
            raise self._connection_error("get task", exc) from exc
        return task_from_json(payload) if payload is not None else None

    def get_by_run_id(self, run_id: str) -> Optional[SimulationTask]:
        try:
            task_id = self._client.get(self._run_key(run_id))
        except Exception as exc:
            raise self._connection_error("get run index", exc) from exc
        if task_id is None:
            return None
        if isinstance(task_id, bytes):
            task_id = task_id.decode("utf-8")
        return self.get_by_task_id(str(task_id))

    def has_run_id(self, run_id: str) -> bool:
        return self.get_by_run_id(run_id) is not None

    def save(self, task: SimulationTask) -> None:
        task_key = self._task_key(task.task_id)
        run_key = self._run_key(task.run_id)

        def operation(pipe):
            existing_task_id = pipe.get(run_key)
            existing_task = pipe.get(task_key)
            if existing_task_id is not None:
                if isinstance(existing_task_id, bytes):
                    existing_task_id = existing_task_id.decode("utf-8")
                if str(existing_task_id) == task.task_id and existing_task is not None:
                    return False
                raise ValueError("run_id already belongs to another task: %s" % task.run_id)
            if existing_task is not None:
                raise ValueError("task_id already exists: %s" % task.task_id)
            pipe.multi()
            pipe.set(task_key, task_to_json(task))
            pipe.set(run_key, task.task_id)
            return True

        self._watch_transaction((task_key, run_key), operation)

    def update(self, task: SimulationTask) -> None:
        task_key = self._task_key(task.task_id)

        def operation(pipe):
            payload = pipe.get(task_key)
            if payload is None:
                raise TaskNotFound("task not found: %s" % task.task_id)
            current = task_from_json(payload)
            if current.run_id != task.run_id:
                raise ValueError("run_id cannot change for an existing task")
            pipe.multi()
            pipe.set(task_key, task_to_json(task))
            return True

        self._watch_transaction((task_key,), operation)

    def transition(self, task_id: str, target: TaskStatus) -> SimulationTask:
        task_key = self._task_key(task_id)
        updated_holder = []

        def operation(pipe):
            payload = pipe.get(task_key)
            if payload is None:
                raise TaskNotFound("task not found: %s" % task_id)
            updated = task_from_json(payload).transition_to(target)
            pipe.multi()
            pipe.set(task_key, task_to_json(updated))
            updated_holder.append(updated)
            return True

        self._watch_transaction((task_key,), operation)
        return updated_holder[-1]

    def _watch_transaction(self, keys, operation) -> None:
        for _ in range(8):
            pipe = None
            try:
                pipe = self._client.pipeline()
                pipe.watch(*keys)
                should_execute = operation(pipe)
                if should_execute:
                    pipe.execute()
                return
            except (TaskNotFound, ValueError, RedisRepositoryError):
                raise
            except Exception as exc:
                if type(exc).__name__ == "WatchError":
                    continue
                raise self._connection_error("transaction", exc) from exc
            finally:
                if pipe is not None:
                    try:
                        pipe.reset()
                    except Exception:
                        pass
        raise RedisRepositoryError("Redis task transaction conflicted too many times")

    def _task_key(self, task_id: str) -> str:
        return "%s:task:%s" % (self.namespace, task_id)

    def _run_key(self, run_id: str) -> str:
        return "%s:run:%s" % (self.namespace, run_id)

    @staticmethod
    def _connection_error(operation: str, exc: Exception) -> RedisRepositoryError:
        return RedisRepositoryError("Redis %s failed: %s" % (operation, exc))
