"""Local and Redis-backed cancellation stores and polling helpers."""

import time

from .ports import CancellationStore
from .redis_repository import RedisDependencyError


class CancellationStoreError(RuntimeError):
    pass


class RedisCancellationStore(CancellationStore):
    def __init__(
        self, client, namespace: str = "simulation", request_ttl: int = 86400
    ) -> None:
        if request_ttl <= 0:
            raise ValueError("request_ttl must be positive")
        self._client = client
        self.namespace = namespace
        self.request_ttl = int(request_ttl)

    @property
    def client(self):
        return self._client

    @classmethod
    def from_url(
        cls,
        redis_url: str,
        namespace: str = "simulation",
        request_ttl: int = 86400,
    ):
        try:
            import redis
        except ImportError as exc:
            raise RedisDependencyError(
                "Python package 'redis' is required for Redis cancellation"
            ) from exc
        try:
            client = redis.Redis.from_url(redis_url, decode_responses=False)
            client.ping()
        except Exception as exc:
            raise CancellationStoreError(
                "cannot connect to Redis cancellation store: %s" % exc
            ) from exc
        return cls(client, namespace=namespace, request_ttl=request_ttl)

    @classmethod
    def from_config(cls, config):
        return cls.from_url(
            str(config["redis_url"]),
            namespace=str(config.get("namespace", "simulation")),
            request_ttl=int(config.get("request_ttl", 86400)),
        )

    def to_config(self, redis_url: str):
        return {
            "redis_url": redis_url,
            "namespace": self.namespace,
            "request_ttl": self.request_ttl,
        }

    def request(self, task_id: str) -> None:
        try:
            self._client.set(self._key(task_id), "1", ex=self.request_ttl)
        except Exception as exc:
            raise CancellationStoreError("Redis cancel request failed: %s" % exc) from exc

    def is_requested(self, task_id: str) -> bool:
        try:
            return self._client.get(self._key(task_id)) is not None
        except Exception as exc:
            raise CancellationStoreError("Redis cancel check failed: %s" % exc) from exc

    def clear(self, task_id: str) -> None:
        try:
            self._client.delete(self._key(task_id))
        except Exception as exc:
            raise CancellationStoreError("Redis cancel clear failed: %s" % exc) from exc

    def _key(self, task_id: str) -> str:
        return "%s:cancel:%s" % (self.namespace, task_id)


class PollingCancellationChecker:
    """Combine immediate local Event checks with throttled shared-store polls."""

    def __init__(
        self,
        task_id: str,
        local_checker,
        shared_store=None,
        poll_interval: float = 0.05,
    ) -> None:
        if poll_interval < 0:
            raise ValueError("poll_interval must not be negative")
        self._task_id = task_id
        self._local_checker = local_checker
        self._shared_store = shared_store
        self._poll_interval = float(poll_interval)
        self._next_poll = 0.0
        self._requested = False

    def __call__(self) -> bool:
        if self._requested or self._local_checker():
            self._requested = True
            return True
        if self._shared_store is None:
            return False
        now = time.monotonic()
        if now >= self._next_poll:
            self._next_poll = now + self._poll_interval
            self._requested = self._shared_store.is_requested(self._task_id)
        return self._requested
