import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/start_rq_worker.py"
SPEC = importlib.util.spec_from_file_location("start_rq_worker", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeConnection:
    def __init__(self, ttl_by_key=None):
        self.ttl_by_key = ttl_by_key or {}

    def ttl(self, key):
        return self.ttl_by_key.get(key, -2)


class FakeQueue:
    def __init__(self, name, connection):
        self.name = name
        self.connection = connection


class FakeWorker:
    workers_by_connection_and_queue = {}

    def __init__(self, key, heartbeat, state="idle", worker_ttl=420):
        self.key = key
        self.last_heartbeat = heartbeat
        self.state = state
        self.worker_ttl = worker_ttl

    def refresh(self):
        return None

    @classmethod
    def all(cls, queue):
        return cls.workers_by_connection_and_queue.get(
            (id(queue.connection), queue.name), []
        )


class RedisFactory:
    def __init__(self, connections):
        self.connections = connections

    def __call__(self, url):
        return self.connections[url]


class WorkerStartDetectionTestCase(unittest.TestCase):
    def setUp(self):
        FakeWorker.workers_by_connection_and_queue = {}
        self.now = datetime(2026, 9, 13, tzinfo=timezone.utc)
        self.redis_a = FakeConnection()
        self.redis_b = FakeConnection()
        self.factory = RedisFactory({"redis://a/0": self.redis_a, "redis://b/0": self.redis_b})

    def detect(self, redis_url="redis://a/0", queue_name="simulation-gpu"):
        return MODULE.live_workers(
            redis_url,
            queue_name,
            redis_factory=self.factory,
            queue_factory=FakeQueue,
            worker_class=FakeWorker,
            now=self.now,
        )

    def register(self, connection, queue, worker, ttl=120):
        connection.ttl_by_key[worker.key] = ttl
        FakeWorker.workers_by_connection_and_queue[(id(connection), queue)] = [worker]

    def test_no_worker_allows_start(self):
        self.assertEqual([], self.detect())

    def test_alive_worker_blocks_duplicate_start(self):
        worker = FakeWorker("rq:worker:alive", self.now - timedelta(seconds=30))
        self.register(self.redis_a, "simulation-gpu", worker)
        self.assertEqual([worker], self.detect())

    def test_stale_or_dead_worker_does_not_block_start(self):
        stale = FakeWorker("rq:worker:stale", self.now - timedelta(seconds=481))
        self.register(self.redis_a, "simulation-gpu", stale, ttl=120)
        self.assertEqual([], self.detect())
        stale.last_heartbeat = self.now - timedelta(seconds=10)
        self.redis_a.ttl_by_key[stale.key] = -2
        self.assertEqual([], self.detect())

    def test_worker_on_different_queue_does_not_block_start(self):
        worker = FakeWorker("rq:worker:other-queue", self.now - timedelta(seconds=10))
        self.register(self.redis_a, "other-gpu", worker)
        self.assertEqual([], self.detect(queue_name="simulation-gpu"))

    def test_worker_on_different_redis_does_not_block_start(self):
        worker = FakeWorker("rq:worker:other-redis", self.now - timedelta(seconds=10))
        self.register(self.redis_b, "simulation-gpu", worker)
        self.assertEqual([], self.detect(redis_url="redis://a/0"))


if __name__ == "__main__":
    unittest.main()
