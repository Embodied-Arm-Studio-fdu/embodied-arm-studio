import json
import tempfile
import unittest
from enum import Enum
from pathlib import Path

from simulation_core import (
    FakeBackendClient,
    InMemoryCancellationStore,
    InMemoryTaskQueue,
    RedisCancellationStore,
    RedisTaskRepository,
    RQTaskQueue,
    SimulationService,
    TaskStatus,
    TrainingTaskConfig,
    task_from_json,
    task_to_json,
)
from simulation_core.rq_jobs import execute_training_task


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []
        self.in_multi = False

    def watch(self, *keys):
        return None

    def get(self, key):
        return self.client.get(key)

    def multi(self):
        self.in_multi = True

    def set(self, key, value):
        self.commands.append((key, value))
        return self

    def execute(self):
        for key, value in self.commands:
            self.client.set(key, value)
        self.commands = []
        return True

    def reset(self):
        self.commands = []
        self.in_multi = False


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.sets = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value.encode() if isinstance(value, str) else value
        return True

    def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    def pipeline(self):
        return FakePipeline(self)

    def incr(self, key):
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = value
        return value

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(
            value.encode() if isinstance(value, str) else value
        )

    def smembers(self, key):
        return self.sets.get(key, set())

    def eval(self, script, count, task_key, record_key, index_key, event_id, payload):
        existing = self.get(task_key)
        if existing is not None:
            return existing
        self.set(task_key, event_id)
        self.set(record_key, payload)
        self.sadd(index_key, event_id)
        return event_id


class FakeRQJob:
    def __init__(self, job_id):
        self.id = job_id
        self.status = "queued"

    def get_status(self, refresh=True):
        return self.status

    def cancel(self):
        self.status = "canceled"


class FakeJobStatus(Enum):
    QUEUED = "queued"


class FakeRQQueue:
    def __init__(self):
        self.jobs = {}
        self.enqueue_calls = []

    def fetch_job(self, job_id):
        return self.jobs.get(job_id)

    def enqueue(self, function, *args, **kwargs):
        job = FakeRQJob(kwargs["job_id"])
        self.jobs[job.id] = job
        self.enqueue_calls.append((function, args, kwargs))
        return job


class FailOnceTaskQueue(InMemoryTaskQueue):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def enqueue(self, task_id):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("temporary queue outage")
        return super().enqueue(task_id)


def training_config(num_updates=1):
    return TrainingTaskConfig(
        task_type="training",
        environment_type="s1_push",
        seed=31,
        rollout_steps=4,
        num_updates=num_updates,
        checkpoint_every=1,
        episode_steps=8,
        physics_steps_per_action=1,
        batch_size=4,
    )


class QueueingTestCase(unittest.TestCase):
    def create_service(self, queue=None, repository=None, cancellation=None, root=None):
        return SimulationService(
            repository or RedisTaskRepository(FakeRedis(), namespace="test"),
            queue or InMemoryTaskQueue(),
            FakeBackendClient(),
            cancellation_store=cancellation,
            artifacts_root=root,
        )

    def test_redis_task_json_round_trip_is_plain_data(self):
        service = self.create_service()
        task = service.create_task(
            "run-json", "project-json", training_config().to_dict()
        )

        payload = task_to_json(task)
        restored = task_from_json(payload)

        self.assertEqual(task, restored)
        decoded = json.loads(payload)
        self.assertIsInstance(decoded, dict)
        self.assertEqual("queued", decoded["status"])
        self.assertNotIn("pickle", payload.lower())

    def test_redis_repository_persists_status_idempotency_and_isolation(self):
        repository = RedisTaskRepository(FakeRedis(), namespace="repository-test")
        queue = InMemoryTaskQueue()
        service = self.create_service(queue=queue, repository=repository)
        first = service.create_task(
            "run-one", "project-redis", training_config().to_dict()
        )
        duplicate = service.create_task(
            "run-one", "ignored-project", training_config().to_dict()
        )
        second = service.create_task(
            "run-two", "project-redis", training_config().to_dict()
        )

        service.transition_task_status(first.task_id, TaskStatus.INITIALIZING)

        self.assertEqual(first.task_id, duplicate.task_id)
        self.assertEqual(TaskStatus.INITIALIZING, repository.get_by_task_id(first.task_id).status)
        self.assertEqual(second.task_id, repository.get_by_run_id("run-two").task_id)
        self.assertNotEqual(first.task_id, second.task_id)
        self.assertEqual(2, len(queue._statuses))

    def test_create_retry_repairs_enqueue_failure_without_duplicate_task(self):
        queue = FailOnceTaskQueue()
        repository = RedisTaskRepository(FakeRedis(), namespace="retry-test")
        service = self.create_service(queue=queue, repository=repository)

        with self.assertRaisesRegex(RuntimeError, "queue outage"):
            service.create_task(
                "run-retry", "project-retry", training_config().to_dict()
            )
        persisted = repository.get_by_run_id("run-retry")
        retried = service.create_task(
            "run-retry", "project-retry", training_config().to_dict()
        )

        self.assertEqual(persisted.task_id, retried.task_id)
        self.assertEqual(2, queue.attempts)
        self.assertEqual("queued", queue.get_job_status(retried.task_id))

    def test_duplicate_rq_enqueue_uses_task_id_as_single_job_id(self):
        fake_rq = FakeRQQueue()
        queue = RQTaskQueue(
            fake_rq,
            redis_url="redis://example.invalid:6379/0",
            artifacts_root="/tmp/artifacts",
        )

        first = queue.enqueue("task-stable")
        second = queue.enqueue("task-stable")

        self.assertEqual("task-stable", first)
        self.assertEqual(first, second)
        self.assertEqual(1, len(fake_rq.enqueue_calls))
        self.assertEqual("task-stable", fake_rq.enqueue_calls[0][2]["job_id"])

        fake_rq.jobs["task-stable"].status = FakeJobStatus.QUEUED
        self.assertEqual("queued", queue.get_job_status("task-stable"))
        self.assertTrue(queue.cancel("task-stable"))
        self.assertEqual("canceled", queue.get_job_status("task-stable"))

    def test_queued_cancel_stops_task_skips_job_and_creates_partial_manifest(self):
        queue = InMemoryTaskQueue()
        cancellation = InMemoryCancellationStore()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            service = self.create_service(
                queue=queue,
                repository=None,
                cancellation=cancellation,
                root=root,
            )
            task = service.create_task(
                "run-cancel", "project-queue", training_config().to_dict()
            )

            stopped = service.request_stop(task.task_id)
            executed = queue.execute_next(
                lambda task_id: self.fail("canceled queued job executed")
            )

            run_root = root / task.project_id / task.run_id
            manifest = json.loads((run_root / "manifest.json").read_text())
            self.assertEqual(TaskStatus.STOPPED, stopped.status)
            self.assertEqual("canceled", queue.get_job_status(task.task_id))
            self.assertIsNone(executed)
            self.assertTrue(cancellation.is_requested(task.task_id))
            self.assertEqual("stopped", manifest["final_status"])
            self.assertFalse((run_root / "checkpoints/model_latest.pt").exists())
            self.assertFalse((run_root / "checkpoints/model_best.pt").exists())

    def test_rq_entry_preflight_skips_a_cancelled_queued_task(self):
        queue = InMemoryTaskQueue()
        cancellation = InMemoryCancellationStore()
        service = self.create_service(queue=queue, cancellation=cancellation)
        task = service.create_task(
            "run-preflight-cancel",
            "project-queue",
            training_config().to_dict(),
        )
        cancellation.request(task.task_id)

        def forbidden_factory(_service):
            self.fail("cancelled queued task constructed a GPUWorker")

        result = execute_training_task(
            task.task_id,
            service,
            worker_factory=forbidden_factory,
            cancellation_store=cancellation,
        )

        self.assertEqual("stopped", result["status"])
        self.assertTrue(result["skipped"])
        self.assertEqual(TaskStatus.STOPPED, service.get_task(task.task_id).status)
        self.assertEqual("canceled", queue.get_job_status(task.task_id))

    def test_fifo_keeps_b_queued_while_a_runs_then_executes_b(self):
        queue = InMemoryTaskQueue()
        service = self.create_service(queue=queue)
        first = service.create_task(
            "run-a", "project-fifo", training_config().to_dict()
        )
        second = service.create_task(
            "run-b", "project-fifo", training_config().to_dict()
        )
        observed = []

        def handler(task_id):
            service.transition_task_status(task_id, TaskStatus.INITIALIZING)
            service.transition_task_status(task_id, TaskStatus.RUNNING)
            if task_id == first.task_id:
                observed.append(
                    (
                        service.get_task(second.task_id).status,
                        queue.get_job_status(second.task_id),
                    )
                )
            service.transition_task_status(task_id, TaskStatus.FINALIZING)
            service.transition_task_status(task_id, TaskStatus.COMPLETED)

        queue.execute_next(handler)
        self.assertEqual([(TaskStatus.QUEUED, "queued")], observed)
        self.assertEqual(TaskStatus.COMPLETED, service.get_task(first.task_id).status)
        self.assertEqual(TaskStatus.QUEUED, service.get_task(second.task_id).status)
        queue.execute_next(handler)
        self.assertEqual(TaskStatus.COMPLETED, service.get_task(second.task_id).status)

    def test_redis_cancel_flag_and_terminal_cancel_idempotency(self):
        client = FakeRedis()
        cancellation = RedisCancellationStore(client, namespace="cancel-test")
        queue = InMemoryTaskQueue()
        service = self.create_service(queue=queue, cancellation=cancellation)
        stopped = service.create_task(
            "run-stopped", "project-terminal", training_config().to_dict()
        )
        service.request_stop(stopped.task_id)

        self.assertTrue(cancellation.is_requested(stopped.task_id))
        self.assertEqual(TaskStatus.STOPPED, service.request_stop(stopped.task_id).status)
        cancellation.clear(stopped.task_id)
        self.assertFalse(cancellation.is_requested(stopped.task_id))

        completed = service.create_task(
            "run-completed", "project-terminal", training_config().to_dict()
        )
        for status in (
            TaskStatus.INITIALIZING,
            TaskStatus.RUNNING,
            TaskStatus.FINALIZING,
            TaskStatus.COMPLETED,
        ):
            service.transition_task_status(completed.task_id, status)
        self.assertEqual(TaskStatus.COMPLETED, service.request_stop(completed.task_id).status)

        failed = service.create_task(
            "run-failed", "project-terminal", training_config().to_dict()
        )
        service.transition_task_status(failed.task_id, TaskStatus.INITIALIZING)
        service.transition_task_status(failed.task_id, TaskStatus.FAILED)
        self.assertEqual(TaskStatus.FAILED, service.request_stop(failed.task_id).status)

    def test_rq_job_wrapper_marks_active_fake_worker_failure_failed(self):
        service = self.create_service(queue=InMemoryTaskQueue())
        task = service.create_task(
            "run-wrapper-fail", "project-wrapper", training_config().to_dict()
        )

        class FailingWorker:
            def __init__(self, current_service):
                self.service = current_service

            def run(self, task_id):
                self.service.transition_task_status(task_id, TaskStatus.INITIALIZING)
                self.service.transition_task_status(task_id, TaskStatus.RUNNING)
                raise RuntimeError("fake RQ wrapper failure")

        with self.assertRaisesRegex(RuntimeError, "wrapper failure"):
            execute_training_task(
                task.task_id,
                service,
                worker_factory=FailingWorker,
            )

        self.assertEqual(TaskStatus.FAILED, service.get_task(task.task_id).status)

    def test_rq_job_wrapper_factory_failure_does_not_leave_task_queued(self):
        service = self.create_service(queue=InMemoryTaskQueue())
        task = service.create_task(
            "run-wrapper-factory-fail",
            "project-wrapper",
            training_config().to_dict(),
        )

        def failing_factory(_service):
            raise RuntimeError("fake worker construction failure")

        with self.assertRaisesRegex(RuntimeError, "construction failure"):
            execute_training_task(
                task.task_id,
                service,
                worker_factory=failing_factory,
            )

        self.assertEqual(TaskStatus.FAILED, service.get_task(task.task_id).status)


if __name__ == "__main__":
    unittest.main()
