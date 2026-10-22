import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from simulation_core import (
    FakeBackendClient,
    FakeScheduler,
    GPUWorker,
    InMemoryTaskRepository,
    RedisRunSequenceGenerator,
    SimulationService,
    TaskStatus,
    TrainingProcessError,
    TrainingTaskConfig,
)


def config():
    return TrainingTaskConfig(
        task_type="training", environment_type="s1_push", seed=5,
        rollout_steps=4, num_updates=1, checkpoint_every=1,
        episode_steps=8, physics_steps_per_action=1, batch_size=4,
    )


def result_payload(task, status="completed"):
    return {
        "task_id": task["task_id"], "run_id": task["run_id"], "status": status,
        "training_steps": 4, "update_steps": 1,
        "metrics_path": "/fake/metrics.jsonl",
        "checkpoint_path": "/fake/model_latest.pt",
        "final_metrics_summary": {"average_reward": 0.5},
    }


def metric_child(task, artifacts_root, connection, cancel_event, shared_cancel_config):
    connection.send({"type": "status", "status": "running"})
    connection.send({"type": "metric", "record": {
        "update_index": 1, "training_steps": 4, "average_reward": 0.25,
        "episode_length": 4.0, "episodes": 1.0, "policy_loss": 0.12,
        "value_loss": 0.08, "entropy": 7.4, "approx_kl": 0.011,
        "clip_fraction": 0.15, "explained_variance": 0.72,
        "learning_rate": 0.0003,
    }})
    connection.send({"type": "status", "status": "finalizing"})
    connection.send({"type": "result", "result": result_payload(task)})
    connection.close()


def failure_child(task, artifacts_root, connection, cancel_event, shared_cancel_config):
    connection.send({"type": "status", "status": "running"})
    connection.send({"type": "error", "error": {
        "type": "RuntimeError", "message": "concise child failure",
        "traceback": "full traceback remains child-side diagnostic",
    }})
    connection.close()


def heartbeat_child(task, artifacts_root, connection, cancel_event, shared_cancel_config):
    connection.send({"type": "status", "status": "running"})
    time.sleep(0.08)
    connection.send({"type": "status", "status": "finalizing"})
    connection.send({"type": "result", "result": result_payload(task)})
    connection.close()


def cooperative_child(task, artifacts_root, connection, cancel_event, shared_cancel_config):
    connection.send({"type": "status", "status": "running"})
    if not cancel_event.wait(2.0):
        os._exit(9)
    connection.send({"type": "result", "result": result_payload(task, "stopped")})
    connection.close()


class AtomicFakeRedis:
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def incr(self, key):
        with self.lock:
            value = int(self.values.get(key, 0)) + 1
            self.values[key] = value
            return value


class RuntimeEventTestCase(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryTaskRepository()
        self.backend = FakeBackendClient()
        self.service = SimulationService(
            self.repository, FakeScheduler(), self.backend,
            task_id_factory=lambda: "task-events",
        )
        self.task = self.service.create_task(
            "run-events", "project-events", config().to_dict()
        )
        self.context = multiprocessing.get_context("fork")
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def worker(self, target, heartbeat_interval=10.0):
        return GPUWorker(
            self.service, Path(self.temp.name),
            multiprocessing_context=self.context, process_target=target,
            heartbeat_interval=heartbeat_interval,
        )

    def test_successful_transitions_publish_once_and_illegal_transition_does_not(self):
        self.service.transition_task_status(self.task.task_id, TaskStatus.INITIALIZING)
        self.service.transition_task_status(self.task.task_id, TaskStatus.RUNNING)
        before = len(self.backend.events)
        with self.assertRaises(Exception):
            self.service.transition_task_status(self.task.task_id, TaskStatus.QUEUED)
        states = [event for event in self.backend.events if event.type.value == "state"]
        self.assertEqual(2, len(states))
        self.assertEqual(before, len(self.backend.events))
        self.assertEqual(
            {"previous_status": "initializing", "status": "running"},
            states[-1].payload,
        )

    def test_child_metric_message_becomes_parent_backend_event_in_order(self):
        result = self.worker(metric_child).run(self.task.task_id)
        events = self.backend.events_for_run(self.task.run_id)
        self.assertEqual(TaskStatus.COMPLETED, result.status)
        self.assertEqual([1, 2, 3, 4], [event.sequence for event in events])
        self.assertEqual(["state", "state", "metric", "state"], [event.type.value for event in events])
        metric = events[2]
        self.assertEqual(4, metric.payload["step"])
        self.assertEqual({
            "update_index": 1, "average_reward": 0.25,
            "episode_length": 4.0, "episodes": 1.0,
            "policy_loss": 0.12, "value_loss": 0.08, "entropy": 7.4,
            "approx_kl": 0.011, "clip_fraction": 0.15,
            "explained_variance": 0.72,
        }, metric.payload["metrics"])
        self.assertNotIn("learning_rate", metric.payload["metrics"])
        self.assertNotIn("update_index", metric.payload)

    def test_child_failure_publishes_concise_error_before_failed(self):
        with self.assertRaises(TrainingProcessError):
            self.worker(failure_child).run(self.task.task_id)
        errors = [event for event in self.backend.events if event.type.value == "error"]
        self.assertEqual(1, len(errors))
        self.assertEqual("TRAINING_FAILED", errors[0].payload["code"])
        self.assertNotIn("traceback", errors[0].payload)
        self.assertEqual(TaskStatus.FAILED, self.service.get_task(self.task.task_id).status)

    def test_heartbeat_only_active_and_monitor_stops_emitting_after_exit(self):
        self.service.publish_heartbeat(self.task.task_id, 0)
        self.assertEqual([], self.backend.events)
        worker = self.worker(heartbeat_child, heartbeat_interval=0.01)
        worker.run(self.task.task_id)
        heartbeats = [event for event in self.backend.events if event.type.value == "heartbeat"]
        self.assertTrue(heartbeats)
        count = len(self.backend.events)
        self.service.publish_heartbeat(self.task.task_id, 4)
        time.sleep(0.03)
        self.assertEqual(count, len(self.backend.events))

    def test_running_stop_publishes_stopping_state(self):
        worker = self.worker(cooperative_child)
        errors = []
        thread = threading.Thread(
            target=lambda: self._capture_run(worker, errors), daemon=True
        )
        thread.start()
        deadline = time.monotonic() + 2
        while self.service.get_task(self.task.task_id).status != TaskStatus.RUNNING:
            if time.monotonic() > deadline:
                self.fail("worker did not reach running")
            time.sleep(0.005)
        worker.stop(self.task.task_id)
        thread.join(2)
        self.assertEqual([], errors)
        payloads = [event.payload for event in self.backend.events if event.type.value == "state"]
        self.assertIn({"previous_status": "running", "status": "stopping"}, payloads)

    def _capture_run(self, worker, errors):
        try:
            worker.run(self.task.task_id)
        except Exception as exc:
            errors.append(exc)


class RedisSequenceTestCase(unittest.TestCase):
    def test_atomic_increment_and_run_isolation(self):
        generator = RedisRunSequenceGenerator(AtomicFakeRedis(), "events-test")
        values = []
        lock = threading.Lock()

        def allocate():
            value = generator.next("run-a")
            with lock:
                values.append(value)

        threads = [threading.Thread(target=allocate) for _ in range(40)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(list(range(1, 41)), sorted(values))
        self.assertEqual(1, generator.next("run-b"))
        self.assertEqual(41, generator.next("run-a"))


if __name__ == "__main__":
    unittest.main()
