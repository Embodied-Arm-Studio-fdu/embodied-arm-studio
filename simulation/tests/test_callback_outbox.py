from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from simulation_core import (
    ArtifactFinalizer, BackendMessageFactory, CallbackDispatcher, DeliveryStatus,
    FailingBackendClient, FakeBackendClient, FakeScheduler,
    InMemoryCallbackOutbox, InMemoryTaskRepository, RedisCallbackOutbox,
    SimulationService, TaskStatus, TrainingTaskResult,
    TrainingTaskConfig, TrainingTaskSpec,
)


STARTED = "2026-08-23T10:00:00+00:00"
ENDED = "2026-08-23T10:01:00+00:00"


class ManualClock:
    def __init__(self):
        self.now = datetime(2026, 8, 23, 10, 2, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def result(status, error=None, partial=False):
    artifacts = [
        {"type": "manifest", "path": "manifest.json", "size_bytes": 123,
         "sha256": "a" * 64},
        {"type": "training_config", "path": "config/training_config.json"},
        {"type": "task_log", "path": "logs/task.log"},
        {"type": "training_metrics", "path": "metrics/training.jsonl"},
    ]
    if not partial:
        artifacts.extend([
            {"type": "checkpoint_best", "path": "checkpoints/model_best.pt"},
            {"type": "checkpoint_latest", "path": "checkpoints/model_latest.pt"},
        ])
    return TrainingTaskResult(
        task_id="task-terminal", run_id="run-terminal", status=status,
        training_steps=32, update_steps=2,
        metrics_path="/host/run/metrics/training.jsonl",
        checkpoint_path="/host/run/checkpoints/model_latest.pt",
        final_metrics_summary={
            "training_steps": 32,
            "average_reward": 0.75,
            "samples_consumed_per_epoch": [16, 16],
        },
        manifest_path="/host/run/manifest.json",
        best_checkpoint_path="/host/run/checkpoints/model_best.pt" if not partial else "",
        error=error, artifacts=tuple(artifacts),
        started_at=STARTED, ended_at=ENDED,
        evaluation={"status": "passed", "policy_passed": True},
    )


def make_service(backend=None, outbox=None, dispatcher=None):
    repository = InMemoryTaskRepository()
    backend = backend or FakeBackendClient()
    outbox = outbox or InMemoryCallbackOutbox()
    service = SimulationService(
        repository, FakeScheduler(), backend,
        task_id_factory=lambda: "task-terminal",
        callback_outbox=outbox, callback_dispatcher=dispatcher,
    )
    task = service.create_task("run-terminal", "project-terminal", {})
    return service, backend, outbox, task


class TerminalWiringTests(unittest.TestCase):
    def test_terminal_callback_only_contains_numeric_metric_summary(self):
        service, _, _, task = make_service()
        for status in (TaskStatus.INITIALIZING, TaskStatus.RUNNING,
                       TaskStatus.FINALIZING, TaskStatus.COMPLETED):
            service.transition_task_status(task.task_id, status)

        callback = service.enqueue_terminal_callback(
            task.task_id, result(TaskStatus.COMPLETED)
        ).callback

        self.assertEqual(32, callback.metrics["training_steps"])
        self.assertEqual(0.75, callback.metrics["average_reward"])
        self.assertNotIn("samples_consumed_per_epoch", callback.metrics)
        self.assertEqual(True, callback.evaluation["policy_passed"])

    def test_final_manifest_metadata_reaches_callback_unchanged(self):
        service, _, _, task = make_service()
        for status in (TaskStatus.INITIALIZING, TaskStatus.RUNNING,
                       TaskStatus.STOPPING, TaskStatus.STOPPED):
            service.transition_task_status(task.task_id, status)
        with tempfile.TemporaryDirectory() as directory:
            finalizer = ArtifactFinalizer(Path(directory))
            finalizer.prepare(TrainingTaskSpec(
                task_id=task.task_id, run_id=task.run_id,
                project_id=task.project_id,
                config=TrainingTaskConfig(
                    task_type="training", environment_type="s1_push", seed=1,
                    rollout_steps=4, num_updates=1, checkpoint_every=1,
                    episode_steps=8, physics_steps_per_action=1, batch_size=4,
                ),
                created_at=task.created_at.isoformat(),
            ))
            summary = finalizer.finalize(TaskStatus.STOPPED)
            base = TrainingTaskResult(
                task_id=task.task_id, run_id=task.run_id,
                status=TaskStatus.STOPPED, training_steps=0, update_steps=0,
                metrics_path="", checkpoint_path="", final_metrics_summary={},
            )
            callback = service.enqueue_terminal_callback(
                task.task_id, base.with_artifacts(summary),
                stop_reason="requested_by_user",
            ).callback
            item = next(x for x in callback.artifacts if x["type"] == "manifest")
            manifest_path = Path(summary.manifest_path)
            self.assertEqual(manifest_path.stat().st_size, item["size_bytes"])
            self.assertEqual(
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(), item["sha256"]
            )

    def test_completed_callback_uses_best_and_relative_artifacts(self):
        service, backend, outbox, task = make_service()
        for status in (TaskStatus.INITIALIZING, TaskStatus.RUNNING, TaskStatus.FINALIZING, TaskStatus.COMPLETED):
            service.transition_task_status(task.task_id, status)
        record = service.enqueue_terminal_callback(task.task_id, result(TaskStatus.COMPLETED))
        callback = record.callback
        self.assertEqual(DeliveryStatus.DELIVERED, record.delivery_status)
        self.assertEqual("completed", callback.status.value)
        self.assertEqual("checkpoint_best", callback.model["type"])
        self.assertEqual("checkpoints/model_best.pt", callback.model["path"])
        self.assertIn("manifest.json", [item["path"] for item in callback.artifacts])
        manifest = next(
            item for item in callback.artifacts if item["path"] == "manifest.json"
        )
        self.assertEqual(123, manifest["size_bytes"])
        self.assertEqual("a" * 64, manifest["sha256"])
        self.assertTrue(all(not item["path"].startswith("/") for item in callback.artifacts))
        self.assertIsNone(callback.error)
        self.assertEqual(backend.terminal_callbacks[0], callback)

    def test_failed_callback_has_structured_error_and_partial_artifacts(self):
        service, backend, _, task = make_service()
        service.transition_task_status(task.task_id, TaskStatus.INITIALIZING)
        service.publish_error(task.task_id, "INITIALIZATION_FAILED", "init failed")
        service.transition_task_status(task.task_id, TaskStatus.FAILED)
        callback = service.enqueue_terminal_callback(
            task.task_id,
            result(TaskStatus.FAILED, {"message": "init failed"}, partial=True),
            error_code="INITIALIZATION_FAILED",
        ).callback
        self.assertEqual({
            "code": "INITIALIZATION_FAILED", "message": "init failed",
            "retryable": False,
        }, callback.error)
        self.assertIsNone(callback.model)
        self.assertEqual("error", backend.events[-1].type.value)

    def test_stopped_callback_reason_and_idempotent_creation(self):
        service, backend, outbox, task = make_service()
        service.transition_task_status(task.task_id, TaskStatus.INITIALIZING)
        service.transition_task_status(task.task_id, TaskStatus.RUNNING)
        service.transition_task_status(task.task_id, TaskStatus.STOPPING)
        service.transition_task_status(task.task_id, TaskStatus.STOPPED)
        first = service.enqueue_terminal_callback(
            task.task_id, result(TaskStatus.STOPPED),
            stop_reason="requested_by_user",
        )
        second = service.enqueue_terminal_callback(
            task.task_id, result(TaskStatus.STOPPED), stop_reason="different",
        )
        self.assertEqual("requested_by_user", first.callback.stop_reason)
        self.assertEqual(first.callback.to_dict(), second.callback.to_dict())
        self.assertEqual(1, len(backend.terminal_callbacks))
        self.assertEqual(
            1,
            sum(item["type"] == "manifest" for item in first.callback.artifacts),
        )

    def test_events_and_callback_share_factory_sequence(self):
        service, _, _, task = make_service()
        service.transition_task_status(task.task_id, TaskStatus.INITIALIZING)
        service.transition_task_status(task.task_id, TaskStatus.RUNNING)
        service.transition_task_status(task.task_id, TaskStatus.FINALIZING)
        service.transition_task_status(task.task_id, TaskStatus.COMPLETED)
        terminal = service.enqueue_terminal_callback(
            task.task_id, result(TaskStatus.COMPLETED)
        ).callback
        self.assertEqual(4, terminal.sequence)


class DispatcherTests(unittest.TestCase):
    def test_exhausted_callback_can_only_be_confirmed_after_controlled_recovery(self):
        clock = ManualClock()
        outbox = InMemoryCallbackOutbox()
        backend = FailingBackendClient(failures_before_success=99)
        dispatcher = CallbackDispatcher(outbox, backend, 1, 0, clock)
        service, _, _, task = make_service(backend, outbox, dispatcher)
        for status in (TaskStatus.INITIALIZING, TaskStatus.RUNNING, TaskStatus.FINALIZING, TaskStatus.COMPLETED):
            service.transition_task_status(task.task_id, status)
        exhausted = service.enqueue_terminal_callback(task.task_id, result(TaskStatus.COMPLETED))
        self.assertEqual(DeliveryStatus.EXHAUSTED, exhausted.delivery_status)
        recovered = dispatcher.confirm_controlled_recovery(
            exhausted.event_id, exhausted.callback
        )
        self.assertEqual(DeliveryStatus.DELIVERED, recovered.delivery_status)
        self.assertEqual(exhausted.attempt_count, recovered.attempt_count)

    def test_outbox_is_written_before_first_send_and_retry_reuses_callback(self):
        clock = ManualClock()
        outbox = InMemoryCallbackOutbox()
        observed = []
        backend = FailingBackendClient(
            failures_before_success=1,
            before_terminal_publish=lambda callback: observed.append(
                outbox.get(callback.event_id) is not None
            ),
        )
        dispatcher = CallbackDispatcher(outbox, backend, 3, 1.0, clock)
        service, _, _, task = make_service(backend, outbox, dispatcher)
        for status in (TaskStatus.INITIALIZING, TaskStatus.RUNNING, TaskStatus.FINALIZING, TaskStatus.COMPLETED):
            service.transition_task_status(task.task_id, status)
        first = service.enqueue_terminal_callback(task.task_id, result(TaskStatus.COMPLETED))
        self.assertEqual([True], observed)
        self.assertEqual(DeliveryStatus.PENDING, first.delivery_status)
        self.assertEqual(1, first.attempt_count)
        self.assertEqual(clock.now + timedelta(seconds=1), first.next_attempt_at)
        original = first.callback.to_dict()
        clock.advance(1)
        dispatcher.dispatch_once()
        delivered = outbox.get(first.event_id)
        self.assertEqual(DeliveryStatus.DELIVERED, delivered.delivery_status)
        self.assertEqual(2, delivered.attempt_count)
        self.assertEqual(original, delivered.callback.to_dict())
        self.assertEqual(
            [first.event_id, first.event_id],
            [item.event_id for item in backend.terminal_attempts],
        )
        self.assertEqual(TaskStatus.COMPLETED, service.get_task(task.task_id).status)

    def test_exponential_backoff_and_exhaustion_do_not_change_terminal_state(self):
        clock = ManualClock()
        outbox = InMemoryCallbackOutbox()
        backend = FailingBackendClient(failures_before_success=99)
        dispatcher = CallbackDispatcher(outbox, backend, 3, 2.0, clock)
        service, _, _, task = make_service(backend, outbox, dispatcher)
        for status in (TaskStatus.INITIALIZING, TaskStatus.RUNNING, TaskStatus.FINALIZING, TaskStatus.COMPLETED):
            service.transition_task_status(task.task_id, status)
        first = service.enqueue_terminal_callback(task.task_id, result(TaskStatus.COMPLETED))
        self.assertEqual(clock.now + timedelta(seconds=2), first.next_attempt_at)
        clock.advance(2)
        second = dispatcher.dispatch_event(first.event_id)
        self.assertEqual(2, second.attempt_count)
        self.assertEqual(clock.now + timedelta(seconds=4), second.next_attempt_at)
        clock.advance(4)
        third = dispatcher.dispatch_event(first.event_id)
        self.assertEqual(DeliveryStatus.EXHAUSTED, third.delivery_status)
        self.assertEqual(3, third.attempt_count)
        self.assertIsNone(third.next_attempt_at)
        self.assertEqual(TaskStatus.COMPLETED, service.get_task(task.task_id).status)


class RedisFake:
    def __init__(self):
        self.values = {}
        self.sets = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value.encode() if isinstance(value, str) else value

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value.encode() if isinstance(value, str) else value)

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


class RedisOutboxTests(unittest.TestCase):
    def test_json_persistence_and_task_idempotency(self):
        redis = RedisFake()
        outbox = RedisCallbackOutbox(redis, "outbox-test")
        factory = BackendMessageFactory()
        callback = factory.terminal_callback(
            "run-terminal", "task-terminal", status="stopped",
            started_at=STARTED, ended_at=ENDED, training_steps=1,
            update_steps=0, stop_reason="requested_by_user", metrics={},
            model=None, artifacts=[], error=None,
        )
        first = outbox.put_if_absent(callback)
        second = outbox.put_if_absent(callback)
        self.assertEqual(first.to_dict(), second.to_dict())
        raw = redis.get("outbox-test:callback_outbox:event:%s" % callback.event_id)
        decoded = json.loads(raw.decode())
        self.assertEqual(callback.event_id, decoded["event_id"])
        self.assertEqual("pending", decoded["delivery_status"])
        self.assertEqual([callback.event_id], [item.event_id for item in outbox.due(first.created_at)])


if __name__ == "__main__":
    unittest.main()
