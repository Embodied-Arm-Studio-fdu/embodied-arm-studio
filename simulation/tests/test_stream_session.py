from dataclasses import replace
from datetime import datetime, timezone
import threading
import unittest
from unittest import mock

from simulation_core.domain import SimulationTask, TaskConfig, TaskStatus
from simulation_core.adapters import FakeBackendClient, FakeScheduler, InMemoryTaskRepository
from simulation_core.service import SimulationService
from simulation_core.training_task_runner import TrainingTaskResult
from simulation_core.stream_session import (
    FakeTaskLifecycleDriver, InvalidStreamTransition, RecordingStreamGateway, StreamDescriptor,
    StreamSessionManager, StreamSessionStatus, TaskStreamLifecycle,
)


class FakeRuntime:
    def __init__(self, fail=False):
        self.started = False; self.stops = 0; self.fail = fail; self.error_callback = None
    def start(self):
        if self.fail: raise RuntimeError("video unavailable")
        self.started = True
    def stop(self): self.stops += 1
    def set_error_callback(self, callback): self.error_callback = callback


def task(status):
    now = datetime.now(timezone.utc)
    return SimulationTask("task-1", "run-1", "project-1", status, now, now, TaskConfig())


class StreamSessionTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = []
        self.manager = StreamSessionManager(lambda _: self.runtimes.append(FakeRuntime()) or self.runtimes[-1])

    def test_fields_urls_serialization_and_safe_ids(self):
        item = self.manager.create_session("run-1", "task-1")
        self.assertEqual(item.status, StreamSessionStatus.CREATED)
        self.assertIn(item.stream_id, item.whep_url)
        self.assertEqual(item.to_dict()["camera_id"], "overview")
        for bad in ("../run", "run/x", "", "x?y"):
            with self.assertRaises(ValueError): self.manager.create_session(bad, "task-1")

    def test_state_machine_invalid_transition_and_stopped_terminal(self):
        item = self.manager.create_session("run-1", "task-1")
        with self.assertRaises(InvalidStreamTransition): item.transition_to(StreamSessionStatus.READY)
        item = self.manager.start_session(item.stream_id)
        self.assertEqual(item.status, StreamSessionStatus.READY)
        item = self.manager.stop_session(item.stream_id)
        self.assertEqual(item.status, StreamSessionStatus.STOPPED)
        with self.assertRaises(InvalidStreamTransition): item.transition_to(StreamSessionStatus.READY)

    def test_create_get_run_binding_stop_idempotent_and_concurrent(self):
        first = self.manager.create_session("run-1", "task-1")
        self.assertEqual(first, self.manager.create_session("run-1", "task-1"))
        self.manager.start_session(first.stream_id)
        threads = [threading.Thread(target=self.manager.stop_session, args=(first.stream_id,)) for _ in range(8)]
        [item.start() for item in threads]; [item.join() for item in threads]
        self.assertEqual(self.runtimes[0].stops, 1)
        self.assertEqual(len(self.manager.list_sessions_for_run("run-1")), 1)

    def test_client_disconnect_reconnect_does_not_stop_runtime(self):
        item = self.manager.create_session("run-1", "task-1")
        self.manager.start_session(item.stream_id)
        self.manager.mark_client_connected(item.stream_id)
        after = self.manager.mark_client_disconnected(item.stream_id)
        self.assertEqual(after.status, StreamSessionStatus.READY)
        self.assertEqual(self.runtimes[0].stops, 0)
        after = self.manager.mark_client_connected(item.stream_id)
        self.assertTrue(after.client_connected); self.assertEqual(after.reconnect_count, 1)

    def test_task_running_and_all_terminal_paths(self):
        for terminal in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED):
            runtimes = []
            manager = StreamSessionManager(lambda _: runtimes.append(FakeRuntime()) or runtimes[-1])
            lifecycle = TaskStreamLifecycle(manager)
            ready = lifecycle.on_task_state(task(TaskStatus.RUNNING))[0]
            self.assertEqual(ready.status, StreamSessionStatus.READY)
            stopped = lifecycle.on_task_state(task(terminal))[0]
            self.assertEqual(stopped.status, StreamSessionStatus.STOPPED)
            self.assertEqual(runtimes[0].stops, 1)

    def test_fake_task_lifecycle_three_scenarios(self):
        expected = {
            "complete": [TaskStatus.QUEUED, TaskStatus.INITIALIZING, TaskStatus.RUNNING,
                         TaskStatus.FINALIZING, TaskStatus.COMPLETED],
            "stop": [TaskStatus.QUEUED, TaskStatus.INITIALIZING, TaskStatus.RUNNING,
                     TaskStatus.STOPPING, TaskStatus.STOPPED],
            "fail": [TaskStatus.QUEUED, TaskStatus.INITIALIZING, TaskStatus.RUNNING,
                     TaskStatus.FAILED],
        }
        for method, history in expected.items():
            runtimes = []
            manager = StreamSessionManager(lambda _: runtimes.append(FakeRuntime()) or runtimes[-1])
            driver = FakeTaskLifecycleDriver(task(TaskStatus.QUEUED), TaskStreamLifecycle(manager))
            getattr(driver, method)()
            self.assertEqual(driver.history, history)
            self.assertEqual(manager.list_sessions_for_run("run-1")[0].status, StreamSessionStatus.STOPPED)

    def test_video_failure_is_isolated_from_task(self):
        runtimes = []
        manager = StreamSessionManager(
            lambda _: runtimes.append(FakeRuntime(fail=True)) or runtimes[-1]
        )
        training = task(TaskStatus.RUNNING)
        session = TaskStreamLifecycle(manager).on_task_state(training)[0]
        self.assertEqual(session.status, StreamSessionStatus.ERROR)
        self.assertEqual(training.status, TaskStatus.RUNNING)
        self.assertIn("video unavailable", session.video_error)

    def test_late_runtime_error_transitions_and_publishes_error(self):
        published = []
        runtimes = []
        manager = StreamSessionManager(
            lambda _: runtimes.append(FakeRuntime()) or runtimes[-1],
            rtsp_base="rtsp://127.0.0.1:18554",
            webrtc_base="http://192.0.2.10:18889",
            session_observer=published.append,
        )
        ready = TaskStreamLifecycle(manager).on_task_state(task(TaskStatus.RUNNING))[0]
        runtimes[0].error_callback(RuntimeError("publisher exited"))
        failed = manager.get_session(ready.stream_id)
        self.assertEqual(failed.status, StreamSessionStatus.ERROR)
        self.assertEqual(published[-1].status, StreamSessionStatus.ERROR)
        self.assertEqual(task(TaskStatus.RUNNING).status, TaskStatus.RUNNING)
        self.assertNotIn("127.0.0.1", failed.whep_url)
        self.assertNotIn("?", failed.whep_url)

    def test_service_optional_lifecycle_observes_committed_states(self):
        lifecycle = mock.Mock()
        repository = InMemoryTaskRepository()
        service = SimulationService(repository, FakeScheduler(), FakeBackendClient(),
                                    task_lifecycle=lifecycle)
        created = service.create_task("run-service", "project", {})
        service.transition_task_status(created.task_id, TaskStatus.INITIALIZING)
        service.transition_task_status(created.task_id, TaskStatus.RUNNING)
        observed = [call.args[0].status for call in lifecycle.on_task_state.call_args_list]
        self.assertEqual(observed, [TaskStatus.INITIALIZING, TaskStatus.RUNNING])

    def test_service_lifecycle_failure_does_not_fail_task(self):
        lifecycle = mock.Mock()
        lifecycle.on_task_state.side_effect = RuntimeError("video failed")
        service = SimulationService(InMemoryTaskRepository(), FakeScheduler(),
                                    FakeBackendClient(), task_lifecycle=lifecycle)
        created = service.create_task("run-isolated", "project", {})
        updated = service.transition_task_status(created.task_id, TaskStatus.INITIALIZING)
        self.assertEqual(updated.status, TaskStatus.INITIALIZING)

    def test_video_error_can_still_complete_and_publish_callback(self):
        backend = FakeBackendClient()
        manager = StreamSessionManager(lambda _: FakeRuntime(fail=True))
        service = SimulationService(InMemoryTaskRepository(), FakeScheduler(), backend,
            task_lifecycle=TaskStreamLifecycle(manager))
        created = service.create_task("run-video-error", "project", {})
        service.transition_task_status(created.task_id, TaskStatus.INITIALIZING)
        running = service.transition_task_status(created.task_id, TaskStatus.RUNNING)
        self.assertEqual(manager.list_sessions_for_run(running.run_id)[0].status,
                         StreamSessionStatus.ERROR)
        service.transition_task_status(created.task_id, TaskStatus.FINALIZING)
        completed = service.transition_task_status(created.task_id, TaskStatus.COMPLETED)
        result = TrainingTaskResult(completed.task_id, completed.run_id,
            TaskStatus.COMPLETED, 1, 1, "metrics/training.jsonl",
            "checkpoints/model_best.pt", {"average_reward": 0.0},
            manifest_path="manifest.json", artifacts=({
                "type": "checkpoint_best", "path": "checkpoints/model_best.pt"
            }, {
                "type": "manifest", "path": "manifest.json",
                "size_bytes": 123, "sha256": "a" * 64,
            }))
        service.enqueue_terminal_callback(completed.task_id, result)
        self.assertEqual(completed.status, TaskStatus.COMPLETED)
        self.assertEqual(backend.terminal_callbacks[0].status.value, "completed")
        self.assertEqual(manager.list_sessions_for_run(running.run_id)[0].status,
                         StreamSessionStatus.STOPPED)

    def test_mock_contracts(self):
        session = self.manager.create_session("run-1", "task-1")
        session = self.manager.start_session(session.stream_id)
        gateway = RecordingStreamGateway()
        record = gateway.publish(session)
        descriptor = StreamDescriptor.from_session(session)
        self.assertEqual(record["status"], "READY")
        self.assertEqual(descriptor.whep_url, session.whep_url)


if __name__ == "__main__": unittest.main()
