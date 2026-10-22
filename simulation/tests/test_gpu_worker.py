import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from simulation_core import (
    ActiveTaskMismatchError,
    FakeBackendClient,
    FakeScheduler,
    GPUWorker,
    InMemoryTaskRepository,
    SimulationService,
    TaskStatus,
    TrainingProcessError,
    TrainingTaskConfig,
    WorkerBusyError,
)
from simulation_core.gpu_worker import classify_process_exit


def _result_payload(task):
    return {
        "task_id": task["task_id"],
        "run_id": task["run_id"],
        "status": "completed",
        "training_steps": 16,
        "update_steps": 1,
        "metrics_path": "/fake/metrics.jsonl",
        "checkpoint_path": "/fake/model_latest.pt",
        "final_metrics_summary": {"average_reward": 0.5},
    }


def fake_success_child(
    task, artifacts_root, connection, cancel_event, shared_cancel_config
):
    connection.send({"type": "status", "status": "running"})
    connection.send({"type": "status", "status": "finalizing"})
    connection.send({"type": "result", "result": _result_payload(task)})
    connection.close()


def fake_failure_child(
    task, artifacts_root, connection, cancel_event, shared_cancel_config
):
    connection.send({"type": "status", "status": "running"})
    connection.send(
        {
            "type": "error",
            "error": {"type": "RuntimeError", "message": "fake training failed"},
        }
    )
    connection.close()


def fake_initialization_failure_child(
    task, artifacts_root, connection, cancel_event, shared_cancel_config
):
    connection.send(
        {
            "type": "error",
            "error": {"type": "RuntimeError", "message": "fake initialization failed"},
        }
    )
    connection.close()


def fake_abnormal_exit_child(
    task, artifacts_root, connection, cancel_event, shared_cancel_config
):
    connection.close()
    os._exit(7)


def fake_slow_success_child(
    task, artifacts_root, connection, cancel_event, shared_cancel_config
):
    connection.send({"type": "status", "status": "running"})
    time.sleep(0.3)
    connection.send({"type": "status", "status": "finalizing"})
    connection.send({"type": "result", "result": _result_payload(task)})
    connection.close()


def fake_cooperative_child(
    task, artifacts_root, connection, cancel_event, shared_cancel_config
):
    connection.send({"type": "status", "status": "running"})
    if not cancel_event.wait(2.0):
        raise RuntimeError("fake child did not receive cancellation")
    result = _result_payload(task)
    result.update(
        status="stopped",
        training_steps=8,
        update_steps=0,
        final_metrics_summary={},
    )
    connection.send({"type": "result", "result": result})
    connection.close()


def fake_delayed_cooperative_child(
    task, artifacts_root, connection, cancel_event, shared_cancel_config
):
    connection.send({"type": "status", "status": "running"})
    if not cancel_event.wait(2.0):
        raise RuntimeError("fake child did not receive cancellation")
    time.sleep(0.15)
    result = _result_payload(task)
    result["status"] = "stopped"
    connection.send({"type": "result", "result": result})
    connection.close()


def fake_unresponsive_child(
    task, artifacts_root, connection, cancel_event, shared_cancel_config
):
    connection.send({"type": "status", "status": "running"})
    time.sleep(10.0)


def fake_shared_cancel_child(
    task, artifacts_root, connection, cancel_event, shared_cancel_config
):
    connection.send({"type": "status", "status": "running"})
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if shared_cancel_config["flag"].value:
            result = _result_payload(task)
            result["status"] = "stopped"
            connection.send({"type": "result", "result": result})
            connection.close()
            return
        time.sleep(0.01)
    raise RuntimeError("fake child did not observe shared cancellation flag")


def training_config():
    return TrainingTaskConfig(
        task_type="training",
        environment_type="s1_push",
        seed=7,
        rollout_steps=16,
        num_updates=1,
        checkpoint_every=1,
        episode_steps=8,
        physics_steps_per_action=1,
        batch_size=16,
    )


class GPUWorkerTestCase(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryTaskRepository()
        self.backend = FakeBackendClient()
        self.ids = iter(("task-1", "task-2"))
        self.service = SimulationService(
            self.repository,
            FakeScheduler(),
            self.backend,
            task_id_factory=lambda: next(self.ids),
        )
        self.context = multiprocessing.get_context("fork")
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.artifacts_root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def create_task(self, run_id="run-1"):
        return self.service.create_task(
            run_id, "project-worker", training_config().to_dict()
        )

    def worker(self, target, stop_timeout=10.0):
        return GPUWorker(
            self.service,
            self.artifacts_root,
            multiprocessing_context=self.context,
            process_target=target,
            stop_timeout=stop_timeout,
        )

    def run_in_thread(self, worker, task_id):
        outcome = {"results": [], "errors": []}

        def target():
            try:
                outcome["results"].append(worker.run(task_id))
            except Exception as exc:
                outcome["errors"].append(exc)

        thread = threading.Thread(target=target)
        thread.start()
        return thread, outcome

    def wait_for_status(self, task_id, status, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.service.get_task(task_id).status == status:
                return
            time.sleep(0.01)
        self.fail("task did not reach status %s" % status.value)

    def test_idle_worker_accepts_task_and_returns_to_idle_after_success(self):
        task = self.create_task()
        worker = self.worker(fake_success_child)

        result = worker.run(task.task_id)

        self.assertEqual(TaskStatus.COMPLETED, result.status)
        self.assertEqual(TaskStatus.COMPLETED, self.service.get_task(task.task_id).status)
        self.assertEqual(0, worker.last_exit_code)
        self.assertIsNotNone(worker.last_child_pid)
        self.assertFalse(worker.is_busy)
        self.assertIsNone(worker.active_child_pid)

    def test_busy_worker_rejects_second_task_without_queueing_it(self):
        first = self.create_task("run-1")
        second = self.create_task("run-2")
        worker = self.worker(fake_slow_success_child)
        thread_error = []

        def run_first():
            try:
                worker.run(first.task_id)
            except Exception as exc:
                thread_error.append(exc)

        thread = threading.Thread(target=run_first)
        thread.start()
        deadline = time.monotonic() + 2.0
        while not worker.is_busy and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(worker.is_busy)
        with self.assertRaises(WorkerBusyError):
            worker.run(second.task_id)
        thread.join(2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual([], thread_error)
        self.assertEqual(TaskStatus.COMPLETED, self.service.get_task(first.task_id).status)
        self.assertEqual(TaskStatus.QUEUED, self.service.get_task(second.task_id).status)
        self.assertFalse(worker.is_busy)

    def test_fake_child_failure_sets_failed_and_worker_returns_to_idle(self):
        task = self.create_task()
        worker = self.worker(fake_failure_child)

        with self.assertRaisesRegex(TrainingProcessError, "fake training failed"):
            worker.run(task.task_id)

        self.assertEqual(TaskStatus.FAILED, self.service.get_task(task.task_id).status)
        self.assertEqual(0, worker.last_exit_code)
        self.assertFalse(worker.is_busy)
        self.assertEqual(TaskStatus.FAILED, worker.last_result.status)
        self.assertTrue(Path(worker.last_result.manifest_path).is_file())

    def test_initialization_failure_sets_failed_without_entering_running(self):
        task = self.create_task()
        worker = self.worker(fake_initialization_failure_child)

        with self.assertRaisesRegex(TrainingProcessError, "initialization failed"):
            worker.run(task.task_id)

        self.assertEqual(TaskStatus.FAILED, self.service.get_task(task.task_id).status)
        self.assertEqual(
            [TaskStatus.QUEUED, TaskStatus.INITIALIZING, TaskStatus.FAILED],
            [item.status for item in self.backend.reported_tasks],
        )
        self.assertFalse(worker.is_busy)

    def test_abnormal_child_exit_is_detected_and_task_does_not_stick(self):
        task = self.create_task()
        worker = self.worker(fake_abnormal_exit_child)

        with self.assertRaisesRegex(TrainingProcessError, "exited with code 7"):
            worker.run(task.task_id)

        self.assertEqual(TaskStatus.FAILED, self.service.get_task(task.task_id).status)
        self.assertEqual(7, worker.last_exit_code)
        self.assertFalse(worker.is_busy)
        self.assertIsNone(worker.active_child_pid)

    def test_sigsegv_exit_is_classified_as_non_retryable_native_signal(self):
        classified = classify_process_exit(-11)
        self.assertEqual("NativeProcessSignal", classified["type"])
        self.assertEqual("SIGSEGV", classified["signal"])
        self.assertEqual(11, classified["signal_number"])
        self.assertEqual(-11, classified["exit_code"])
        self.assertIs(classified["retryable"], False)

    def test_parent_repository_observes_full_success_state_chain(self):
        task = self.create_task()
        worker = self.worker(fake_success_child)

        worker.run(task.task_id)

        self.assertEqual(
            [
                TaskStatus.QUEUED,
                TaskStatus.INITIALIZING,
                TaskStatus.RUNNING,
                TaskStatus.FINALIZING,
                TaskStatus.COMPLETED,
            ],
            [item.status for item in self.backend.reported_tasks],
        )

    def test_running_task_cooperatively_stops_and_worker_returns_idle(self):
        task = self.create_task()
        worker = self.worker(fake_cooperative_child)
        thread, outcome = self.run_in_thread(worker, task.task_id)
        self.wait_for_status(task.task_id, TaskStatus.RUNNING)

        stopped = worker.stop(task.task_id)
        thread.join(2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual([], outcome["errors"])
        self.assertEqual(TaskStatus.STOPPED, stopped.status)
        self.assertEqual(TaskStatus.STOPPED, outcome["results"][0].status)
        self.assertEqual(8, outcome["results"][0].training_steps)
        self.assertEqual(
            [
                TaskStatus.QUEUED,
                TaskStatus.INITIALIZING,
                TaskStatus.RUNNING,
                TaskStatus.STOPPING,
                TaskStatus.STOPPED,
            ],
            [item.status for item in self.backend.reported_tasks],
        )
        self.assertFalse(worker.last_stop_forced)
        self.assertFalse(worker.is_busy)
        self.assertIsNone(worker.active_child_pid)
        self.assertIsNone(worker.active_task_id)

    def test_repeated_stop_while_stopping_is_idempotent(self):
        task = self.create_task()
        worker = self.worker(fake_delayed_cooperative_child)
        run_thread, outcome = self.run_in_thread(worker, task.task_id)
        self.wait_for_status(task.task_id, TaskStatus.RUNNING)
        first_results = []
        first_stop = threading.Thread(
            target=lambda: first_results.append(worker.stop(task.task_id))
        )
        first_stop.start()
        self.wait_for_status(task.task_id, TaskStatus.STOPPING)

        second = worker.stop(task.task_id)
        first_stop.join(2.0)
        run_thread.join(2.0)

        self.assertEqual(TaskStatus.STOPPED, second.status)
        self.assertEqual(TaskStatus.STOPPED, first_results[0].status)
        self.assertEqual([], outcome["errors"])
        self.assertEqual(1, [
            item.status for item in self.backend.reported_tasks
        ].count(TaskStatus.STOPPING))

    def test_stop_after_stopped_is_idempotent(self):
        task = self.create_task()
        worker = self.worker(fake_cooperative_child)
        thread, outcome = self.run_in_thread(worker, task.task_id)
        self.wait_for_status(task.task_id, TaskStatus.RUNNING)
        worker.stop(task.task_id)
        thread.join(2.0)

        stopped_again = worker.stop(task.task_id)

        self.assertEqual(TaskStatus.STOPPED, stopped_again.status)
        self.assertEqual([], outcome["errors"])

    def test_stop_does_not_rewrite_completed_task(self):
        task = self.create_task()
        worker = self.worker(fake_success_child)
        worker.run(task.task_id)

        unchanged = worker.stop(task.task_id)

        self.assertEqual(TaskStatus.COMPLETED, unchanged.status)

    def test_stop_does_not_rewrite_failed_task(self):
        task = self.create_task()
        worker = self.worker(fake_failure_child)
        with self.assertRaises(TrainingProcessError):
            worker.run(task.task_id)

        unchanged = worker.stop(task.task_id)

        self.assertEqual(TaskStatus.FAILED, unchanged.status)

    def test_stop_rejects_task_id_that_does_not_match_active_task(self):
        first = self.create_task("run-1")
        second = self.create_task("run-2")
        worker = self.worker(fake_cooperative_child)
        thread, outcome = self.run_in_thread(worker, first.task_id)
        self.wait_for_status(first.task_id, TaskStatus.RUNNING)

        with self.assertRaisesRegex(ActiveTaskMismatchError, second.task_id):
            worker.stop(second.task_id)

        worker.stop(first.task_id)
        thread.join(2.0)
        self.assertEqual([], outcome["errors"])
        self.assertEqual(TaskStatus.QUEUED, self.service.get_task(second.task_id).status)

    def test_unresponsive_child_is_terminated_after_stop_timeout_and_cleaned(self):
        task = self.create_task()
        worker = self.worker(fake_unresponsive_child, stop_timeout=0.1)
        thread, outcome = self.run_in_thread(worker, task.task_id)
        self.wait_for_status(task.task_id, TaskStatus.RUNNING)
        child_pid = worker.active_child_pid

        stopped = worker.stop(task.task_id)
        thread.join(2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual([], outcome["errors"])
        self.assertEqual(TaskStatus.STOPPED, stopped.status)
        self.assertEqual(TaskStatus.STOPPED, outcome["results"][0].status)
        self.assertTrue(worker.last_stop_forced)
        self.assertLess(worker.last_exit_code, 0)
        self.assertFalse(worker.is_busy)
        self.assertIsNone(worker.active_child_pid)
        self.assertTrue(Path(worker.last_result.manifest_path).is_file())
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pid, 0)

    def test_external_shared_cancel_flag_stops_without_local_worker_stop(self):
        flag = self.context.Value("b", False)

        class SharedCancellationStore:
            def request(self, task_id):
                flag.value = True

            def is_requested(self, task_id):
                return bool(flag.value)

            def clear(self, task_id):
                flag.value = False

        service = SimulationService(
            self.repository,
            FakeScheduler(),
            self.backend,
            task_id_factory=lambda: "task-shared-cancel",
            cancellation_store=SharedCancellationStore(),
        )
        task = service.create_task(
            "run-shared-cancel",
            "project-worker",
            training_config().to_dict(),
        )
        worker = GPUWorker(
            service,
            self.artifacts_root,
            multiprocessing_context=self.context,
            process_target=fake_shared_cancel_child,
            shared_cancel_config={"flag": flag},
        )
        thread, outcome = self.run_in_thread(worker, task.task_id)
        deadline = time.monotonic() + 2.0
        while service.get_task(task.task_id).status != TaskStatus.RUNNING:
            if time.monotonic() >= deadline:
                self.fail("shared-cancel task did not enter running")
            time.sleep(0.01)

        service.request_stop(task.task_id)
        thread.join(2.0)

        self.assertEqual([], outcome["errors"])
        self.assertEqual(TaskStatus.STOPPED, service.get_task(task.task_id).status)
        self.assertEqual(TaskStatus.STOPPED, outcome["results"][0].status)
        self.assertFalse(worker.is_busy)


if __name__ == "__main__":
    unittest.main()
