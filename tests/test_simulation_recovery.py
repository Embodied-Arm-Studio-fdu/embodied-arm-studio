import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

from apps.api.main import create_app
from robotics.projects.models import (
    TrainingProfile,
    evaluation_rubric,
    parameter_schema,
    success_condition,
    worker_config,
)
from robotics.runs import ExecutionMode, InMemoryRunRepository, RunService, RunStatus
from robotics.simulation import (
    SimulationDispatchService,
    SimulationRecoveryService,
    SimulationStopService,
    SimulationTaskResponse,
)
from tests.http_client import ASGITestClient
from tests.test_simulation_dispatch_api import MemoryCreationService


class RecoveryFakeClient:
    def __init__(self, barrier=None):
        self.barrier = barrier
        self.create_calls = []
        self.stop_calls = []
        self.fail_create = set()
        self.fail_stop = set()
        self.owners = {}
        self.closed = False

    @staticmethod
    def snapshot(task_id, run_id, project_id, status):
        now = datetime(2026, 8, 27, 14, 0, tzinfo=timezone.utc)
        return SimulationTaskResponse(
            task_id=task_id,
            run_id=run_id,
            project_id=project_id,
            status=status,
            created_at=now,
            updated_at=now,
            config={},
        )

    def create_task(self, run_id, project_id, config):
        self.create_calls.append(run_id)
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        if run_id in self.fail_create:
            raise RuntimeError("injected create failure")
        task_id = "task-" + run_id
        self.owners[task_id] = (run_id, project_id)
        return self.snapshot(task_id, run_id, project_id, "queued")

    def stop_task(self, task_id):
        self.stop_calls.append(task_id)
        if task_id in self.fail_stop:
            raise RuntimeError("injected stop failure")
        run_id, project_id = self.owners[task_id]
        return self.snapshot(task_id, run_id, project_id, "stopped")

    def close(self):
        self.closed = True


class SimulationRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryRunRepository()
        self.service = RunService(self.repository)
        self.client = RecoveryFakeClient()
        self.dispatch = SimulationDispatchService(self.service, self.client)
        self.stop = SimulationStopService(self.service, self.client)
        self.recovery = SimulationRecoveryService(
            self.service, self.dispatch, self.stop, max_runs=20
        )

    def create_run(self, mode=ExecutionMode.SIMULATION, bound=False, status=None):
        seed = 7
        training_config = worker_config(TrainingProfile.SHORT, True, seed)
        teaching_snapshot = {
            "schema_version": "1.0",
            "project_id": "recovery-project",
            "course_id": "recovery-course",
            "class_id": "recovery-class",
            "title": "Recovery",
            "description": "",
            "instructions": "",
            "task_type": "training",
            "environment_type": "s1_push",
            "algorithm": "PPO",
            "success_condition": success_condition(),
            "training_profile": "short",
            "expected_training_steps": 32,
            "video_enabled": True,
            "parameter_schema": parameter_schema(),
            "resolved_parameters": {"seed": seed},
            "evaluation_rubric": evaluation_rubric(TrainingProfile.SHORT),
            "config_version": 1,
            "enabled": True,
            "project_enabled_at_creation": True,
            "starter_code_sha256": "0" * 64,
            "training_config": training_config,
        }
        run = self.service.create_run(
            user_id="recovery-user",
            project_id="recovery-project",
            execution_mode=mode,
            class_id=("recovery-class" if mode is ExecutionMode.SIMULATION else None),
            parameters=({"seed": seed} if mode is ExecutionMode.SIMULATION else {}),
            project_config_snapshot=(
                teaching_snapshot if mode is ExecutionMode.SIMULATION else {}
            ),
            code="",
        )
        if bound:
            task_id = "task-" + run.run_id
            run = self.service.bind_simulation_task(run.run_id, task_id)
            self.client.owners[task_id] = (run.run_id, run.project_id)
        if status == RunStatus.RUNNING:
            self.service.transition_status(run.run_id, RunStatus.INITIALIZING)
            run = self.service.transition_status(run.run_id, RunStatus.RUNNING)
        elif status == RunStatus.STOPPING:
            run = self.service.request_stop(run.run_id)
        elif status == RunStatus.COMPLETED:
            self.service.transition_status(run.run_id, RunStatus.INITIALIZING)
            self.service.transition_status(run.run_id, RunStatus.RUNNING)
            self.service.transition_status(run.run_id, RunStatus.FINALIZING)
            run = self.service.transition_status(run.run_id, RunStatus.COMPLETED)
        return run

    def test_reconciles_queued_unbound_and_stopping_bound(self):
        queued = self.create_run()
        stopping = self.create_run(bound=True, status=RunStatus.STOPPING)
        result = self.recovery.reconcile_simulation_runs()
        self.assertEqual((result.dispatched, result.stop_retried, result.failed), (1, 1, 0))
        self.assertEqual(
            self.service.get_run(queued.run_id).simulation_task_id,
            "task-" + queued.run_id,
        )
        self.assertEqual(self.client.stop_calls, [stopping.simulation_task_id])
        self.assertEqual(self.service.get_run(stopping.run_id).status, RunStatus.STOPPING)

    def test_skips_mock_terminal_and_already_bound_active_runs(self):
        self.create_run(mode=ExecutionMode.MOCK)
        self.create_run(bound=True)
        self.create_run(bound=True, status=RunStatus.RUNNING)
        self.create_run(bound=True, status=RunStatus.COMPLETED)
        result = self.recovery.reconcile_simulation_runs()
        self.assertEqual((result.dispatched, result.stop_retried, result.failed), (0, 0, 0))
        self.assertEqual(self.client.create_calls, [])
        self.assertEqual(self.client.stop_calls, [])

    def test_one_failure_does_not_block_later_run(self):
        successful = self.create_run()
        failing = self.create_run()
        # list_runs is newest-first, so failing is examined first.
        self.client.fail_create.add(failing.run_id)
        result = self.recovery.reconcile_simulation_runs()
        self.assertEqual((result.dispatched, result.failed), (1, 1))
        self.assertIsNone(self.service.get_run(failing.run_id).simulation_task_id)
        self.assertIsNotNone(self.service.get_run(successful.run_id).simulation_task_id)

    def test_max_runs_bounds_each_pass(self):
        for _ in range(3):
            self.create_run()
        bounded = SimulationRecoveryService(
            self.service, self.dispatch, self.stop, max_runs=2
        )
        result = bounded.reconcile_simulation_runs()
        self.assertEqual(result.examined, 2)
        self.assertEqual(result.dispatched, 2)

    def test_two_reconcilers_bind_the_same_task(self):
        run = self.create_run()
        barrier_client = RecoveryFakeClient(threading.Barrier(2))
        first = SimulationRecoveryService(
            self.service,
            SimulationDispatchService(self.service, barrier_client),
            SimulationStopService(self.service, barrier_client),
        )
        second = SimulationRecoveryService(
            self.service,
            SimulationDispatchService(self.service, barrier_client),
            SimulationStopService(self.service, barrier_client),
        )
        results = []
        threads = [
            threading.Thread(target=lambda recovery=item: results.append(
                recovery.reconcile_simulation_runs()
            ))
            for item in (first, second)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            self.service.get_run(run.run_id).simulation_task_id,
            "task-" + run.run_id,
        )
        self.assertEqual(barrier_client.create_calls, [run.run_id, run.run_id])

    def test_app_startup_runs_once_and_shutdown_closes_client(self):
        run = self.create_run()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(
                root / "runs",
                tick_seconds=0,
                project_root=root / "projects",
                run_repository=self.repository,
                run_creation_service=MemoryCreationService(self.repository),
                simulation_client=self.client,
            )
            with ASGITestClient(app):
                self.assertEqual(
                    app.state.run_service.get_run(run.run_id).simulation_task_id,
                    "task-" + run.run_id,
                )
                self.assertEqual(app.state.simulation_recovery_result.dispatched, 1)
        self.assertTrue(self.client.closed)

    def test_app_startup_retries_stopping_run(self):
        run = self.create_run(bound=True, status=RunStatus.STOPPING)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(
                root / "runs",
                tick_seconds=0,
                project_root=root / "projects",
                run_repository=self.repository,
                run_creation_service=MemoryCreationService(self.repository),
                simulation_client=self.client,
            )
            with ASGITestClient(app):
                self.assertEqual(app.state.simulation_recovery_result.stop_retried, 1)
                self.assertEqual(self.client.stop_calls, [run.simulation_task_id])
                self.assertEqual(
                    app.state.run_service.get_run(run.run_id).status,
                    RunStatus.STOPPING,
                )

    def test_startup_recovery_failure_does_not_block_read_api(self):
        run = self.create_run()
        self.client.fail_create.add(run.run_id)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_app(
                root / "runs",
                tick_seconds=0,
                project_root=root / "projects",
                run_repository=self.repository,
                run_creation_service=MemoryCreationService(self.repository),
                simulation_client=self.client,
            )
            with ASGITestClient(app) as api:
                self.assertEqual(api.get("/api/health").status_code, 200)
                self.assertEqual(app.state.simulation_recovery_result.failed, 1)


if __name__ == "__main__":
    unittest.main()
