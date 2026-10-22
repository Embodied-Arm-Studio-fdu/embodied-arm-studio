import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs import ExecutionMode, InMemoryRunRepository, RunService, RunStatus
from robotics.simulation import (
    SimulationNetworkError,
    SimulationRejectedError,
    SimulationResponseError,
    SimulationServerError,
    SimulationTaskResponse,
    SimulationTimeoutError,
)
from tests.http_client import ASGITestClient
from tests.test_simulation_dispatch_api import MemoryCreationService


class StopFakeClient:
    def __init__(self):
        self.owners = {}
        self.stop_calls = []
        self.error = None
        self.mismatch = None
        self.closed = False

    def create_task(self, run_id, project_id, config):
        raise AssertionError("stop tests must not create tasks")

    def register(self, task_id, run_id, project_id):
        self.owners[task_id] = (run_id, project_id)

    def stop_task(self, task_id):
        self.stop_calls.append(task_id)
        if self.error is not None:
            raise self.error
        run_id, project_id = self.owners[task_id]
        if self.mismatch == "task":
            task_id = "wrong-task"
        elif self.mismatch == "run":
            run_id = "wrong-run"
        elif self.mismatch == "project":
            project_id = "wrong-project"
        now = datetime(2026, 8, 27, 13, 0, tzinfo=timezone.utc)
        return SimulationTaskResponse(
            task_id=task_id,
            run_id=run_id,
            project_id=project_id,
            status="stopped",
            created_at=now,
            updated_at=now,
            config={},
        )

    def close(self):
        self.closed = True


class SimulationStopApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = InMemoryRunRepository()
        self.service = RunService(self.repository)
        self.simulation = StopFakeClient()
        root = Path(self.temp.name)
        self.app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=self.repository,
            run_creation_service=MemoryCreationService(self.repository),
            simulation_client=self.simulation,
            current_user_provider=StaticCurrentUserProvider("stop-user"),
        )
        self.client = ASGITestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def create_run(self, status=RunStatus.QUEUED, bound=True):
        run = self.service.create_run(
            user_id="stop-user",
            project_id="stop-project",
            execution_mode=ExecutionMode.SIMULATION,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        if bound:
            task_id = "task-" + run.run_id
            run = self.service.bind_simulation_task(run.run_id, task_id)
            self.simulation.register(task_id, run.run_id, run.project_id)
        if status == RunStatus.INITIALIZING:
            run = self.service.transition_status(run.run_id, status)
        elif status == RunStatus.RUNNING:
            self.service.transition_status(run.run_id, RunStatus.INITIALIZING)
            run = self.service.transition_status(run.run_id, status)
        elif status == RunStatus.FINALIZING:
            self.service.transition_status(run.run_id, RunStatus.INITIALIZING)
            self.service.transition_status(run.run_id, RunStatus.RUNNING)
            run = self.service.transition_status(run.run_id, status)
        elif status == RunStatus.STOPPING:
            run = self.service.request_stop(run.run_id)
        elif status == RunStatus.COMPLETED:
            self.service.transition_status(run.run_id, RunStatus.INITIALIZING)
            self.service.transition_status(run.run_id, RunStatus.RUNNING)
            self.service.transition_status(run.run_id, RunStatus.FINALIZING)
            run = self.service.transition_status(run.run_id, status)
        elif status == RunStatus.FAILED:
            self.service.transition_status(run.run_id, RunStatus.INITIALIZING)
            run = self.service.transition_status(run.run_id, status)
        elif status == RunStatus.STOPPED:
            if bound:
                self.service.request_stop(run.run_id)
                run = self.service.transition_status(run.run_id, RunStatus.STOPPED)
            else:
                run = self.service.request_stop(run.run_id)
        return run

    def stop(self, run_id):
        return self.client.post("/api/v1/runs/%s/stop" % run_id)

    def test_queued_unbound_stops_without_http(self):
        run = self.create_run(bound=False)
        response = self.stop(run.run_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "stopped")
        self.assertEqual(self.simulation.stop_calls, [])

    def test_queued_bound_and_running_enter_stopping_and_call_http(self):
        for status in (RunStatus.QUEUED, RunStatus.RUNNING):
            with self.subTest(status=status.value):
                run = self.create_run(status)
                response = self.stop(run.run_id)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "stopping")
                self.assertIsNone(response.json()["ended_at"])
                self.assertEqual(self.simulation.stop_calls[-1], run.simulation_task_id)

    def test_stopping_retries_stop(self):
        run = self.create_run(RunStatus.STOPPING)
        first = self.stop(run.run_id)
        second = self.stop(run.run_id)
        self.assertEqual(first.json()["status"], "stopping")
        self.assertEqual(second.json()["status"], "stopping")
        self.assertEqual(self.simulation.stop_calls, [run.simulation_task_id] * 2)

    def test_initializing_and_finalizing_remain_conflicts(self):
        for status in (RunStatus.INITIALIZING, RunStatus.FINALIZING):
            run = self.create_run(status)
            self.assertEqual(self.stop(run.run_id).status_code, 409)
        self.assertEqual(self.simulation.stop_calls, [])

    def test_terminal_repeats_are_idempotent_without_http(self):
        for status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.STOPPED):
            run = self.create_run(status)
            response = self.stop(run.run_id)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], status.value)
        self.assertEqual(self.simulation.stop_calls, [])

    def test_stop_failure_stays_stopping_and_can_retry(self):
        run = self.create_run(RunStatus.RUNNING)
        cases = [
            (SimulationTimeoutError("timeout"), 503),
            (SimulationNetworkError("network"), 503),
            (SimulationServerError(500), 503),
            (SimulationRejectedError(401), 503),
            (SimulationRejectedError(404), 502),
            (SimulationRejectedError(409), 409),
            (SimulationRejectedError(422), 502),
            (SimulationResponseError("invalid"), 502),
        ]
        for error, expected in cases:
            self.simulation.error = error
            response = self.stop(run.run_id)
            self.assertEqual(response.status_code, expected)
            self.assertEqual(self.service.get_run(run.run_id).status, RunStatus.STOPPING)
        self.simulation.error = None
        self.assertEqual(self.stop(run.run_id).status_code, 200)
        self.assertEqual(self.service.get_run(run.run_id).status, RunStatus.STOPPING)

    def test_stop_snapshot_identity_mismatch_is_rejected(self):
        for mismatch in ("task", "run", "project"):
            with self.subTest(mismatch=mismatch):
                run = self.create_run(RunStatus.RUNNING)
                self.simulation.mismatch = mismatch
                response = self.stop(run.run_id)
                self.assertEqual(response.status_code, 502)
                self.assertEqual(self.service.get_run(run.run_id).status, RunStatus.STOPPING)
        self.simulation.mismatch = None


if __name__ == "__main__":
    unittest.main()
