import tempfile
import unittest
from pathlib import Path

from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs import (
    ExecutionMode,
    InMemoryRunRepository,
    RunKind,
    RunStatus,
)
from robotics.simulation import (
    SimulationNetworkError,
    SimulationRejectedError,
    SimulationResponseError,
    SimulationServerError,
    SimulationTimeoutError,
    StreamDescriptor,
    StreamSessionStatus,
)
from tests.http_client import ASGITestClient


STREAM_FIELDS = {
    "stream_id",
    "run_id",
    "task_id",
    "camera_id",
    "status",
    "webrtc_url",
    "whep_url",
    "width",
    "height",
    "fps",
}


class FakeSimulationClient:
    def __init__(self):
        self.calls = []
        self.stop_calls = []
        self.sessions = {}
        self.error = None
        self.override = None
        self.closed = False

    def create_task(self, run_id, project_id, config):
        raise AssertionError("stream tests must not dispatch a task")

    def stop_task(self, task_id):
        self.stop_calls.append(task_id)
        raise AssertionError("video control must not stop a task")

    def create_stream_session(self, task_id, camera_id="overview"):
        self.calls.append((task_id, camera_id))
        if self.error is not None:
            raise self.error
        if self.override is not None:
            return self.override
        key = (task_id, camera_id)
        if key not in self.sessions:
            opaque = "stream-opaque-%d" % (len(self.sessions) + 1)
            self.sessions[key] = StreamDescriptor(
                stream_id=opaque,
                run_id="run-placeholder",
                task_id=task_id,
                camera_id=camera_id,
                status=StreamSessionStatus.READY,
                webrtc_url="https://media.example/%s" % opaque,
                whep_url="https://media.example/%s/whep" % opaque,
                width=640,
                height=480,
                fps=15,
            )
        return self.sessions[key]

    def close(self):
        self.closed = True


class V1RunStreamApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.simulation = FakeSimulationClient()
        self.app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider("trusted-user"),
            simulation_client=self.simulation,
        )
        self.client = ASGITestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def create_run(
        self,
        *,
        user_id="trusted-user",
        mode=ExecutionMode.SIMULATION,
        bind=True,
        status=RunStatus.RUNNING,
    ):
        service = self.app.state.run_service
        run = service.create_run(
            user_id=user_id,
            project_id="project-1",
            execution_mode=mode,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        if mode == ExecutionMode.SIMULATION and bind:
            run = service.bind_simulation_task(run.run_id, "task-%s" % run.run_id)
        if status != RunStatus.QUEUED:
            run = service.transition_status(run.run_id, RunStatus.INITIALIZING)
        if status not in {RunStatus.QUEUED, RunStatus.INITIALIZING}:
            run = service.transition_status(run.run_id, RunStatus.RUNNING)
        if status in {RunStatus.FINALIZING, RunStatus.COMPLETED}:
            run = service.transition_status(run.run_id, RunStatus.FINALIZING)
        if status == RunStatus.COMPLETED:
            run = service.transition_status(run.run_id, RunStatus.COMPLETED)
        return run

    def stream_url(self, run_id, camera_id=None):
        url = "/api/v1/runs/%s/stream" % run_id
        return url if camera_id is None else url + "?camera_id=" + camera_id

    def prepare_descriptor(self, run, camera_id="overview"):
        descriptor = StreamDescriptor(
            stream_id="stream-opaque-fixed",
            run_id=run.run_id,
            task_id=run.simulation_task_id,
            camera_id=camera_id,
            status=StreamSessionStatus.READY,
            webrtc_url="https://media.example/stream-opaque-fixed",
            whep_url="https://media.example/stream-opaque-fixed/whep",
            width=640,
            height=480,
            fps=15,
        )
        self.simulation.sessions[(run.simulation_task_id, camera_id)] = descriptor
        return descriptor

    def test_running_stream_defaults_and_repeated_get_is_idempotent(self):
        run = self.create_run()
        expected = self.prepare_descriptor(run)
        first = self.client.get(self.stream_url(run.run_id))
        second = self.client.get(self.stream_url(run.run_id, "overview"))
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(first.json(), expected.to_dict())
        self.assertEqual(set(first.json()), STREAM_FIELDS)
        self.assertNotIn("rtsp", first.text.lower())
        self.assertEqual(
            self.simulation.calls,
            [(run.simulation_task_id, "overview")] * 2,
        )

    def test_explicit_camera_is_forwarded_and_identity_is_checked(self):
        run = self.create_run()
        self.prepare_descriptor(run, "overview")
        response = self.client.get(self.stream_url(run.run_id, "overview"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["camera_id"], "overview")

        valid = self.prepare_descriptor(run)
        for override in (
            StreamDescriptor(**dict(valid.to_dict(), run_id="another-run")),
            StreamDescriptor(**dict(valid.to_dict(), task_id="another-task")),
            StreamDescriptor(**dict(valid.to_dict(), camera_id="another-camera")),
        ):
            with self.subTest(override=override):
                self.simulation.override = override
                mismatch = self.client.get(self.stream_url(run.run_id))
                self.assertEqual(mismatch.status_code, 502)
        self.simulation.override = None

    def test_unknown_unauthorized_and_invalid_camera_are_hidden_or_rejected(self):
        unauthorized = self.create_run(user_id="another-user")
        self.assertEqual(
            self.client.get(self.stream_url("missing")).status_code, 404
        )
        self.assertEqual(
            self.client.get(self.stream_url(unauthorized.run_id)).status_code,
            404,
        )
        for camera_id in ("../camera", "bad/camera", "wrist", ""):
            with self.subTest(camera_id=camera_id):
                self.assertEqual(
                    self.client.get(
                        self.stream_url(unauthorized.run_id, camera_id)
                    ).status_code,
                    422,
                )

    def test_mock_unbound_nonrunning_and_terminal_runs_are_conflicts(self):
        runs = [
            self.create_run(mode=ExecutionMode.MOCK),
            self.create_run(bind=False, status=RunStatus.QUEUED),
            self.create_run(status=RunStatus.QUEUED),
            self.create_run(status=RunStatus.INITIALIZING),
            self.create_run(status=RunStatus.FINALIZING),
            self.create_run(status=RunStatus.COMPLETED),
        ]
        for run in runs:
            with self.subTest(mode=run.execution_mode, status=run.status):
                response = self.client.get(self.stream_url(run.run_id))
                self.assertEqual(response.status_code, 409)
        self.assertEqual(self.simulation.calls, [])

    def test_upstream_error_matrix_is_redacted_and_never_changes_run(self):
        run = self.create_run()
        before = self.app.state.run_service.get_run(run.run_id)
        cases = [
            (SimulationTimeoutError("secret timeout"), 503),
            (SimulationNetworkError("secret network"), 503),
            (SimulationServerError(500), 503),
            (SimulationServerError(503, "stream_not_ready", "secret"), 503),
            (SimulationRejectedError(401, "unauthorized", "secret"), 503),
            (SimulationRejectedError(404, "task_not_found", "secret"), 502),
            (
                SimulationRejectedError(
                    409, "stream_session_not_allowed", "secret"
                ),
                409,
            ),
            (SimulationRejectedError(400, "invalid_request", "secret"), 502),
            (SimulationRejectedError(400, "invalid_camera_id", "secret"), 502),
            (SimulationResponseError("rtsp://internal secret-token"), 502),
        ]
        for error, status_code in cases:
            with self.subTest(error=type(error).__name__, status=status_code):
                self.simulation.error = error
                response = self.client.get(self.stream_url(run.run_id))
                self.assertEqual(response.status_code, status_code)
                self.assertNotIn("secret", response.text)
                self.assertNotIn("rtsp", response.text.lower())
                self.assertEqual(
                    self.app.state.run_service.get_run(run.run_id), before
                )
        self.simulation.error = None
        self.assertEqual(self.simulation.stop_calls, [])

    def test_shutdown_closes_http_client_without_stopping_training(self):
        run = self.create_run()
        self.prepare_descriptor(run)
        self.assertEqual(
            self.client.get(self.stream_url(run.run_id)).status_code, 200
        )
        self.client.close()
        self.assertTrue(self.simulation.closed)
        self.assertEqual(self.simulation.stop_calls, [])
        self.assertEqual(
            self.app.state.run_service.get_run(run.run_id).status,
            RunStatus.RUNNING,
        )

    def test_missing_stream_client_is_503_and_only_get_route_is_published(self):
        root = Path(self.temp_dir.name)
        app = create_app(
            root / "unconfigured-runs",
            tick_seconds=0,
            project_root=root / "unconfigured-projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider("trusted-user"),
            simulation_environment={},
        )
        service = app.state.run_service
        run = service.create_run(
            user_id="trusted-user",
            project_id="project-1",
            execution_mode=ExecutionMode.SIMULATION,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        service.bind_simulation_task(run.run_id, "task-unconfigured")
        service.transition_status(run.run_id, RunStatus.INITIALIZING)
        service.transition_status(run.run_id, RunStatus.RUNNING)
        with ASGITestClient(app) as client:
            response = client.get(self.stream_url(run.run_id))
        self.assertEqual(response.status_code, 503)
        paths = app.openapi()["paths"]
        self.assertEqual(
            set(paths["/api/v1/runs/{run_id}/stream"]), {"get"}
        )
        self.assertNotIn(
            "/api/v1/runs/{run_id}/stream-session", paths
        )


if __name__ == "__main__":
    unittest.main()
