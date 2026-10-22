import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs import (
    ExecutionMode,
    InMemoryRunRepository,
    RunEvent,
    RunEventType,
    RunKind,
    RunStatus,
)
from tests.http_client import ASGITestClient


CANONICAL_STATUSES = {
    "queued",
    "initializing",
    "running",
    "finalizing",
    "completed",
    "stopping",
    "stopped",
    "failed",
}

RUN_FIELDS = {
    "run_id",
    "user_id",
    "project_id",
    "class_id",
    "status",
    "execution_mode",
    "run_kind",
    "parameters",
    "project_config_snapshot",
    "code_sha256",
    "simulation_task_id",
    "progress",
    "metrics_summary",
    "error",
    "last_sequence",
    "training_steps",
    "update_steps",
    "stop_reason",
    "created_at",
    "started_at",
    "ended_at",
    "updated_at",
}


class V1RunApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider("mock-adapter"),
        )
        self.client = ASGITestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def launch(self, client=None, max_steps=5):
        client = client or self.client
        response = client.post(
            "/api/runs/train",
            json={
                "environment_id": "mock-reach",
                "strategy_id": "joint-p-control",
                "seed": 17,
                "max_steps": max_steps,
                "config": {"action_scale": 0.4},
            },
        )
        self.assertEqual(response.status_code, 202)
        return response.json()

    def wait_terminal(self, run_id, client=None):
        client = client or self.client
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            response = client.get("/api/v1/runs/%s" % run_id)
            self.assertEqual(response.status_code, 200)
            if response.json()["status"] in {"completed", "failed", "stopped"}:
                return response.json()
            time.sleep(0.002)
        self.fail("v1 Run did not finish")

    def test_legacy_create_is_visible_in_v1_list_and_detail(self):
        created = self.launch()
        detail = self.wait_terminal(created["run_id"])
        listed = self.client.get("/api/v1/runs")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()[0], detail)
        self.assertEqual(set(detail), RUN_FIELDS)
        self.assertIn(detail["status"], CANONICAL_STATUSES)
        self.assertEqual(detail["execution_mode"], "mock")
        self.assertEqual(detail["run_kind"], "train")
        self.assertTrue(detail["created_at"].endswith("Z"))
        self.assertTrue(detail["updated_at"].endswith("Z"))
        self.assertIsNone(detail["error"])

    def test_v1_list_has_stable_newest_first_order(self):
        first = self.launch()
        self.wait_terminal(first["run_id"])
        second = self.launch()
        self.wait_terminal(second["run_id"])
        listed = self.client.get("/api/v1/runs").json()
        self.assertEqual(
            [run["run_id"] for run in listed],
            [second["run_id"], first["run_id"]],
        )

    def test_canonical_events_are_sorted_and_support_sequence_catchup(self):
        created = self.launch(max_steps=6)
        self.wait_terminal(created["run_id"])
        response = self.client.get(
            "/api/v1/runs/%s/events" % created["run_id"]
        )
        self.assertEqual(response.status_code, 200)
        events = response.json()
        sequences = [event["sequence"] for event in events]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(sequences, list(range(1, len(events) + 1)))
        self.assertTrue(all(event["schema_version"] == "1.0" for event in events))
        self.assertTrue(all(event["type"] in {"state", "metric", "heartbeat", "error"} for event in events))
        threshold = sequences[len(sequences) // 2]
        caught_up = self.client.get(
            "/api/v1/runs/%s/events?after_sequence=%s"
            % (created["run_id"], threshold)
        )
        self.assertEqual(
            [event["sequence"] for event in caught_up.json()],
            [sequence for sequence in sequences if sequence > threshold],
        )

    def test_invalid_sequence_and_unknown_resources_map_to_http_errors(self):
        created = self.launch()
        self.assertEqual(
            self.client.get(
                "/api/v1/runs/%s/events?after_sequence=-1" % created["run_id"]
            ).status_code,
            422,
        )
        self.assertEqual(self.client.get("/api/v1/runs/missing").status_code, 404)
        self.assertEqual(
            self.client.get("/api/v1/runs/missing/events").status_code, 404
        )

    def test_running_stop_and_terminal_repeat_are_canonical_and_idempotent(self):
        root = Path(self.temp_dir.name)
        app = create_app(
            root / "slow-runs",
            tick_seconds=0.005,
            project_root=root / "slow-projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider("mock-adapter"),
        )
        with ASGITestClient(app) as client:
            created = self.launch(client=client, max_steps=500)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                current = client.get("/api/v1/runs/%s" % created["run_id"])
                if current.json()["status"] == "running":
                    break
                time.sleep(0.002)
            self.assertEqual(current.json()["status"], "running")
            stopped = client.post("/api/v1/runs/%s/stop" % created["run_id"])
            self.assertEqual(stopped.status_code, 200)
            self.assertIn(stopped.json()["status"], {"stopping", "stopped"})
            terminal = self.wait_terminal(created["run_id"], client)
            self.assertEqual(terminal["status"], "stopped")
            repeated = client.post("/api/v1/runs/%s/stop" % created["run_id"])
            self.assertEqual(repeated.status_code, 200)
            self.assertEqual(repeated.json(), terminal)

    def test_initializing_and_finalizing_stop_return_409(self):
        service = self.app.state.adapter.run_service
        for status in (RunStatus.INITIALIZING, RunStatus.FINALIZING):
            run = service.create_run(
                user_id="mock-adapter",
                project_id="test-project",
                execution_mode=ExecutionMode.MOCK,
                run_kind=RunKind.TRAIN,
                parameters={},
                project_config_snapshot={},
                code="",
            )
            service.transition_status(run.run_id, RunStatus.INITIALIZING)
            if status == RunStatus.FINALIZING:
                service.transition_status(run.run_id, RunStatus.RUNNING)
                service.transition_status(run.run_id, RunStatus.FINALIZING)
            response = self.client.post("/api/v1/runs/%s/stop" % run.run_id)
            self.assertEqual(response.status_code, 409)

    def test_structured_canonical_error_is_not_downgraded(self):
        service = self.app.state.adapter.run_service
        run = service.create_run(
            user_id="mock-adapter",
            project_id="test-project",
            execution_mode=ExecutionMode.MOCK,
            run_kind=RunKind.PLAY,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        service.apply_event(RunEvent(
            event_id="test-error-event",
            sequence=1,
            type=RunEventType.ERROR,
            run_id=run.run_id,
            occurred_at=datetime.now(timezone.utc),
            payload={"code": "TEST", "message": "structured", "retryable": False},
        ))
        detail = self.client.get("/api/v1/runs/%s" % run.run_id)
        self.assertEqual(
            detail.json()["error"],
            {"code": "TEST", "message": "structured", "retryable": False},
        )

    def test_legacy_run_and_events_remain_legacy_shapes(self):
        created = self.launch()
        self.wait_terminal(created["run_id"])
        legacy_run = self.client.get("/api/runs/%s" % created["run_id"]).json()
        self.assertEqual(legacy_run["mode"], "train")
        self.assertNotIn("execution_mode", legacy_run)
        self.assertIn(legacy_run["status"], {"queued", "running", "completed", "stopped", "failed"})
        legacy_events = self.client.get(
            "/api/runs/%s/events" % created["run_id"]
        ).json()
        self.assertTrue(all(event["type"] in {"status", "metric", "error"} for event in legacy_events))
        self.assertTrue(all("sequence" not in event for event in legacy_events))
        self.assertTrue(all("schema_version" not in event for event in legacy_events))

    def test_app_instances_are_isolated_and_openapi_has_v1_create(self):
        created = self.launch()
        root = Path(self.temp_dir.name)
        second_app = create_app(
            root / "second-runs",
            tick_seconds=0,
            project_root=root / "second-projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider("mock-adapter"),
        )
        with ASGITestClient(second_app) as second:
            self.assertEqual(second.get("/api/v1/runs").json(), [])
            self.assertEqual(
                second.get("/api/v1/runs/%s" % created["run_id"]).status_code,
                404,
            )
        paths = self.app.openapi()["paths"]
        self.assertIn("/api/runs", paths)
        self.assertIn("/api/v1/runs", paths)
        self.assertEqual(set(paths["/api/v1/runs"]), {"get", "post"})
        self.assertEqual(
            set(paths["/api/v1/runs/{run_id}/stop"]), {"post"}
        )


if __name__ == "__main__":
    unittest.main()
