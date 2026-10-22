import tempfile
import time
import unittest
from pathlib import Path

from apps.api.main import create_app
from tests.http_client import ASGITestClient


RUN_HTTP_ROUTES = {
    ("/api/runs", "GET"),
    ("/api/runs/play", "POST"),
    ("/api/runs/train", "POST"),
    ("/api/runs/compare", "POST"),
    ("/api/runs/{run_id}", "GET"),
    ("/api/runs/{run_id}/stop", "POST"),
    ("/api/runs/{run_id}/events", "GET"),
    ("/api/runs/{run_id}/artifacts", "GET"),
    ("/api/runs/{run_id}/export", "POST"),
}


class RunRouterCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.app = create_app(
            root / "runs", tick_seconds=0, project_root=root / "projects"
        )
        self.client = ASGITestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def launch_and_wait(self, kind="train", max_steps=5, client=None):
        client = client or self.client
        response = client.post(
            "/api/runs/%s" % kind,
            json={
                "environment_id": "mock-reach",
                "strategy_id": "joint-p-control",
                "seed": 9,
                "max_steps": max_steps,
                "config": {"action_scale": 0.4},
            },
        )
        self.assertEqual(response.status_code, 202)
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            record = client.get("/api/runs/%s" % run_id)
            self.assertEqual(record.status_code, 200)
            if record.json()["status"] not in {"queued", "running"}:
                return run_id, record.json()
            time.sleep(0.002)
        self.fail("Run Router test run did not finish")

    def test_run_routes_are_registered_once_with_original_methods(self):
        registered = []
        websocket_routes = []
        canonical_websocket_routes = []
        for route in self.app.routes:
            if route.path.startswith("/api/runs"):
                registered.extend((route.path, method) for method in route.methods)
            if route.path == "/ws/runs/{run_id}":
                websocket_routes.append(route)
            if route.path == "/api/v1/runs/{run_id}/events" and not hasattr(
                route, "methods"
            ):
                canonical_websocket_routes.append(route)
        self.assertEqual(set(registered), RUN_HTTP_ROUTES)
        self.assertEqual(len(registered), len(RUN_HTTP_ROUTES))
        self.assertEqual(len(websocket_routes), 1)
        self.assertEqual(len(canonical_websocket_routes), 1)

    def test_train_play_list_and_detail_keep_legacy_json(self):
        train_id, train = self.launch_and_wait("train")
        play_id, play = self.launch_and_wait("play")
        self.assertEqual(train["mode"], "train")
        self.assertEqual(play["mode"], "play")
        listed = self.client.get("/api/runs")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(
            {record["run_id"] for record in listed.json()}, {train_id, play_id}
        )
        detail = self.client.get("/api/runs/%s" % train_id)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json(), train)

    def test_events_and_artifacts_keep_legacy_formats(self):
        run_id, _ = self.launch_and_wait(max_steps=4)
        events = self.client.get("/api/runs/%s/events" % run_id)
        self.assertEqual(events.status_code, 200)
        self.assertTrue(events.json())
        self.assertTrue(
            all(event["type"] in {"status", "metric", "error"} for event in events.json())
        )
        self.assertEqual(
            set(events.json()[0]),
            {"run_id", "type", "payload", "event_id", "timestamp", "version"},
        )
        artifacts = self.client.get("/api/runs/%s/artifacts" % run_id)
        self.assertEqual(artifacts.status_code, 200)
        self.assertIn(
            "metrics.jsonl", {artifact["name"] for artifact in artifacts.json()}
        )

    def test_compare_and_export_keep_existing_contracts(self):
        first_id, _ = self.launch_and_wait(max_steps=3)
        second_id, _ = self.launch_and_wait(max_steps=4)
        compared = self.client.post(
            "/api/runs/compare", json={"run_ids": [first_id, second_id]}
        )
        self.assertEqual(compared.status_code, 200)
        self.assertEqual(
            [item["run"]["run_id"] for item in compared.json()["runs"]],
            [first_id, second_id],
        )
        self.assertIn("distance", compared.json()["runs"][0]["final_metrics"])
        exported = self.client.post(
            "/api/runs/%s/export" % first_id, json={"format": "manifest"}
        )
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.json()["run_id"], first_id)
        self.assertFalse(exported.json()["real_robot_control_enabled"])

    def test_stop_and_http_error_mappings_are_unchanged(self):
        root = Path(self.temp_dir.name)
        slow_app = create_app(
            root / "slow-runs",
            tick_seconds=0.005,
            project_root=root / "slow-projects",
        )
        with ASGITestClient(slow_app) as client:
            response = client.post(
                "/api/runs/train",
                json={
                    "environment_id": "mock-reach",
                    "strategy_id": "joint-p-control",
                    "max_steps": 500,
                },
            )
            self.assertEqual(response.status_code, 202)
            run_id = response.json()["run_id"]
            stopped = client.post("/api/runs/%s/stop" % run_id)
            self.assertEqual(stopped.status_code, 200)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                stopped = client.get("/api/runs/%s" % run_id)
                if stopped.json()["status"] == "stopped":
                    break
                time.sleep(0.002)
            self.assertEqual(stopped.json()["status"], "stopped")
        self.assertEqual(self.client.get("/api/runs/missing").status_code, 404)
        invalid = self.client.post(
            "/api/runs/train",
            json={
                "environment_id": "mock-reach",
                "strategy_id": "joint-p-control",
                "max_steps": 0,
            },
        )
        self.assertEqual(invalid.status_code, 422)

    def test_openapi_run_paths_and_request_fields_are_unchanged(self):
        schema = self.app.openapi()
        path_methods = {
            (path, method.upper())
            for path, operations in schema["paths"].items()
            if path.startswith("/api/runs")
            for method in operations
        }
        self.assertEqual(path_methods, RUN_HTTP_ROUTES)
        run_create = schema["components"]["schemas"]["RunCreate"]
        self.assertEqual(
            set(run_create["properties"]),
            {"environment_id", "strategy_id", "seed", "max_steps", "config"},
        )
        self.assertEqual(set(run_create["required"]), {"environment_id", "strategy_id"})
        self.assertEqual(run_create["properties"]["seed"]["default"], 42)
        self.assertEqual(run_create["properties"]["max_steps"]["default"], 120)
        self.assertEqual(run_create["properties"]["max_steps"]["minimum"], 1)
        self.assertEqual(run_create["properties"]["max_steps"]["maximum"], 2000)
        compare = schema["components"]["schemas"]["CompareRequest"]["properties"]["run_ids"]
        self.assertEqual(compare["minItems"], 2)
        self.assertEqual(compare["maxItems"], 4)
        export = schema["components"]["schemas"]["ExportRequest"]
        self.assertEqual(export["properties"]["format"]["default"], "manifest")

    def test_create_app_instances_do_not_share_runs(self):
        run_id, _ = self.launch_and_wait()
        root = Path(self.temp_dir.name)
        second_app = create_app(
            root / "second-runs",
            tick_seconds=0,
            project_root=root / "second-projects",
        )
        with ASGITestClient(second_app) as second_client:
            self.assertEqual(second_client.get("/api/runs").json(), [])
            self.assertEqual(
                second_client.get("/api/runs/%s" % run_id).status_code, 404
            )


if __name__ == "__main__":
    unittest.main()
