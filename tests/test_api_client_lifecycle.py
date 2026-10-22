import tempfile
import threading
import time
import unittest
from pathlib import Path

from apps.api.main import create_app
from tests.http_client import ASGITestClient


class ApiClientLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.app_number = 0

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_app(self):
        self.app_number += 1
        suffix = str(self.app_number)
        return create_app(
            self.root / ("runs-" + suffix),
            tick_seconds=0,
            project_root=self.root / ("projects-" + suffix),
        )

    def launch(self, client):
        return client.post(
            "/api/runs/train",
            json={
                "environment_id": "mock-reach",
                "strategy_id": "joint-p-control",
                "max_steps": 4,
            },
        )

    def wait_run(self, client, run_id):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            response = client.get("/api/runs/%s" % run_id)
            if response.json()["status"] not in {"queued", "running"}:
                return response
            time.sleep(0.002)
        self.fail("HTTP lifecycle Run did not finish")

    def test_health_uses_real_asgi_request(self):
        with ASGITestClient(self.make_app()) as client:
            response = client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["adapter"], "mock")

    def test_unknown_run_returns_404(self):
        with ASGITestClient(self.make_app()) as client:
            response = client.get("/api/runs/not-a-run")
            self.assertEqual(response.status_code, 404)
            self.assertIn("未知运行", response.json()["detail"])

    def test_training_returns_202(self):
        with ASGITestClient(self.make_app()) as client:
            response = self.launch(client)
            self.assertEqual(response.status_code, 202)
            self.wait_run(client, response.json()["run_id"])

    def test_multiple_requests_do_not_hang(self):
        with ASGITestClient(self.make_app()) as client:
            statuses = [client.get("/api/health").status_code for _ in range(10)]
            self.assertEqual(statuses, [200] * 10)
            self.assertEqual(client.get("/api/runs").status_code, 200)

    def test_multiple_apps_and_clients_open_and_close(self):
        for _ in range(3):
            with ASGITestClient(self.make_app()) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)

    def test_two_apps_keep_run_state_isolated(self):
        first_app = self.make_app()
        second_app = self.make_app()
        with ASGITestClient(first_app) as first, ASGITestClient(second_app) as second:
            launched = self.launch(first)
            run_id = launched.json()["run_id"]
            self.wait_run(first, run_id)
            self.assertEqual(second.get("/api/runs").json(), [])
            self.assertEqual(second.get("/api/runs/%s" % run_id).status_code, 404)

    def test_client_close_leaves_no_test_threads(self):
        baseline = {thread.ident for thread in threading.enumerate()}
        with ASGITestClient(self.make_app()) as client:
            self.assertEqual(client.get("/api/health").status_code, 200)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            residual = [
                thread
                for thread in threading.enumerate()
                if thread.ident not in baseline and thread.is_alive()
            ]
            if not residual:
                break
            time.sleep(0.005)
        self.assertEqual(residual, [])

    def test_failed_request_does_not_prevent_close_or_reopen(self):
        client = ASGITestClient(self.make_app())
        self.assertEqual(client.get("/api/runs/missing").status_code, 404)
        client.close()
        with ASGITestClient(self.make_app()) as replacement:
            self.assertEqual(replacement.get("/api/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
