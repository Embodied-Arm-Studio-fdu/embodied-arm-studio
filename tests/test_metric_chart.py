import io
import unittest
from types import SimpleNamespace

from PIL import Image
from robotics.artifacts.charts import metric_chart_png


class MetricChartTests(unittest.TestCase):
    def test_png_contains_real_curve_and_is_nonblank(self):
        events = [SimpleNamespace(type=SimpleNamespace(value="metric"), payload={"metrics": {"update_index": index, "average_reward": index * 0.2}}) for index in range(1, 4)]
        content = metric_chart_png(events, "average_reward")
        self.assertTrue(content.startswith(b"\x89PNG\r\n\x1a\n"))
        image = Image.open(io.BytesIO(content))
        self.assertEqual(image.size, (1600, 640))
        self.assertLess(image.getextrema()[0][0], image.getextrema()[0][1])

    def test_missing_and_nonfinite_data_are_not_fabricated(self):
        events = [SimpleNamespace(type=SimpleNamespace(value="metric"), payload={"metrics": {"update_index": 1, "average_reward": float("nan")}})]
        with self.assertRaises(ValueError):
            metric_chart_png(events, "average_reward")

    def test_chart_api_enforces_run_authorization(self):
        import tempfile
        from pathlib import Path
        from datetime import datetime, timezone
        from apps.api.main import create_app
        from tests.auth import StaticCurrentUserProvider
        from tests.http_client import ASGITestClient
        from robotics.runs import ExecutionMode, RunEvent, RunEventType
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(storage_root=Path(directory) / "runs", project_root=Path(directory) / "projects", run_repository_mode="memory", current_user_provider=StaticCurrentUserProvider("alice"))
            service = app.state.run_service
            own = service.create_run(user_id="alice", project_id="project", execution_mode=ExecutionMode.SIMULATION, parameters={}, project_config_snapshot={}, code="")
            other = service.create_run(user_id="bob", project_id="project", execution_mode=ExecutionMode.SIMULATION, parameters={}, project_config_snapshot={}, code="")
            event = RunEvent(event_id="chart-event", sequence=1, type=RunEventType.METRIC, run_id=own.run_id, task_id=None, occurred_at=datetime.now(timezone.utc), payload={"metrics": {"update_index": 1, "average_reward": 0.25}})
            service.repository.append_event(event)
            with ASGITestClient(app) as client:
                response = client.get(f"/api/v1/runs/{own.run_id}/charts/average_reward.png")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.content.startswith(b"\x89PNG"))
                self.assertEqual(client.get(f"/api/v1/runs/{other.run_id}/charts/average_reward.png").status_code, 404)
                self.assertEqual(client.get(f"/api/v1/runs/{own.run_id}/charts/entropy.png").status_code, 404)