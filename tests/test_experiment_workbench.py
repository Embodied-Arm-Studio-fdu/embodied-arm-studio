import tempfile
import unittest
from pathlib import Path

from robotics.experiment_workbench import ExperimentWorkbenchService, preflight


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = ExperimentWorkbenchService(Path(self.directory.name), Path(__file__).parents[1] / "simulation/src/simulation_core")

    def test_owned_history_and_restore_preserve_revisions(self):
        initial = self.service.document("alice", "s1_reach")
        changed = self.service.save("alice", "s1_reach", "def combine_reward(terms):\n    return 0\n", "starter")
        self.assertEqual(self.service.document("bob", "s1_reach")["revision_id"], "starter")
        restored = self.service.restore("alice", "s1_reach", "starter", changed["revision_id"])
        self.assertEqual(restored["source"], initial["source"])
        self.assertEqual(len(restored["history"]), 2)
        with self.assertRaises(RuntimeError):
            self.service.save("alice", "s1_reach", "stale", changed["revision_id"])
        with self.assertRaises(ValueError):
            self.service.restore("bob", "s1_reach", changed["revision_id"], "starter")

    def test_source_inventory_is_read_only_and_rejects_path_escape(self):
        self.assertIn("push_adapter.py", self.service.sources())
        self.assertIn("evaluation_subprocess.py", self.service.sources())
        self.assertIn("scripts/place_diagnostic_evaluation.py", self.service.sources())
        self.assertTrue(self.service.source("push_adapter.py")["read_only"])
        with self.assertRaises(ValueError):
            self.service.source("../../apps/api/main.py")

    def test_preflight_rejects_bad_contract_and_division(self):
        self.assertFalse(preflight("s1_reach", "def combine_reward(terms):\n    return 1 / 0")["valid"])
        self.assertTrue(preflight("s1_place", self.service.document("alice", "s1_place")["starter"])["valid"])
        self.assertTrue(preflight("s1_reach", self.service.document("alice", "s1_reach")["starter"])["valid"])

    def test_authenticated_api_drafts_and_diagnosis(self):
        import os
        os.environ["RUN_REPOSITORY_MODE"] = "memory"
        from apps.api.main import create_app
        from tests.auth import StaticCurrentUserProvider
        from tests.http_client import ASGITestClient
        app = create_app(storage_root=Path(self.directory.name) / "runs", project_root=Path(self.directory.name) / "projects", run_repository_mode="memory", current_user_provider=StaticCurrentUserProvider("alice"))
        with ASGITestClient(app) as client:
            self.assertEqual(client.get("/api/v1/workbench/sources", headers={"Authorization": ""}).status_code, 401)
            starter = client.get("/api/v1/workbench/drafts/s1_reach").json()
            sources = client.get("/api/v1/workbench/sources").json()["files"]
            self.assertIn("scripts/place_diagnostic_evaluation.py", sources)
            self.assertGreater(len(client.get("/api/v1/workbench/sources/training_task_runner.py").json()["content"]), 10000)
            place = client.get("/api/v1/workbench/drafts/s1_place").json()
            self.assertTrue(client.post("/api/v1/workbench/drafts/s1_place/preflight", json={"source": place["source"]}).json()["valid"])
            english = client.post("/api/v1/workbench/drafts/s1_place/diagnose", json={"source": place["source"], "locale": "en"}).json()["content"]
            self.assertFalse(any("\u4e00" <= char <= "\u9fff" for char in english))
            saved = client.put("/api/v1/workbench/drafts/s1_reach", json={"source": starter["source"], "base_revision": "starter"})
            self.assertEqual(saved.status_code, 200)
            conflict = client.put("/api/v1/workbench/drafts/s1_reach", json={"source": "stale", "base_revision": "starter"})
            self.assertEqual(conflict.status_code, 409)
            diagnosis = client.post("/api/v1/workbench/drafts/s1_reach/diagnose", json={"source": "def combine_reward(terms):\n    return 1 / 0"})
            self.assertEqual(diagnosis.status_code, 200)
            self.assertIn("预检未通过", diagnosis.json()["content"])
            denied = client.post("/api/v1/workbench/drafts/s1_reach/diagnose", json={"source": starter["source"], "run_id": "unreadable-run"})
            self.assertEqual(denied.status_code, 404)