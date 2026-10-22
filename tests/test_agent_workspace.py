import json
import tempfile
import unittest
from pathlib import Path

from agent.workspace import WorkspaceService


class WorkspaceServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        template_root = root / "templates"
        template_root.mkdir()
        (template_root / "controller.json").write_text('{\n  "action_scale": 0.35\n}\n', encoding="utf-8")
        self.service = WorkspaceService(root / "projects", template_root)
        self.project_id = self.service.create_project("lesson-one")["project_id"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_patch_requires_approval_and_can_rollback(self):
        diff = "\n".join([
            "--- a/controller.json",
            "+++ b/controller.json",
            "@@ -1,3 +1,3 @@",
            " {",
            '-  "action_scale": 0.35',
            '+  "action_scale": 0.55',
            " }",
        ])
        proposal = self.service.propose_patch(self.project_id, "controller.json", diff)
        self.assertEqual(proposal.status, "pending")
        self.assertIn("0.35", self.service.read_file(self.project_id, "controller.json"))
        with self.assertRaisesRegex(ValueError, "二次确认"):
            self.service.approve(proposal.patch_id)

        applied = self.service.approve(proposal.patch_id, confirm_high_risk=True)
        self.assertIn("0.55", self.service.read_file(self.project_id, "controller.json"))
        self.service.rollback(self.project_id, applied.snapshot_id)
        self.assertIn("0.35", self.service.read_file(self.project_id, "controller.json"))
        audit_records = [json.loads(line) for line in self.service.audit_path.read_text(encoding="utf-8").splitlines()]
        self.assertIn("approve_patch", [record["action"] for record in audit_records])

    def test_path_traversal_and_wrong_baseline_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "非法工作区路径"):
            self.service.read_file(self.project_id, "../controller.json")
        diff = "\n".join([
            "--- a/controller.json",
            "+++ b/controller.json",
            "@@ -1 +1 @@",
            '-{"action_scale": 99}',
            '+{"action_scale": 0.4}',
        ])
        with self.assertRaisesRegex(ValueError, "基线"):
            self.service.propose_patch(self.project_id, "controller.json", diff)

    def test_workspace_crud_search_and_snapshot(self):
        self.service.create_directory(self.project_id, "src")
        self.service.write_file(self.project_id, "src/task.py", "reward = distance\n")
        self.assertEqual(self.service.read_file(self.project_id, "src/task.py"), "reward = distance\n")
        self.assertEqual(self.service.search(self.project_id, "distance")[0]["line"], 1)

        self.service.move(self.project_id, "src/task.py", "src/environment.py")
        snapshot_id = self.service.create_snapshot(self.project_id)["snapshot_id"]
        self.service.write_file(self.project_id, "src/environment.py", "reward = 0\n")
        self.service.rollback(self.project_id, snapshot_id)

        self.assertIn("src/environment.py", self.service.list_files(self.project_id))
        self.assertIn("distance", self.service.read_file(self.project_id, "src/environment.py"))
        tree = self.service.file_tree(self.project_id)
        self.assertEqual(tree[0]["type"], "directory")

    def test_workspace_history_diff_and_starter_restore(self):
        self.service.write_file(self.project_id, "controller.json", '{\n  "action_scale": 0.45\n}\n')
        self.service.write_file(self.project_id, "controller.json", '{\n  "action_scale": 0.55\n}\n')

        history = self.service.file_history(self.project_id, "controller.json")
        self.assertEqual(len(history), 2)
        self.assertIn("0.55", history[0]["content"])
        self.assertEqual(self.service.starter_diff(self.project_id, "controller.json")["status"], "modified")

        self.service.restore_version(self.project_id, "controller.json", history[1]["version_id"])
        self.assertIn("0.45", self.service.read_file(self.project_id, "controller.json"))
        restored = self.service.restore_starter(self.project_id)
        self.assertTrue(restored["snapshot_id"].startswith("snapshot-"))
        self.assertIn("0.35", self.service.read_file(self.project_id, "controller.json"))
        self.assertEqual(self.service.starter_diff(self.project_id, "controller.json")["status"], "clean")

    def test_workspace_rejects_unsafe_paths_and_extensions(self):
        with self.assertRaisesRegex(ValueError, "非法工作区路径"):
            self.service.write_file(self.project_id, "../escape.py", "pass")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            self.service.write_file(self.project_id, "payload.sh", "exit 0")


if __name__ == "__main__":
    unittest.main()