import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

from robotics.projects import TrainingProfile, worker_config
from robotics.runs.postgres_idempotency import PostgreSQLProjectRunCreationService


class RunCreationConfigVersionContractTest(unittest.TestCase):
    def test_production_creation_passes_the_project_config_version(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(
            PostgreSQLProjectRunCreationService.create_run
        )))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "worker_config"
        ]
        self.assertEqual(len(calls), 1)
        version_keywords = [
            keyword.value for keyword in calls[0].keywords
            if keyword.arg == "config_version"
        ]
        self.assertEqual(len(version_keywords), 1)
        self.assertEqual(
            ast.dump(version_keywords[0]),
            ast.dump(ast.parse("project.config_version", mode="eval").body),
        )

    def test_worker_config_uses_the_frozen_project_version(self):
        project = SimpleNamespace(
            training_profile=TrainingProfile.STANDARD.value,
            video_enabled=False,
            config_version=3,
        )

        actual = worker_config(
            TrainingProfile(project.training_profile),
            project.video_enabled,
            48,
            config_version=project.config_version,
        )

        self.assertEqual(actual["rollout_steps"], 200)
        self.assertEqual(actual["num_updates"], 100)
        self.assertEqual(actual["episode_steps"], 256)
        self.assertEqual(actual["physics_steps_per_action"], 8)
        self.assertEqual(actual["checkpoint_every"], 10)
        self.assertEqual(actual["run_mode"], "training")
        self.assertFalse(actual["video_enabled"])

    def test_legacy_standard_version_remains_dispatchable(self):
        legacy = worker_config(
            TrainingProfile.STANDARD,
            False,
            42,
            config_version=1,
        )

        self.assertEqual(legacy["rollout_steps"], 16)
        self.assertEqual(legacy["num_updates"], 20)
        self.assertNotIn("run_mode", legacy)


if __name__ == "__main__":
    unittest.main()
