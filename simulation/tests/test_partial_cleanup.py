import tempfile
import unittest
from pathlib import Path

from simulation_core.artifacts import ArtifactPathError
from simulation_core.partial_cleanup import (
    TRAINING_VIDEO_PARTIAL,
    apply_failed_run_partial_cleanup,
    plan_failed_run_partials,
)


class PartialCleanupTestCase(unittest.TestCase):
    def test_plan_is_run_scoped_and_dry_run_never_deletes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "project" / "run-one" / TRAINING_VIDEO_PARTIAL
            second = root / "project" / "run-two" / TRAINING_VIDEO_PARTIAL
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"one")
            second.write_bytes(b"two")

            plan = plan_failed_run_partials(root, "project", "run-one")
            removed = apply_failed_run_partial_cleanup(root, plan)

            self.assertEqual((TRAINING_VIDEO_PARTIAL,), plan.targets)
            self.assertEqual((), removed)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_explicit_apply_removes_only_the_displayed_run_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "project" / "run-one" / TRAINING_VIDEO_PARTIAL
            second = root / "project" / "run-two" / TRAINING_VIDEO_PARTIAL
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            plan = plan_failed_run_partials(root, "project", "run-one")

            removed = apply_failed_run_partial_cleanup(root, plan, apply=True)

            self.assertEqual((TRAINING_VIDEO_PARTIAL,), removed)
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())

    def test_changed_target_set_and_path_escape_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = plan_failed_run_partials(root, "project", "run")
            partial = root / "project" / "run" / TRAINING_VIDEO_PARTIAL
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"late")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                apply_failed_run_partial_cleanup(root, plan, apply=True)
            with self.assertRaises(ArtifactPathError):
                plan_failed_run_partials(root, "../project", "run")


if __name__ == "__main__":
    unittest.main()
