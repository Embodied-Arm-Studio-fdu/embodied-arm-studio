import unittest

from simulation_core.task_catalog import task_catalog_document
from simulation_core.task_registry import PRODUCTION_TASK_REGISTRY


class ExperimentalIsolationTests(unittest.TestCase):
    def test_place_is_now_a_formal_production_task(self):
        self.assertTrue(PRODUCTION_TASK_REGISTRY.get("s1_place").supports_training)

    def test_catalog_version_digest_and_ids_are_unchanged(self):
        document=task_catalog_document()
        self.assertEqual("task-catalog/v2", document["catalog_version"])
        self.assertIn("s1_place", tuple(item["id"] for item in document["tasks"]))

    def test_experimental_tasks_are_not_resolvable(self):
        self.assertNotIn("s1_place", ())


if __name__ == "__main__": unittest.main()
