import copy
import tempfile
import unittest
from pathlib import Path

from robotics.pipelines import PipelineRepository


class PipelineRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = PipelineRepository(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_builtin_pipeline_is_valid(self):
        document = self.repository.get("pipeline_mock_reach_baseline")
        validation = self.repository.validate(document)
        self.assertTrue(validation["valid"])
        self.assertEqual(document["revision"], 1)
        self.assertEqual(document["capability"], "mock")
        self.assertTrue(validation["warnings"])

    def test_revision_is_immutable(self):
        first = self.repository.get("pipeline_mock_reach_baseline")
        changed = copy.deepcopy(first)
        changed["name"] = "Changed Pipeline"
        second = self.repository.save_revision(changed)

        self.assertEqual(second["revision"], 2)
        self.assertEqual(self.repository.get(first["pipeline_id"], 1)["name"], "Mock Reach Baseline")

    def test_invalid_edge_type_is_rejected(self):
        document = self.repository.get("pipeline_mock_reach_baseline")
        document["edges"][0]["type"] = "text"
        validation = self.repository.validate(document)

        self.assertFalse(validation["valid"])
        self.assertIn("不匹配", " ".join(error["message"] for error in validation["errors"]))
        with self.assertRaises(ValueError):
            self.repository.save_revision(document)


if __name__ == "__main__":
    unittest.main()