from copy import deepcopy
import json
from pathlib import Path
import unittest

from simulation_core import (
    CATALOG_VERSION, PRODUCTION_TASK_REGISTRY, TaskRegistry,
    canonical_json_bytes, sha256_digest, task_catalog_document,
)


EXPECTED_ROLES = {
    "s1_push": "training",
    "s1_reach": "training",
    "s1_grasp": "training",
    "s1_lift": "training",
    "s1_lower_release": "training",
    "s1_place": "training",
    "demo_s1_target_pose": "atomic_demo",
    "demo_s1_line_tracking": "atomic_demo",
    "demo_s1_circle_tracking": "atomic_demo",
    "demo_s1_joint_position": "atomic_demo",
    "demo_s1_joint_sweep": "diagnostic",
    "demo_s1_scripted_pick_place": "atomic_demo",
}


def default_config(definition):
    return {
        parameter["name"]: deepcopy(parameter["default"])
        for parameter in definition.parameters
        if not (parameter.get("nullable") and parameter["default"] is None)
    }


class BrainSemanticCatalogTests(unittest.TestCase):
    def setUp(self):
        self.document = task_catalog_document()
        self.tasks = {task["id"]: task for task in self.document["tasks"]}

    def test_exactly_twelve_unique_tasks_and_complete_semantics(self):
        self.assertEqual(12, len(self.tasks))
        self.assertEqual(set(EXPECTED_ROLES), set(self.tasks))
        required = {
            "task_id", "display_names", "summary", "run_kind",
            "environment_type", "planning_role", "intent_tags",
            "preconditions", "parameter_schema_ref", "parameter_schema_digest",
            "expected_metrics", "expected_artifacts", "goal_success",
            "resource_profile", "composition", "safety",
        }
        for task_id, task in self.tasks.items():
            with self.subTest(task_id=task_id):
                self.assertTrue(required.issubset(task))
                self.assertEqual(task_id, task["task_id"])
                self.assertEqual(task_id, task["environment_type"])
                self.assertEqual({"zh-CN", "en"}, set(task["display_names"]))
                self.assertEqual({"zh-CN", "en"}, set(task["summary"]))
                self.assertTrue(task["intent_tags"])
                self.assertTrue(task["preconditions"])
                self.assertTrue(task["expected_metrics"])

    def test_planning_roles_success_composition_resources_and_safety(self):
        for task_id, role in EXPECTED_ROLES.items():
            task = self.tasks[task_id]
            self.assertEqual(role, task["planning_role"])
            expected_availability = "formal" if role == "training" else "metrics_only"
            self.assertEqual(expected_availability, task["goal_success"]["availability"])
            self.assertEqual(False, task["composition"]["workflow_enabled"])
            self.assertEqual(False, task["composition"]["preserves_world_state"])
            self.assertEqual(False, task["safety"]["real_robot_allowed"])
            self.assertEqual(False, task["safety"]["arbitrary_joint_targets"])
            self.assertEqual(True, task["safety"]["parameter_whitelist_only"])
            self.assertEqual(True, task["resource_profile"]["gpu_required"])
            self.assertEqual("long" if role == "training" else "short",
                             task["resource_profile"]["duration_class"])
        for task_id in ("s1_push", "s1_reach"):
            self.assertEqual("evaluation", self.tasks[task_id]["goal_success"]["source"])
            self.assertIn("does not mean policy passed",
                          self.tasks[task_id]["goal_success"]["description"])

    def test_catalog_version_digest_stability_semantic_change_and_order(self):
        self.assertEqual("task-catalog/v2", CATALOG_VERSION)
        digest_input = {
            "catalog_version": self.document["catalog_version"],
            "tasks": self.document["tasks"],
        }
        self.assertEqual(self.document["catalog_digest"], sha256_digest(digest_input))
        self.assertEqual(self.document, task_catalog_document())

        changed = deepcopy(digest_input)
        changed["tasks"][0]["summary"]["en"] += " changed"
        self.assertNotEqual(self.document["catalog_digest"], sha256_digest(changed))

        reversed_registry = TaskRegistry(reversed(PRODUCTION_TASK_REGISTRY.definitions()))
        self.assertEqual(self.document, task_catalog_document(reversed_registry))

    def test_backend_canonicalization_vector(self):
        fixture_path = Path(__file__).parent / "fixtures" / "task_catalog_canonicalization_v1.json"
        with fixture_path.open("r", encoding="utf-8") as stream:
            vector = json.load(stream)
        self.assertEqual(vector["canonical_json"].encode("utf-8"),
                         canonical_json_bytes(vector["canonical_input"]))
        self.assertEqual(vector["expected_digest"],
                         sha256_digest(vector["canonical_input"]))

    def test_parameter_schema_digest_ref_and_parser_alignment(self):
        digests = set()
        for definition in PRODUCTION_TASK_REGISTRY.definitions():
            task = self.tasks[definition.id]
            schema = task["parameter_schema"]
            digest = sha256_digest(schema)
            with self.subTest(task_id=definition.id):
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual("TaskDefinition.config_parser",
                                 schema["validation_authority"])
                self.assertEqual(digest, task["parameter_schema_digest"])
                self.assertEqual(digest, sha256_digest(definition.parameter_schema()))
                self.assertEqual(
                    "registry://task/%s/parameters?digest=%s" % (definition.id, digest),
                    task["parameter_schema_ref"],
                )
                parsed = definition.parse_config(default_config(definition))
                self.assertEqual(definition.id, parsed.environment_type)

                extra = default_config(definition)
                extra["unknown_parameter"] = True
                with self.assertRaises(ValueError):
                    definition.parse_config(extra)

                invalid = default_config(definition)
                invalid["seed"] = -1
                with self.assertRaises(ValueError):
                    definition.parse_config(invalid)

                wrong_type = default_config(definition)
                wrong_type["seed"] = True
                with self.assertRaises(ValueError):
                    definition.parse_config(wrong_type)
            digests.add(digest)
        self.assertEqual(12, len(digests))

        changed = deepcopy(self.tasks["s1_push"]["parameter_schema"])
        changed["parameters"][0]["default"] = "changed"
        self.assertNotEqual(self.tasks["s1_push"]["parameter_schema_digest"],
                            sha256_digest(changed))

    def test_training_profile_drift_and_conditions_are_explicit(self):
        push = self.tasks["s1_push"]
        reach = self.tasks["s1_reach"]
        by_name = lambda task: {item["name"]: item for item in task["parameters"]}
        self.assertEqual(256, by_name(push)["episode_steps"]["default"])
        self.assertEqual(128, by_name(reach)["episode_steps"]["default"])
        self.assertEqual(
            {"episode_steps": {"const": 256}, "physics_steps_per_action": {"const": 8}},
            push["parameter_schema"]["constraints"][0]["then"],
        )
        self.assertEqual(
            {"episode_steps": {"const": 128}, "physics_steps_per_action": {"const": 8}},
            reach["parameter_schema"]["constraints"][0]["then"],
        )
        self.assertEqual("cross_field", reach["parameter_schema"]["constraints"][1]["kind"])
        for name in ("object_size", "object_start", "goal_position"):
            self.assertEqual("backend_resolved", by_name(push)[name]["planning_access"])

    def test_legacy_catalog_fields_are_preserved(self):
        legacy = {
            "id", "display_name", "task_type", "robot", "supports_training",
            "supports_video", "parameters",
        }
        for task in self.tasks.values():
            self.assertTrue(legacy.issubset(task))
            self.assertIsInstance(task["display_name"], str)
            self.assertIsInstance(task["parameters"], list)


if __name__ == "__main__":
    unittest.main()
