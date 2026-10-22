import math
import unittest

from simulation_core.experimental.manipulation.evidence import (
    GraspEvidenceConfig, evaluate_grasp,
)
from simulation_core.experimental.manipulation.grasp_env import (
    GRASP_VERIFIER_IMPLEMENTATION_REVISION, GraspPPOEnvironment,
    candidate_local_ee_rise,
)
from simulation_core.task_catalog import task_catalog_document
from simulation_core.task_registry import PRODUCTION_TASK_REGISTRY


EXPECTED_IDS = [
    "demo_s1_circle_tracking", "demo_s1_joint_position",
    "demo_s1_joint_sweep", "demo_s1_line_tracking",
    "demo_s1_target_pose", "s1_push", "s1_reach",
]
EXPECTED_DIGEST = "sha256:c0f3feb0771f4ab636c61fa53ba0d49d5f2a489a224b6890ae38ed798423f791"


class GraspVerifierReferenceTests(unittest.TestCase):
    def _evidence(self, **overrides):
        values = {
            "config": GraspEvidenceConfig(self_contact_floor=.01),
            "actual_jaw": .033,
            "object_in_corridor": True,
            "object_table_clearance": .020,
            "object_rise": .020661,
            "ee_rise": .020878,
            "relative_translation_drift": .000545,
            "relative_rotation_drift": .0735,
            "state_values": (0.0,) * 20,
            "left_force": 3.4, "right_force": 3.2, "object_force": 1.0,
        }
        values.update(overrides)
        return evaluate_grasp(**values)

    def test_stage_a_bug_fixture_uses_candidate_local_reference(self):
        reset_ee_z = .432657
        candidate_ee_z = .398354
        terminal_ee_z = .419232
        old_rise = terminal_ee_z - reset_ee_z
        new_rise = candidate_local_ee_rise(terminal_ee_z, candidate_ee_z)
        self.assertAlmostEqual(-.013425, old_rise, places=6)
        self.assertAlmostEqual(.020878, new_rise, places=6)
        self.assertFalse(self._evidence(ee_rise=old_rise).grasp_verified)
        self.assertTrue(self._evidence(ee_rise=new_rise).grasp_verified)
        self.assertEqual("grasp-verifier/v1.1", GRASP_VERIFIER_IMPLEMENTATION_REVISION)

    def test_anchor_is_immutable_and_resettable(self):
        env = object.__new__(GraspPPOEnvironment)
        env._verification_start_ee_pose = None
        snapshot = {"ee_state": [.1, -.2, .398354, 0, 0, 0, 1]}
        env._set_verification_anchor(snapshot)
        self.assertEqual(tuple(snapshot["ee_state"][:7]), env._verification_start_ee_pose)
        with self.assertRaises(RuntimeError):
            env._set_verification_anchor({"ee_state": [0, 0, .4, 0, 0, 0, 1]})
        env._clear_verification_anchor()
        self.assertIsNone(env._verification_start_ee_pose)
        with self.assertRaises(RuntimeError):
            candidate_local_ee_rise(.42, None)

    def test_cube_not_following_still_fails(self):
        evidence = self._evidence(object_rise=.002, ee_rise=.020)
        self.assertFalse(evidence.grasp_verified)
        self.assertFalse(evidence.flags["co_motion"])

    def test_dropped_cube_still_fails(self):
        evidence = self._evidence(
            object_table_clearance=-.001, object_rise=-.002,
            relative_translation_drift=.020,
            left_force=0.0, right_force=0.0, object_force=0.0)
        self.assertFalse(evidence.grasp_verified)
        self.assertFalse(evidence.flags["object_off_table"])

    def test_cube_rise_threshold_is_unchanged(self):
        evidence = self._evidence(object_rise=.0119, ee_rise=.020)
        self.assertFalse(evidence.grasp_verified)
        self.assertFalse(evidence.flags["object_rose"])

    def test_clearance_threshold_is_unchanged(self):
        evidence = self._evidence(object_table_clearance=.0079)
        self.assertFalse(evidence.grasp_verified)
        self.assertFalse(evidence.flags["object_off_table"])

    def test_translation_threshold_is_unchanged(self):
        evidence = self._evidence(relative_translation_drift=.0121)
        self.assertFalse(evidence.grasp_verified)
        self.assertFalse(evidence.flags["translation_stable"])

    def test_rotation_threshold_is_unchanged(self):
        evidence = self._evidence(relative_rotation_drift=.251)
        self.assertFalse(evidence.grasp_verified)
        self.assertFalse(evidence.flags["rotation_stable"])

    def test_jaw_safety_threshold_is_unchanged(self):
        evidence = self._evidence(actual_jaw=.0099)
        self.assertFalse(evidence.grasp_verified)
        self.assertFalse(evidence.flags["jaw_safe"])

    def test_non_finite_still_fails(self):
        evidence = self._evidence(state_values=(0.0, math.nan))
        self.assertFalse(evidence.grasp_verified)
        self.assertFalse(evidence.flags["finite"])

    def test_production_isolation(self):
        self.assertIn("s1_place", [d.id for d in PRODUCTION_TASK_REGISTRY.definitions()])
        document = task_catalog_document(PRODUCTION_TASK_REGISTRY)
        self.assertEqual("task-catalog/v2", document["catalog_version"])


if __name__ == "__main__":
    unittest.main()
