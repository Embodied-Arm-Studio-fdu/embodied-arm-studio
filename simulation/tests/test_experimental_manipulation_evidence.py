import unittest

from simulation_core.experimental.manipulation.evidence import (
    GraspEvidenceConfig, evaluate_grasp, quaternion_distance,
)


class EvidenceTests(unittest.TestCase):
    def kwargs(self):
        return dict(config=GraspEvidenceConfig(), actual_jaw=.025,
            object_in_corridor=True, object_table_clearance=.02,
            object_rise=.025, ee_rise=.025, relative_translation_drift=.002,
            relative_rotation_drift=.02, state_values=(0.,)*20,
            left_force=1., right_force=1., object_force=1.)

    def test_verified_requires_combination_not_contact_bit(self):
        result=evaluate_grasp(**self.kwargs())
        self.assertTrue(result.grasp_candidate); self.assertTrue(result.grasp_verified)
        values=self.kwargs(); values.update(left_force=100.,right_force=100.,object_force=100.,object_table_clearance=0.)
        result=evaluate_grasp(**values)
        self.assertTrue(result.flags["force_evidence"])
        self.assertFalse(result.grasp_verified)
        self.assertEqual("object_off_table",result.failure_reason)

    def test_force_is_optional_for_verified_grasp(self):
        values=self.kwargs(); values.update(left_force=0.,right_force=0.,object_force=0.)
        result=evaluate_grasp(**values)
        self.assertTrue(result.grasp_verified)
        self.assertFalse(result.flags["force_evidence"])

    def test_unsafe_jaw_and_nonfinite_rejected(self):
        values=self.kwargs(); values.update(actual_jaw=.005)
        self.assertFalse(evaluate_grasp(**values).grasp_candidate)
        values=self.kwargs(); values.update(state_values=(float("nan"),))
        self.assertFalse(evaluate_grasp(**values).grasp_verified)

    def test_quaternion_distance_sign_invariant(self):
        self.assertAlmostEqual(0.,quaternion_distance((0,0,0,1),(0,0,0,-1)))


if __name__ == "__main__": unittest.main()

