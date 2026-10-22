import math
import unittest
from types import SimpleNamespace

from simulation_core.experimental.manipulation.evidence import (
    GraspEvidenceConfig, evaluate_grasp,
)
from simulation_core.experimental.manipulation.grasp_env import (
    GraspEnvConfig, GraspPPOEnvironment,
)


def snapshot(*, jaw=.033, aligned=True, left=1.0, right=1.0,
             cube_force=1.0, non_finite=False):
    cube = [.45, 0., .398, 0., 0., 0., 1.] + [0.] * 6
    ee = ([.458, -.009, .386] if aligned else [.391, -.082, .390]) + [0., 0., 0., 1.] + [0.] * 6
    q = [0., 1., 1., 0., 0., 0., jaw, jaw]
    if non_finite:
        q[0] = float("nan")
    forces = {"8_Link": left, "7_Link": right, "cube": cube_force,
              "table": 0.0, "ee_link": 0.0}
    return {
        "q": q, "dq": [0.] * 8, "ee_state": ee, "cube_state": cube,
        "estimated_aperture": .0561,
        "bodies": {name: {"force": [value, 0., 0.], "state": [0.] * 13}
                   for name, value in forces.items()},
    }


class CandidateStabilizationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.env = GraspPPOEnvironment(GraspEnvConfig())
        self.command = SimpleNamespace(jaw_target=.025, workspace_clamped=False,
                                       joint_limit_clamped=False)

    def test_all_base_evidence_held_for_six_steps_forms_candidate(self):
        results = [self.env._candidate(snapshot(), self.command) for _ in range(6)]
        self.assertEqual([False] * 5 + [True], results)
        self.assertEqual(6, self.env.config.candidate_window)

    def test_blockage_loss_on_step_three_prevents_candidate(self):
        results = []
        for step in range(6):
            state = snapshot(jaw=.026 if step == 2 else .033)
            results.append(self.env._candidate(state, self.command))
        self.assertFalse(any(results))

    def test_bilateral_force_loss_on_step_four_prevents_candidate(self):
        results = []
        for step in range(6):
            state = snapshot(left=0.0 if step == 3 else 1.0)
            results.append(self.env._candidate(state, self.command))
        self.assertFalse(any(results))

    def test_corridor_loss_prevents_candidate(self):
        results = [self.env._candidate(snapshot(aligned=step != 2), self.command)
                   for step in range(6)]
        self.assertFalse(any(results))

    def test_unsafe_and_non_finite_keep_failure_semantics(self):
        unsafe = snapshot(left=51.0)
        self.assertEqual("UNSAFE_FORCE", self.env._abort_reason(unsafe, self.command))
        non_finite = snapshot(non_finite=True)
        self.assertEqual("NON_FINITE", self.env._abort_reason(non_finite, self.command))

    def test_candidate_does_not_change_verifier_success_thresholds(self):
        result = evaluate_grasp(
            config=GraspEvidenceConfig(self_contact_floor=.01), actual_jaw=.033,
            object_in_corridor=True, object_table_clearance=.020,
            object_rise=.020, ee_rise=.021,
            relative_translation_drift=.001, relative_rotation_drift=.05,
            state_values=(0.0,) * 20, left_force=1.0, right_force=1.0,
            object_force=1.0)
        self.assertTrue(result.grasp_verified)

    def test_cube_not_following_lift_is_not_verified(self):
        result = evaluate_grasp(
            config=GraspEvidenceConfig(self_contact_floor=.01), actual_jaw=.033,
            object_in_corridor=True, object_table_clearance=.002,
            object_rise=.002, ee_rise=.025,
            relative_translation_drift=.023, relative_rotation_drift=.05,
            state_values=(0.0,) * 20, left_force=1.0, right_force=1.0,
            object_force=1.0)
        self.assertFalse(result.grasp_verified)


if __name__ == "__main__":
    unittest.main()
