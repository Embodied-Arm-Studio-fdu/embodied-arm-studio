import unittest

from simulation_core.experimental.manipulation.grasp_diagnostics import (
    build_event_timeline, classify_cube_contact, verifier_failure,
)
from simulation_core.experimental.manipulation.grasp_schema import (
    GRASP_ACTION_SCHEMA_ID, GRASP_OBSERVATION_DIM, GRASP_OBSERVATION_SCHEMA_ID,
    GRASP_REWARD_VERSION, GRASP_SUCCESS_CRITERIA_VERSION,
)


class GraspDiagnosticGeometryTests(unittest.TestCase):
    def test_cube_contact_regions_are_cube_local(self):
        self.assertEqual("Y_POS_SIDE", classify_cube_contact((0, .022, 0))["region"])
        self.assertEqual("Y_NEG_SIDE", classify_cube_contact((0, -.022, 0))["region"])
        self.assertEqual("TABLE_NEAR", classify_cube_contact((0, 0, -.0225))["region"])
        self.assertAlmostEqual(.5, classify_cube_contact((0, .022, 0))["normalized_height"])

    def test_contract_is_locked(self):
        self.assertEqual("s1-grasp-observation/v1", GRASP_OBSERVATION_SCHEMA_ID)
        self.assertEqual(40, GRASP_OBSERVATION_DIM)
        self.assertEqual("s1-grasp-action/v1", GRASP_ACTION_SCHEMA_ID)
        self.assertEqual("s1-grasp-reward/v3", GRASP_REWARD_VERSION)
        self.assertEqual("s1-grasp-success/v1", GRASP_SUCCESS_CRITERIA_VERSION)

    def test_timeline_aligns_candidate_before_verifier(self):
        rows = []
        for index in range(32):
            rows.append({
                "stage": "LEARNING" if index < 24 else "VERIFYING",
                "physics_step": index, "time": index / 60,
                "control_step": index // 4,
                "jaw_actual": [.04, .04], "jaw_command": [.035, .035],
                "contacts": [], "ee_pose": [0, 0, index / 1000, 0, 0, 0, 1],
            })
        timeline = build_event_timeline(rows, candidate_window=6)
        self.assertEqual(23, timeline["T4"]["row_index"])
        self.assertEqual(24, timeline["T5"]["row_index"])

    def test_verifier_exposes_reset_relative_ee_rise(self):
        rows = []
        for index in range(8):
            stage = "LEARNING" if index < 4 else "VERIFYING"
            z = .388 + max(0, index - 3) * .005
            rows.append({
                "stage": stage, "physics_step": index, "time": index / 60,
                "control_step": index // 4, "jaw_actual": [.033, .033],
                "jaw_command": [.025, .025], "contacts": [],
                "ee_pose": [0, 0, z, 0, 0, 0, 1], "cube_pose": [0, 0, .3995, 0, 0, 0, 1],
                "cube_rise_from_reset": 0.0, "cube_table_clearance": .002,
                "ee_cube_relative_translation": [0, 0, .0115],
                "q": [0] * 8, "dq": [0] * 8,
                "net_contact_force": {"8_Link": [0, 0, 0], "7_Link": [0, 0, 0]},
                "table_support_ratio": 1.0,
            })
        timeline = build_event_timeline(rows, candidate_window=1)
        failure = verifier_failure(rows, timeline, initial_ee_z=.433)
        self.assertEqual("CUBE_NOT_CLEAR_OF_TABLE", failure["first_failure"])
        self.assertLess(failure["formal_reset_relative_ee_rise"], 0.0)


if __name__ == "__main__":
    unittest.main()
