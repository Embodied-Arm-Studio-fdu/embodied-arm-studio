import unittest

from simulation_core.ppo.policy import REACH_POLICY_SCHEMA, flatten_observation


class ReachObservationTests(unittest.TestCase):
    def test_exact_18d_order(self):
        obs = {"arm_q":range(6), "arm_dq":range(10,16), "ee_to_goal_xy":(20,21),
               "ee_velocity_xy":(30,31), "previous_action_xy":(40,41)}
        value = flatten_observation(obs, REACH_POLICY_SCHEMA)
        self.assertEqual((18,), tuple(value.shape))
        self.assertEqual(list(range(6))+list(range(10,16))+[20,21,30,31,40,41], value.tolist())
        self.assertEqual(1, REACH_POLICY_SCHEMA.observation_schema_version)
        self.assertEqual("s1_reach", REACH_POLICY_SCHEMA.task_id)


if __name__ == "__main__": unittest.main()
