import unittest

from simulation_core.evaluation_subprocess import evaluation_contract
from simulation_core.reach_adapter import ReachAdapter, ReachConfig


class Backend:
    def __init__(self):
        from simulation_core.push_adapter import PUSH_FIXED_ORIENTATION_XYZW, PUSH_FIXED_TARGET_Z
        self.ee=(.36,0.,PUSH_FIXED_TARGET_Z)+PUSH_FIXED_ORIENTATION_XYZW+(0.,)*6
    def setup(self): pass
    def reset(self): pass
    def read_robot_state(self):
        from simulation_core.s1_isaac_gym_adapter import S1State
        return S1State((0.,1.,1.,0.,0.,0.,.01,.01),(0.,)*8,self.ee,[[0.]*6]*6)
    def set_ee_target(self,target): return True
    def set_goal_overlay(self,target,radius): pass
    def step(self,n): pass
    def close(self): pass


class ReachEvaluationTests(unittest.TestCase):
    def test_target_sequence_reproducible_but_not_fixed(self):
        def sample():
            env=ReachAdapter(ReachConfig(seed=10000),backend=Backend()); env.initialize()
            values=[]
            for _ in range(4): env.reset(); values.append(env.target_xy)
            env.close(); return values
        self.assertEqual(sample(),sample()); self.assertGreater(len(set(sample())),1)

    def test_reach_quality_threshold_is_90_percent_and_80_wilson(self):
        base={"task_id":"s1_reach","episode_count":100,"success_count":90,"success_rate":.9,
              "confidence_interval_95":{"lower":.81,"upper":.95}}
        self.assertTrue(evaluation_contract(base)["policy_passed"])
        base["confidence_interval_95"]["lower"]=.79
        self.assertFalse(evaluation_contract(base)["policy_passed"])


if __name__ == "__main__": unittest.main()
