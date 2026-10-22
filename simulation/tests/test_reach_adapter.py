import math
import unittest

from simulation_core.push_adapter import PUSH_FIXED_ORIENTATION_XYZW, PUSH_FIXED_TARGET_Z
from simulation_core.reach_adapter import ReachActionMapper, ReachAdapter, ReachConfig
from simulation_core.s1_isaac_gym_adapter import S1State


class FakeBackend:
    def __init__(self, ee=(.36, 0., PUSH_FIXED_TARGET_Z)):
        self.ee = list(ee) + list(PUSH_FIXED_ORIENTATION_XYZW) + [0.] * 6
        self.q = [0., 1., 1., 0., 0., 0., .01, .01]
        self.dq = [0.] * 8; self.target = None; self.overlay = None; self.closed = False
    def setup(self): pass
    def reset(self): pass
    def read_robot_state(self): return S1State(tuple(self.q), tuple(self.dq), tuple(self.ee), [[0.]*6]*6)
    def set_ee_target(self, target): self.target = target; return True
    def set_goal_overlay(self, target_xy, radius): self.overlay = (target_xy, radius)
    def step(self, n):
        if self.target:
            self.ee[0] = self.target.position[0]; self.ee[1] = self.target.position[1]
    def close(self): self.closed = True


class ReachAdapterTests(unittest.TestCase):
    def test_normalized_action_scale_fixed_z_orientation_and_clamp(self):
        config = ReachConfig()
        mapper = ReachActionMapper(config); mapper.reset((.36, 0., .1))
        target = mapper.map((1., -.5))
        self.assertAlmostEqual(.365, target.position[0]); self.assertAlmostEqual(-.0025, target.position[1])
        self.assertEqual(PUSH_FIXED_TARGET_Z, target.position[2])
        self.assertEqual(PUSH_FIXED_ORIENTATION_XYZW, target.orientation_xyzw)
        for _ in range(100): target = mapper.map((1., 1.))
        self.assertEqual(config.workspace_max_xy_m, target.position[:2])

    def test_reset_and_three_step_success_hold(self):
        backend = FakeBackend(); env = ReachAdapter(ReachConfig(seed=3), backend=backend)
        env.initialize(); obs = env.reset()
        self.assertEqual((0., 0.), obs["previous_action_xy"])
        self.assertEqual(.02, backend.overlay[1])
        backend.ee[0], backend.ee[1] = env.target_xy
        env._mapper._target_xy = env.target_xy
        for expected_done in (False, False, True):
            _, _, done, info = env.step((0., 0.)); self.assertEqual(expected_done, done)
        self.assertTrue(info["success"]); env.close(); self.assertTrue(backend.closed)

    def test_timeout_and_target_sampling_contract(self):
        config = ReachConfig(seed=8, max_steps=2, success_hold_steps=3)
        backend = FakeBackend(); env = ReachAdapter(config, backend=backend); env.initialize()
        env.reset(); target = env.target_xy
        self.assertTrue(config.target_center_xy_m[0]-config.target_half_extent_xy_m[0] <= target[0] <= config.target_center_xy_m[0]+config.target_half_extent_xy_m[0])
        self.assertGreater(math.hypot(target[0]-backend.ee[0], target[1]-backend.ee[1]), config.success_distance_m)
        env.step((0.,0.)); _,_,done,info=env.step((0.,0.)); self.assertTrue(done); self.assertTrue(info["timeout"])


if __name__ == "__main__": unittest.main()
