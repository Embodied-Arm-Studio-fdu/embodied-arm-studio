import unittest
from types import SimpleNamespace

import torch

from simulation_core.experimental.manipulation.auto_release_env import (
    AutoReleaseFullPlaceEnvironment,
)
from simulation_core.ppo.policy import MLPActorCritic


READY = {
    "target_zone": True, "legal_release_height": True, "stable_grasp": True,
    "vertical_speed_small": True, "translation_stable": True,
    "rotation_stable": True, "finite": True, "safe_force": True,
    "no_penetration": True, "workspace_valid": True,
}


class AutoReleaseTests(unittest.TestCase):
    def fake_env(self, condition):
        env = AutoReleaseFullPlaceEnvironment()
        env._release_started = env._release_latched = False
        env._release_height = None
        env._release_trigger_streak = env._auto_release_streak = 0
        env.scene = SimpleNamespace(snapshot=lambda: {})
        env._release_ready_condition = lambda snapshot, clearance: dict(condition)
        return env

    def test_requires_three_consecutive_ready_steps(self):
        env = self.fake_env(READY)
        self.assertFalse(env._apply_release_trigger(1.0, .010))
        self.assertFalse(env._apply_release_trigger(-1.0, .010))
        self.assertTrue(env._apply_release_trigger(0.0, .010))
        self.assertTrue(env._auto_release_ready)
        self.assertTrue(env._release_latched)

    def test_high_state_never_releases_and_resets_streak(self):
        condition = dict(READY); condition["legal_release_height"] = False
        env = self.fake_env(condition)
        for _ in range(10):
            self.assertFalse(env._apply_release_trigger(1.0, .040))
        self.assertEqual(0, env._auto_release_streak)
        self.assertFalse(env._release_latched)

    def test_masked_fourth_dimension_has_no_ppo_gradient(self):
        policy = MLPActorCritic(observation_dim=3, action_dim=4,
                                action_low=(-1.,) * 4, action_high=(1.,) * 4)
        policy.set_active_action_mask((1, 1, 1, 0))
        observation = torch.zeros((2, 3))
        actions = torch.tensor(((.1, .2, .3, -.9), (.1, .2, .3, .9)))
        log_prob, entropy, _ = policy.evaluate_actions(observation, actions)
        self.assertAlmostEqual(float(log_prob[0]), float(log_prob[1]), places=6)
        (log_prob.sum() + entropy.sum()).backward()
        self.assertEqual(0.0, float(policy.actor[-1].weight.grad[3].abs().max()))
        self.assertEqual(0.0, float(policy.log_std.grad[3].abs()))


if __name__ == "__main__":
    unittest.main()
