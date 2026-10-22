import unittest

from simulation_core.reach_adapter import ReachAdapter, ReachConfig


class ReachRewardTests(unittest.TestCase):
    def setUp(self):
        self.env = object.__new__(ReachAdapter); self.env.config = ReachConfig()
        self.env._previous_distance = .10; self.env._mapper = __import__(
            "simulation_core.reach_adapter", fromlist=["ReachActionMapper"]
        ).ReachActionMapper(self.env.config)

    def test_progress_direction_stationary_and_success(self):
        toward = self.env.compute_reward_terms(.099, (0.,0.))
        away = self.env.compute_reward_terms(.101, (0.,0.))
        still = self.env.compute_reward_terms(.10, (0.,0.))
        success = self.env.compute_reward_terms(.019, (0.,0.), success=True)
        self.assertGreater(sum(toward.values()), 0.)
        self.assertLess(sum(away.values()), 0.)
        self.assertAlmostEqual(0., sum(still.values()))
        self.assertGreater(success["success_bonus"], 0.)
        self.assertGreater(self.env.config.progress_scale*.001, self.env.config.action_penalty_scale)


if __name__ == "__main__": unittest.main()
