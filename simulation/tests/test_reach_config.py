import unittest

from simulation_core.domain import ReachTrainingTaskConfig


def config_values(**changes):
    values = dict(
        task_type="training", environment_type="s1_reach", seed=7,
        rollout_steps=200, num_updates=100, checkpoint_every=10,
        episode_steps=128, physics_steps_per_action=8, update_epochs=1,
        batch_size=16, device="cuda:0", run_mode="training",
    )
    values.update(changes); return values


class ReachConfigTests(unittest.TestCase):
    def test_production_profile_and_typed_target_config(self):
        config = ReachTrainingTaskConfig(**config_values())
        self.assertEqual((0.36, 0.0), config.target_center_xy_m)
        self.assertEqual((0.03, 0.08), config.target_half_extent_xy_m)
        self.assertEqual(0.005, config.action_delta_m)
        self.assertEqual(0.02, config.success_distance_m)
        self.assertEqual(3, config.success_hold_steps)

    def test_strict_fields_and_training_geometry(self):
        with self.assertRaises(TypeError): ReachTrainingTaskConfig(**config_values(extra=True))
        with self.assertRaises(ValueError): ReachTrainingTaskConfig(**config_values(episode_steps=256))
        with self.assertRaises(ValueError): ReachTrainingTaskConfig(**config_values(action_delta_m=.02))


if __name__ == "__main__": unittest.main()
