import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import hashlib

import torch

from simulation_core.experimental.manipulation.lift_env import LiftEnvConfig, LiftPPOEnvironment
from simulation_core.experimental.manipulation.lift_ppo import (
    ExperimentalLiftPPOTrainer, LIFT_PPO_CONTRACT,
)
from simulation_core.experimental.manipulation.lift_schema import (
    LIFT_ACTION_DIM, LIFT_ACTION_SCHEMA_ID, LIFT_OBSERVATION_DIM,
    LIFT_OBSERVATION_SCHEMA_ID, LIFT_POLICY_SCHEMA, compute_lift_reward,
)
from simulation_core.ppo.policy import PUSH_POLICY_SCHEMA, REACH_POLICY_SCHEMA, flatten_observation
from simulation_core.ppo.trainer import PPOConfig, PPOTaskContract, validate_checkpoint_action_schema


def snapshot():
    return {
        "q": [0., 1., 1., 0., 0., 0., .033, .033], "dq": [0.] * 8,
        "ee_state": [.458, -.009, .386, 0., 0., 0., 1.] + [0.] * 6,
        "cube_state": [.45, 0., .398, 0., 0., 0., 1.] + [0.] * 6,
        "estimated_aperture": .0561,
        "bodies": {name: {"force": [1., 0., 0.], "state": [0.] * 13}
                   for name in ("8_Link", "7_Link", "cube", "table", "ee_link")},
    }


class FakeLiftEnvironment:
    config = LiftEnvConfig()
    def reset(self):
        return {name: (0.,) * size for name, size in zip(
            LIFT_POLICY_SCHEMA.observation_fields, LIFT_POLICY_SCHEMA.observation_sizes)}
    def step(self, action):
        return self.reset(), 0.0, True, {
            "success": False, "timeout": True, "failure": False,
            "failure_reason": "TIMEOUT", "distance_to_goal": .06,
            "initial_distance_to_goal": .06, "action_norm": 0.,
            "saturation_rate": 0., "reward_terms": {}, "cube_rise": 0.,
        }


class ExperimentalLiftTests(unittest.TestCase):
    def test_schema_is_versioned_18d_and_action_only_dz(self):
        self.assertEqual("s1-lift-observation/v1", LIFT_OBSERVATION_SCHEMA_ID)
        self.assertEqual("s1-lift-action/v1", LIFT_ACTION_SCHEMA_ID)
        self.assertEqual(18, LIFT_OBSERVATION_DIM)
        self.assertEqual(1, LIFT_ACTION_DIM)
        self.assertIn("fixed_xy_orientation_and_jaw", LIFT_POLICY_SCHEMA.action_semantics)

    def test_observation_is_finite_and_ordered(self):
        env = LiftPPOEnvironment(); env._initial_cube = (.45, 0., .398)
        env._reference_relative = (-.008, .009, .012)
        env._reference_rotation = (0., 0., 0., 1.); env._previous_action = (0.,)
        env.compositor._commanded_jaw = .012
        observation = env._observation(snapshot())
        vector = flatten_observation(observation, LIFT_POLICY_SCHEMA)
        self.assertEqual((18,), tuple(vector.shape)); self.assertTrue(torch.isfinite(vector).all())
        self.assertEqual(LIFT_POLICY_SCHEMA.observation_fields, tuple(observation))

    def test_reward_progress_cannot_be_reearned_by_oscillation(self):
        cfg = LiftEnvConfig().reward
        common = dict(config=cfg, in_target_band=False, hold_valid=False,
                      translation_drift=0., cube_angular_speed=0.,
                      cube_vertical_speed=0., action=(0.,), terminal_failure=False)
        up, _ = compute_lift_reward(new_max_progress=.003, **common)
        down, _ = compute_lift_reward(new_max_progress=0., **common)
        reup, _ = compute_lift_reward(new_max_progress=0., **common)
        self.assertGreater(up, 0.); self.assertEqual(0., down + reup)

    def test_target_hold_and_horizon_are_locked(self):
        config = LiftEnvConfig()
        self.assertEqual(.060, config.target_lift)
        self.assertEqual((.055, .065), config.target_band)
        self.assertEqual(1., config.hold_seconds)
        self.assertEqual(.003, config.dz_scale)
        long_hold = replace(config, hold_seconds=5.0, max_episode_seconds=10.0)
        self.assertEqual(5.0, long_hold.hold_seconds)
        with self.assertRaises(ValueError):
            replace(config, hold_seconds=5.0, max_episode_seconds=8.0)

    def test_p3_1_checkpoint_loads_only_as_hold_horizon_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            ppo = PPOConfig(rollout_steps=2, update_epochs=1, batch_size=2)
            source = FakeLiftEnvironment()
            trainer = ExperimentalLiftPPOTrainer(source, config=ppo)
            path = trainer.save_checkpoint(Path(temp), "p", "r")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            target = FakeLiftEnvironment()
            target.config = replace(target.config, hold_seconds=5.0,
                                    max_episode_seconds=10.0)
            replayer = ExperimentalLiftPPOTrainer(target, config=ppo)
            replayer.load_hold_extension_checkpoint(path, digest)
            with self.assertRaises(ValueError):
                replayer.load_checkpoint(path)
            target.config = replace(target.config, dz_scale=.002)
            incompatible = ExperimentalLiftPPOTrainer(target, config=ppo)
            with self.assertRaises(ValueError):
                incompatible.load_hold_extension_checkpoint(path, digest)

    def test_checkpoint_identity_rejects_push_reach_and_grasp(self):
        with tempfile.TemporaryDirectory() as temp:
            ppo = PPOConfig(rollout_steps=2, update_epochs=1, batch_size=2)
            trainer = ExperimentalLiftPPOTrainer(FakeLiftEnvironment(), config=ppo)
            path = trainer.save_checkpoint(Path(temp), "p", "r")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual("s1_lift", payload["task_id"])
            self.assertEqual(1, payload["action_dim"])
            trainer.load_checkpoint(path)
            with self.assertRaises(ValueError): validate_checkpoint_action_schema(payload)
            with self.assertRaises(ValueError): validate_checkpoint_action_schema(
                payload, PPOTaskContract(policy_schema=REACH_POLICY_SCHEMA))
            grasp = dict(payload, task_id="s1_grasp", action_dim=4,
                         action_semantics="normalized_grasp")
            with self.assertRaises(ValueError): validate_checkpoint_action_schema(
                grasp, LIFT_PPO_CONTRACT)


if __name__ == "__main__": unittest.main()
