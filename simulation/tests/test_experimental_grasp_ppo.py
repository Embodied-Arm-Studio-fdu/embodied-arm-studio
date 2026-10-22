import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import hashlib

import torch

from simulation_core.experimental.manipulation.grasp_env import GraspEnvConfig, GraspPPOEnvironment
from simulation_core.experimental.manipulation.grasp_ppo import (
    ExperimentalGraspPPOTrainer, GRASP_PPO_CONTRACT,
)
from simulation_core.experimental.manipulation.grasp_schema import (
    GRASP_ACTION_DIM, GRASP_ACTION_SCHEMA_ID, GRASP_OBSERVATION_DIM,
    GRASP_OBSERVATION_FIELDS, GRASP_OBSERVATION_SCHEMA_ID, GRASP_POLICY_SCHEMA,
    compute_grasp_reward,
)
from simulation_core.ppo.policy import PUSH_POLICY_SCHEMA, REACH_POLICY_SCHEMA, flatten_observation
from simulation_core.ppo.trainer import (
    PPOTaskContract, PPOConfig, validate_checkpoint_action_schema,
    validate_checkpoint_observation_schema,
)


def snapshot():
    return {
        "q": [0., 1.0, 1.0, 0., 0., 0., .05, .05], "dq": [0.] * 8,
        "ee_state": [.391, -.082, .39, 0., 0., 0., 1.] + [0.] * 6,
        "cube_state": [.45, 0., .398, 0., 0., 0., 1.] + [0.] * 6,
        "estimated_aperture": .0901,
        "bodies": {name: {"force": [0., 0., 0.], "state": [0.] * 13}
                   for name in ("8_Link", "7_Link", "cube", "table", "ee_link")},
    }


class FakeGraspEnvironment:
    config = GraspEnvConfig()
    def reset(self):
        return {name: (0.,) * size for name, size in zip(
            GRASP_POLICY_SCHEMA.observation_fields, GRASP_POLICY_SCHEMA.observation_sizes)}
    def step(self, action):
        return self.reset(), 0.0, True, {"success": False, "timeout": True,
            "failure": True, "failure_reason": "TIMEOUT", "distance_to_goal": 1.0,
            "initial_distance_to_goal": 1.0, "action_norm": 0.0,
            "saturation_rate": 0.0, "reward_terms": {}}


class ExperimentalGraspPPOTests(unittest.TestCase):
    def test_schema_is_versioned_40d_and_4d(self):
        self.assertEqual("s1-grasp-observation/v1", GRASP_OBSERVATION_SCHEMA_ID)
        self.assertEqual("s1-grasp-action/v1", GRASP_ACTION_SCHEMA_ID)
        self.assertEqual(40, GRASP_OBSERVATION_DIM)
        self.assertEqual(4, GRASP_ACTION_DIM)
        self.assertEqual(12, len(GRASP_OBSERVATION_FIELDS))

    def test_fixed_normalized_observation_is_finite_and_ordered(self):
        env = GraspPPOEnvironment()
        env._initial_cube = (.45, 0., .398); env._initial_ee = (.391, -.082, .39)
        env._previous_action = (0.,) * 4
        observation = env._observation(snapshot())
        vector = flatten_observation(observation, GRASP_POLICY_SCHEMA)
        self.assertEqual((40,), tuple(vector.shape)); self.assertTrue(torch.isfinite(vector).all())
        self.assertEqual(GRASP_OBSERVATION_FIELDS, tuple(observation))

    def test_action_mapping_is_delta_xyz_and_absolute_grip(self):
        env = GraspPPOEnvironment(); env._target = (.40, 0., .40)
        values, xyz, jaw = env._map_action((1., -1., 1., -1.))
        self.assertEqual((.405, -.005, .4025), xyz)
        self.assertAlmostEqual(.025, jaw)
        _, _, open_jaw = env._map_action((0., 0., 0., 1.))
        self.assertAlmostEqual(.05, open_jaw)

    def test_reward_gates_closure_and_signed_progress_prevents_oscillation(self):
        cfg = GraspEnvConfig().reward
        common = dict(config=cfg, alignment_progress=0., descend_progress=0.,
            xy_aligned=False, candidate_event=False, verified_event=False,
            action=(0.,0.,0.,0.), saturated=0., clamped=False, failure=False)
        far, far_terms = compute_grasp_reward(
            approach_progress=0., closure_potential_progress=0.,
            blockage_potential_progress=0., **common)
        self.assertEqual(0., far_terms["closure_progress"]); self.assertEqual(0., far)
        forward, _ = compute_grasp_reward(
            approach_progress=.01, closure_potential_progress=0.,
            blockage_potential_progress=0., **common)
        backward, _ = compute_grasp_reward(
            approach_progress=-.01, closure_potential_progress=0.,
            blockage_potential_progress=0., **common)
        self.assertAlmostEqual(0., forward + backward)

    def test_checkpoint_identity_and_cross_load_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            config = PPOConfig(rollout_steps=2, update_epochs=1, batch_size=2)
            trainer = ExperimentalGraspPPOTrainer(FakeGraspEnvironment(), config=config)
            path = trainer.save_checkpoint(Path(temp), "p", "r")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual("s1_grasp", payload["task_id"])
            self.assertEqual(40, payload["observation_dim"]); self.assertEqual(4, payload["action_dim"])
            self.assertEqual(GRASP_OBSERVATION_SCHEMA_ID, payload["observation_schema_id"])
            trainer.load_checkpoint(path)
            with self.assertRaises(ValueError): validate_checkpoint_action_schema(payload)
            reach = PPOTaskContract(policy_schema=REACH_POLICY_SCHEMA)
            with self.assertRaises(ValueError): validate_checkpoint_observation_schema(payload, reach)
            push_like = dict(payload, task_id="s1_push", action_dim=2,
                             action_semantics=PUSH_POLICY_SCHEMA.action_semantics)
            with self.assertRaises(ValueError): validate_checkpoint_action_schema(
                push_like, GRASP_PPO_CONTRACT)

    def test_p2_r8_transfer_resets_training_and_preserves_verified_best(self):
        with tempfile.TemporaryDirectory() as temp:
            config = PPOConfig(rollout_steps=2, update_epochs=1, batch_size=2)
            source = FakeGraspEnvironment()
            trainer = ExperimentalGraspPPOTrainer(source, config=config)
            path = trainer.save_checkpoint(Path(temp), "p", "r")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            target = FakeGraspEnvironment()
            target.config = replace(
                target.config, randomize_xy=.006, random_seed=7,
            )
            transferred = ExperimentalGraspPPOTrainer(target, config=config)
            transferred.initialize_from_p2_r8(path, digest)
            self.assertEqual(0, transferred.training_steps)
            self.assertEqual(0, transferred.update_steps)
            self.assertEqual(digest, transferred.curriculum_parent_checkpoint_sha256)
            self.assertEqual(
                (1.0, 0.98, 0.92999, 0.0, 0.0),
                transferred.initial_best_checkpoint_rank,
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                transferred.initialize_from_p2_r8(path, "0" * 64)
            incompatible = FakeGraspEnvironment()
            incompatible.config = replace(incompatible.config, xyz_delta_scale=.0025)
            rejected = ExperimentalGraspPPOTrainer(incompatible, config=config)
            with self.assertRaisesRegex(ValueError, "XY randomization/seed"):
                rejected.initialize_from_p2_r8(path, digest)

    def test_horizon_and_p1_physics_baseline_are_locked(self):
        config = GraspEnvConfig()
        self.assertEqual(120, config.learning_horizon)
        self.assertEqual(4, config.physics_steps_per_action)
        self.assertEqual(.05, config.open_safe); self.assertEqual(.025, config.close_safe)
        self.assertEqual(.001, config.jaw_rate_limit); self.assertEqual(50., config.force_abort)
        with self.assertRaises(ValueError): GraspEnvConfig(randomize_xy=.02)


if __name__ == "__main__": unittest.main()
