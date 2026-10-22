import tempfile
import unittest
from pathlib import Path

import torch

from simulation_core.experimental.manipulation.grasp_env import (
    CURRICULUM_STAGE_A, GraspEnvConfig,
)
from simulation_core.experimental.manipulation.grasp_ppo import ExperimentalGraspPPOTrainer
from simulation_core.experimental.manipulation.grasp_schema import (
    GRASP_POLICY_SCHEMA, GRASP_REWARD_VERSION, GraspRewardConfig,
    closure_reward_potentials, compute_grasp_reward,
)
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document
from simulation_core.task_registry import PRODUCTION_TASK_REGISTRY


def potentials(command=.035, actual=.035, **overrides):
    kwargs = dict(open_safe=.05, close_safe=.025, commanded_jaw=command,
        actual_jaw=actual, finite=True, in_corridor=True, xy_aligned=True,
        at_grasp_height=True, no_premature_push=True, safe_contact=True,
        bilateral_contact=True, cube_contact=True)
    kwargs.update(overrides)
    return closure_reward_potentials(**kwargs)


def shaping(close_delta=0., blockage_delta=0.):
    total, terms = compute_grasp_reward(
        config=GraspRewardConfig(), approach_progress=0., alignment_progress=0.,
        descend_progress=0., xy_aligned=True,
        closure_potential_progress=close_delta,
        blockage_potential_progress=blockage_delta,
        candidate_event=False, verified_event=False,
        action=(0., 0., 0., 0.), saturated=0., clamped=False, failure=False)
    return total, terms


class FakeEnvironment:
    config = GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_A)
    def reset(self):
        return {name: (0.,) * size for name, size in zip(
            GRASP_POLICY_SCHEMA.observation_fields, GRASP_POLICY_SCHEMA.observation_sizes)}
    def step(self, action):
        return self.reset(), 0., True, {"success": False, "timeout": True,
            "failure": True, "failure_reason": "TIMEOUT", "distance_to_goal": 1.,
            "initial_distance_to_goal": 1., "action_norm": 0.,
            "saturation_rate": 0., "reward_terms": {}}


class GraspRewardV3Tests(unittest.TestCase):
    def test_version_and_weights(self):
        cfg = GraspRewardConfig()
        self.assertEqual("s1-grasp-reward/v3", GRASP_REWARD_VERSION)
        self.assertEqual(2., cfg.closure_progress)
        self.assertEqual(3., cfg.blockage_progress)
        self.assertEqual(5., cfg.candidate_bonus)
        self.assertEqual(20., cfg.verified_bonus)

    def test_far_air_close_and_wrong_height_are_gated(self):
        close, block, flags = potentials(in_corridor=False, bilateral_contact=False,
                                         cube_contact=False)
        self.assertEqual((0., 0.), (close, block))
        self.assertFalse(flags["closure_gate"])
        close, block, _ = potentials(at_grasp_height=False)
        self.assertEqual((0., 0.), (close, block))

    def test_table_unsafe_or_pushed_cube_is_gated(self):
        self.assertEqual((0., 0.), potentials(safe_contact=False)[:2])
        self.assertEqual((0., 0.), potentials(no_premature_push=False)[:2])

    def test_correct_pose_close_progress_is_positive(self):
        fractions = [potentials(command=value)[0]
                     for value in (.05, .045, .04, .035, .03, .025)]
        rewards = [shaping(fractions[i] - fractions[i-1], 0.)[1]["closure_progress"]
                   for i in range(1, len(fractions))]
        self.assertTrue(all(value > 0 for value in rewards))
        self.assertAlmostEqual(2., sum(rewards))

    def test_hover_has_zero_progress(self):
        total, terms = shaping(0., 0.)
        self.assertEqual(0., terms["closure_progress"])
        self.assertEqual(0., terms["blockage_progress"])
        self.assertEqual(0., total)

    def test_reopen_is_negative_and_cycle_not_positive(self):
        close_035 = potentials(command=.035)[0]
        close_040 = potentials(command=.040)[0]
        opening = shaping(close_040 - close_035, 0.)[0]
        reclosing = shaping(close_035 - close_040, 0.)[0]
        self.assertLess(opening, 0.)
        self.assertAlmostEqual(0., opening + reclosing)

    def test_blockage_requires_bilateral_cube_contact(self):
        self.assertEqual(0., potentials(command=.03, actual=.032,
                                        bilateral_contact=False)[1])
        self.assertEqual(0., potentials(command=.03, actual=.032,
                                        cube_contact=False)[1])

    def test_blockage_progress_is_continuous_and_hover_zero(self):
        gaps = (0., .0005, .001, .0015, .002)
        values = [potentials(command=.03, actual=.03 + gap)[1] for gap in gaps]
        rewards = [shaping(0., values[i] - values[i-1])[1]["blockage_progress"]
                   for i in range(1, len(values))]
        for reward in rewards:
            self.assertAlmostEqual(.75, reward)
        self.assertEqual(0., shaping(0., values[-1] - values[-1])[1]["blockage_progress"])

    def test_blockage_noise_cycle_has_zero_net_shaping(self):
        values = [.5, .75, .5, .75, .5]
        total = sum(shaping(0., right-left)[1]["blockage_progress"]
                    for left, right in zip(values, values[1:]))
        self.assertAlmostEqual(0., total)

    def test_v2_checkpoint_is_rejected_by_v3(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = ExperimentalGraspPPOTrainer(
                FakeEnvironment(), config=PPOConfig(rollout_steps=2, update_epochs=1, batch_size=2))
            path = trainer.save_checkpoint(Path(temp), "p", "r")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["reward_version"] = "s1-grasp-reward/v2"
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                trainer.load_checkpoint(path)

    def test_production_isolation(self):
        self.assertIn("s1_place", [d.id for d in PRODUCTION_TASK_REGISTRY.definitions()])
        document = task_catalog_document()
        self.assertEqual("task-catalog/v2", document["catalog_version"])


if __name__ == "__main__":
    unittest.main()
