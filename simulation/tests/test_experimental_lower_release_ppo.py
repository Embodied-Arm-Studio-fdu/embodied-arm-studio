import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from simulation_core.experimental.manipulation.lower_release_env import (
    LOWER_RELEASE_TARGET, LowerReleaseEnvConfig, LowerReleaseEnvironment,
)
from simulation_core.experimental.manipulation.lower_release_ppo import (
    ExperimentalLowerReleasePPOTrainer,
)
from simulation_core.experimental.manipulation.lower_release_schema import (
    LOWER_RELEASE_ACTIVE_ACTION_MASK, LOWER_RELEASE_POLICY_SCHEMA,
    LowerReleaseRewardConfig, compute_lower_release_reward,
)
from simulation_core.experimental.manipulation.place_schema import (
    PLACE_ACTION_SCHEMA_ID, PLACE_OBSERVATION_SCHEMA_ID,
)
from simulation_core.experimental.manipulation.staged_place_env import (
    StagedFullPlaceEnvironment,
)
from simulation_core.ppo.trainer import PPOConfig


class FakeLowerReleaseEnvironment:
    config = LowerReleaseEnvConfig()
    target_center = LOWER_RELEASE_TARGET
    def reset(self):
        return {name: (0.0,) * size for name, size in zip(
            LOWER_RELEASE_POLICY_SCHEMA.observation_fields,
            LOWER_RELEASE_POLICY_SCHEMA.observation_sizes)}
    def step(self, action):
        return self.reset(), 0.0, True, {
            "success": False, "timeout": True, "failure": False,
            "failure_reason": "TIMEOUT", "distance_to_goal": .06,
            "initial_distance_to_goal": .06, "action_norm": 0.0,
            "saturation_rate": 0.0, "object_displacement": 0.0,
            "reward_terms": {},
        }


class LowerReleaseContractTests(unittest.TestCase):
    def test_staged_reset_runs_discarded_priming_then_visible_reset(self):
        env = object.__new__(StagedFullPlaceEnvironment)
        with patch.object(
                StagedFullPlaceEnvironment, "_reset_once",
                side_effect=("discarded", "visible")) as reset_once:
            self.assertEqual("visible", env.reset())
        self.assertEqual(2, reset_once.call_count)

    def test_contract_reuses_32d_4d_ids_with_p4_2_mask(self):
        self.assertEqual("s1-place-observation/v1", PLACE_OBSERVATION_SCHEMA_ID)
        self.assertEqual("s1-place-action/v1", PLACE_ACTION_SCHEMA_ID)
        self.assertEqual(32, LOWER_RELEASE_POLICY_SCHEMA.observation_dim)
        self.assertEqual(4, LOWER_RELEASE_POLICY_SCHEMA.action_dim)
        self.assertEqual((0, 0, 1, 1), LOWER_RELEASE_ACTIVE_ACTION_MASK)
        self.assertEqual((.450, .080, .458), LOWER_RELEASE_TARGET)

    def test_release_trigger_requires_two_consecutive_samples_and_legal_height(self):
        env = LowerReleaseEnvironment()
        self.assertFalse(env._apply_release_trigger(0.7, .008))
        self.assertFalse(env._release_latched)
        self.assertTrue(env._apply_release_trigger(0.7, .008))
        self.assertTrue(env._release_started)
        self.assertTrue(env._release_latched)

        high = LowerReleaseEnvironment()
        self.assertFalse(high._apply_release_trigger(0.7, .020))
        self.assertTrue(high._apply_release_trigger(0.7, .020))
        self.assertTrue(high._release_started)
        self.assertFalse(high._release_latched)

        reset = LowerReleaseEnvironment()
        self.assertFalse(reset._apply_release_trigger(0.7, .008))
        self.assertFalse(reset._apply_release_trigger(-0.1, .008))
        self.assertFalse(reset._apply_release_trigger(0.7, .008))
        self.assertFalse(reset._release_started)

    def test_opening_credit_is_incremental_and_height_gated(self):
        common = dict(config=LowerReleaseRewardConfig(), downward_progress=0.0,
                      premature_open_progress=0.0, in_target_zone=True,
                      table_contact=False, settle_valid=False, success_event=False,
                      action=(0.0, 0.0, 0.0, 0.5), terminal_failure=False)
        first, first_terms = compute_lower_release_reward(
            legal_open_progress=.25, **common)
        repeat, repeat_terms = compute_lower_release_reward(
            legal_open_progress=0.0, **common)
        self.assertGreater(first_terms["release_readiness"], 0.0)
        self.assertEqual(0.0, repeat_terms["release_readiness"])
        self.assertGreater(first, repeat)
        premature, terms = compute_lower_release_reward(
            config=LowerReleaseRewardConfig(), downward_progress=0.0,
            legal_open_progress=0.0, premature_open_progress=.25,
            in_target_zone=True, table_contact=False, settle_valid=False,
            success_event=False, action=(0.0, 0.0, 0.0, .5),
            terminal_failure=False)
        self.assertLess(terms["release_readiness"], 0.0)
        self.assertLess(premature, repeat)

    def test_checkpoint_identity_mask_and_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            ppo = PPOConfig(rollout_steps=2, update_epochs=1, batch_size=2)
            trainer = ExperimentalLowerReleasePPOTrainer(
                FakeLowerReleaseEnvironment(), config=ppo)
            trainer.parent_checkpoint = {"sha256": "test"}
            path = trainer.save_checkpoint(Path(temporary), "p", "r")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual("s1_place", payload["task_id"])
            self.assertEqual((0, 0, 1, 1), tuple(payload["active_action_mask"]))
            self.assertEqual("policy_trigger_latched_open", payload["release_semantics"])
            trainer.load_checkpoint(path)
            payload["active_action_mask"] = (0, 0, 1, 0)
            bad = Path(temporary) / "bad.pt"; torch.save(payload, bad)
            with self.assertRaises(ValueError): trainer.load_checkpoint(bad)


if __name__ == "__main__": unittest.main()
