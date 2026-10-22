import tempfile
import unittest
from pathlib import Path

import torch

from simulation_core.experimental.manipulation.place_env import (
    PlaceEnvConfig, PlaceTransportEnvironment, advance_transport_success_event,
    transport_success_predicate,
)
from simulation_core.experimental.manipulation.place_ppo import ExperimentalPlacePPOTrainer
from simulation_core.experimental.manipulation.place_schema import (
    PLACE_ACTION_DIM, PLACE_ACTION_SCHEMA_ID, PLACE_ACTIVE_ACTION_MASK,
    PLACE_OBSERVATION_DIM, PLACE_OBSERVATION_SCHEMA_ID, PLACE_POLICY_SCHEMA,
    compute_place_reward,
)
from simulation_core.ppo.policy import flatten_observation
from simulation_core.ppo.trainer import PPOConfig


def snapshot():
    return {"q": [0., 1., 1., 0., 0., 0., .031, .031], "dq": [0.] * 8,
            "ee_state": [.458, -.009, .446, 0., 0., 0., 1.] + [0.] * 6,
            "cube_state": [.45, 0., .458, 0., 0., 0., 1.] + [0.] * 6,
            "estimated_aperture": .0521,
            "bodies": {name: {"force": [1., 0., 0.], "state": [0.] * 13}
                       for name in ("8_Link", "7_Link", "cube", "table", "ee_link")}}


class FakePlaceEnvironment:
    config = PlaceEnvConfig()
    target_center = (.45, .08, .458)
    target_selection = {"selection_reason": "test"}
    def reset(self):
        return {name: (0.,) * size for name, size in zip(
            PLACE_POLICY_SCHEMA.observation_fields, PLACE_POLICY_SCHEMA.observation_sizes)}
    def step(self, action):
        return self.reset(), 0., True, {"success": False, "timeout": True,
            "failure": False, "failure_reason": "TIMEOUT", "distance_to_goal": .08,
            "initial_distance_to_goal": .08, "action_norm": 0., "saturation_rate": 0.,
            "reward_terms": {}}


class PlaceContractTests(unittest.TestCase):
    def test_transport_success_helper_is_15mm_for_one_second(self):
        condition = transport_success_predicate(
            PlaceEnvConfig(), xy_error=.015, cube_rise=.060, grasp_valid=True,
            jaw_actual=.02, translation_drift=.005, rotation_drift=.10,
            finite=True, maximum_force=1.0, workspace_violation=False)
        self.assertTrue(all(condition.values()))
        self.assertIn("xy_within_15mm", condition)
        steps = 0
        for _ in range(59):
            steps, event = advance_transport_success_event(steps, condition, 60)
            self.assertFalse(event)
        steps, event = advance_transport_success_event(steps, condition, 60)
        self.assertTrue(event)

    def test_transport_success_helper_rejects_workspace_violation(self):
        condition = transport_success_predicate(
            PlaceEnvConfig(), xy_error=.010, cube_rise=.060, grasp_valid=True,
            jaw_actual=.02, translation_drift=.005, rotation_drift=.10,
            finite=True, maximum_force=1.0, workspace_violation=True)
        self.assertFalse(condition["workspace_valid"])
        self.assertFalse(all(condition.values()))

    def test_schema_is_32d_with_four_dimensional_masked_action(self):
        self.assertEqual("s1-place-observation/v1", PLACE_OBSERVATION_SCHEMA_ID)
        self.assertEqual("s1-place-action/v1", PLACE_ACTION_SCHEMA_ID)
        self.assertEqual(32, PLACE_OBSERVATION_DIM)
        self.assertEqual(4, PLACE_ACTION_DIM)
        self.assertEqual((1, 1, 0, 0), PLACE_ACTIVE_ACTION_MASK)

    def test_observation_is_finite_and_ordered(self):
        env = PlaceTransportEnvironment(); env._initial_cube = (.45, 0., .398)
        env._reference_relative = (-.008, .009, .012)
        env._reference_rotation = (0., 0., 0., 1.)
        env.target_center = (.45, .08, .458); env._place_previous_action = (0.,) * 4
        env.compositor._commanded_jaw = .012
        observation = env._place_observation(snapshot())
        vector = flatten_observation(observation, PLACE_POLICY_SCHEMA)
        self.assertEqual((32,), tuple(vector.shape)); self.assertTrue(torch.isfinite(vector).all())
        self.assertEqual(PLACE_POLICY_SCHEMA.observation_fields, tuple(observation))

    def test_target_selection_derives_safe_80mm_candidate(self):
        env = PlaceTransportEnvironment(); env._select_target(snapshot())
        self.assertAlmostEqual(.080, env.config.transport_distance)
        self.assertAlmostEqual(.080, env.target_center[1])
        self.assertTrue(env.target_selection["geometry_passed"])
        self.assertGreaterEqual(env.target_selection["base_clearance"], .25)

    def test_new_best_progress_cannot_be_reearned(self):
        config = PlaceEnvConfig().reward
        common = dict(config=config, in_target_zone=False, height_error=0.,
                      translation_drift=0., angular_speed=0., action=(0.,) * 4,
                      terminal_failure=False)
        first, _ = compute_place_reward(new_best_progress=.004, **common)
        repeat, _ = compute_place_reward(new_best_progress=0., **common)
        self.assertGreater(first, 0.); self.assertEqual(0., repeat)

    def test_snapshot_restores_nested_config_and_frozen_reward(self):
        from dataclasses import asdict
        from simulation_core.experimental.manipulation.place_env import place_config_from_snapshot, place_config_digest
        from simulation_core.reward_extension import starter_source, RewardExpression
        source = starter_source("s1_place")
        expression = RewardExpression("s1_place", source)
        config = PlaceEnvConfig(reward_extension={"schema": "reward-expression/v1", "source": source, "sha256": expression.digest})
        restored = place_config_from_snapshot(asdict(config))
        self.assertEqual(restored, config)
        self.assertTrue(callable(restored.workspace.clamp))
        self.assertEqual(place_config_digest(restored), place_config_digest(config))
        with self.assertRaises(ValueError):
            PlaceEnvConfig(reward_extension={"schema": "reward-expression/v1", "source": source, "sha256": "wrong"})

    def test_place_report_is_accepted_by_platform_evidence_validator(self):
        import json
        from simulation_core.experimental.manipulation.place_ppo import canonical_place_evaluation
        from simulation_core.evaluation_subprocess import validate_evaluation_outputs, evaluation_contract
        rows = [{"success": index != 99, "failure_reason": "SUCCESS" if index != 99 else "WORKSPACE_VIOLATION", "reward": 2.0, "steps": 80, "xy_error": .01, "minimum_xy_error": .005, "object_displacement": .07, "action_norm": .25, "action_saturation_rate": .1} for index in range(100)]
        summary, records = canonical_place_evaluation({"success_count": 99}, rows, "a" * 64)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "evaluation"; output.mkdir()
            (output / "summary.json").write_text(json.dumps(summary))
            (output / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            validated = validate_evaluation_outputs(Path(temp))
            self.assertEqual(evaluation_contract(validated)["status"], "passed")
            self.assertEqual(validated["failure_rate"], .01)
            self.assertEqual(validated["selected_success_seed"], 0)
            self.assertAlmostEqual(validated["mean_action_norm"], .25)

    def test_checkpoint_identity_and_active_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            ppo = PPOConfig(rollout_steps=2, update_epochs=1, batch_size=2)
            trainer = ExperimentalPlacePPOTrainer(FakePlaceEnvironment(), config=ppo)
            path = trainer.save_checkpoint(Path(temp), "p", "r")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual("s1_place", payload["task_id"])
            self.assertEqual(4, payload["action_dim"])
            self.assertEqual((1, 1, 0, 0), tuple(payload["active_action_mask"]))
            trainer.load_checkpoint(path)
            legacy = dict(payload)
            legacy.pop("ppo_training_schema_version", None)
            legacy_path = Path(temp) / "legacy.pt"; torch.save(legacy, legacy_path)
            trainer.load_checkpoint(legacy_path, evaluation_only=True)
            self.assertTrue(trainer.normalizer.frozen)
            with self.assertRaises(ValueError): trainer.load_checkpoint(legacy_path)
            payload["active_action_mask"] = (1, 1, 1, 0)
            bad = Path(temp) / "bad.pt"; torch.save(payload, bad)
            with self.assertRaises(ValueError): trainer.load_checkpoint(bad)
            with self.assertRaises(ValueError): trainer.load_checkpoint(bad, evaluation_only=True)


if __name__ == "__main__": unittest.main()
