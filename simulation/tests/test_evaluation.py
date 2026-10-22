import json
import tempfile
import unittest
from pathlib import Path

import torch

from simulation_core.evaluation import (
    DEFAULT_EVALUATION_SEEDS,
    DeterministicEvaluationRunner,
    EvaluationConfig,
    checkpoint_rank_key,
    wilson_confidence_interval,
)
from simulation_core.ppo import OBSERVATION_DIM, ObservationNormalizer, PPOTrainer


def observation():
    return {
        "arm_q": (0.0,) * 6,
        "arm_dq": (0.0,) * 6,
        "ee_to_object_xy": (0.0,) * 2,
        "object_to_goal_xy": (0.0,) * 2,
        "ee_velocity_xy": (0.0,) * 2,
        "object_velocity_xy": (0.0,) * 2,
        "target_tracking_error_xy": (0.0,) * 2,
        "previous_action_xy": (0.0,) * 2,
    }


class RecordingPolicy:
    def __init__(self):
        self.eval_called = False
        self.calls = []

    def eval(self):
        self.eval_called = True
        return self

    def act(self, value, deterministic=False):
        self.calls.append((deterministic, torch.is_grad_enabled()))
        return torch.zeros(2), torch.zeros(()), torch.zeros(())


class OutcomeEnvironment:
    def __init__(self, success):
        self.success = success
        self.closed = False

    def initialize(self):
        return self

    def reset(self):
        return observation()

    def step(self, action):
        distance = 0.01 if self.success else 0.5
        return observation(), 2.0 if self.success else -1.0, True, {
            "success": self.success,
            "timeout": not self.success,
            "distance_to_goal": distance,
            "initial_distance_to_goal": 1.0,
            "object_displacement": 0.25 if self.success else 0.0,
        }

    def close(self):
        self.closed = True


class EvaluationTestCase(unittest.TestCase):
    def evaluate(self, success_for_seed):
        with tempfile.TemporaryDirectory() as directory:
            policy = RecordingPolicy()
            runner = DeterministicEvaluationRunner(
                lambda seed: OutcomeEnvironment(success_for_seed(seed)),
                EvaluationConfig(Path(directory)),
            )
            summary = runner.run(
                policy=policy, normalizer=ObservationNormalizer(frozen=True)
            )
            episodes = [
                json.loads(line)
                for line in (
                    Path(directory) / "evaluation_episodes.jsonl"
                ).read_text().splitlines()
            ]
            saved_summary = json.loads(
                (Path(directory) / "evaluation_summary.json").read_text()
            )
            self.assertTrue(policy.eval_called)
            self.assertEqual(100, len(policy.calls))
            self.assertTrue(all(deterministic for deterministic, _ in policy.calls))
            self.assertTrue(all(not grad_enabled for _, grad_enabled in policy.calls))
            self.assertEqual(list(DEFAULT_EVALUATION_SEEDS), saved_summary["seeds"])
            self.assertEqual(summary["success_count"], saved_summary["success_count"])
            self.assertEqual(100, len(episodes))
            self.assertEqual(
                {
                    "seed", "success", "timeout", "final_distance",
                    "minimum_distance", "object_displacement", "return", "length",
                },
                set(episodes[0]),
            )
            return summary

    def test_all_success_wilson_interval(self):
        summary = self.evaluate(lambda seed: True)
        self.assertEqual(100, summary["success_count"])
        self.assertEqual(1.0, summary["success_rate"])
        interval = summary["confidence_interval_95"]
        self.assertAlmostEqual(0.9630065018, interval["lower"], places=8)
        self.assertEqual(1.0, interval["upper"])

    def test_all_fail_wilson_interval(self):
        summary = self.evaluate(lambda seed: False)
        self.assertEqual(0, summary["success_count"])
        self.assertEqual(0.0, summary["success_rate"])
        interval = summary["confidence_interval_95"]
        self.assertEqual(0.0, interval["lower"])
        self.assertAlmostEqual(0.0369934982, interval["upper"], places=8)

    def test_mixed_wilson_interval(self):
        summary = self.evaluate(lambda seed: seed < 50)
        self.assertEqual(50, summary["success_count"])
        self.assertEqual(0.5, summary["success_rate"])
        interval = summary["confidence_interval_95"]
        self.assertAlmostEqual(0.4038315304, interval["lower"], places=8)
        self.assertAlmostEqual(0.5961684696, interval["upper"], places=8)

    def test_config_rejects_fewer_than_one_hundred_seeds(self):
        with self.assertRaisesRegex(ValueError, "at least 100"):
            EvaluationConfig(Path("unused"), seeds=range(99))

    def test_checkpoint_ranking_prioritizes_evaluation_success_rate(self):
        high_reward_low_success = checkpoint_rank_key(
            {
                "success_rate": 0.2,
                "confidence_interval_95": {"lower": 0.1},
                "mean_final_distance": 0.1,
            },
            training_average_reward=100.0,
        )
        low_reward_high_success = checkpoint_rank_key(
            {
                "success_rate": 0.8,
                "confidence_interval_95": {"lower": 0.7},
                "mean_final_distance": 0.2,
            },
            training_average_reward=-100.0,
        )
        self.assertGreater(low_reward_high_success, high_reward_low_success)

    def test_checkpoint_evaluation_loads_and_bitwise_freezes_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = PPOTrainer(OutcomeEnvironment(True))
            for value in (torch.zeros(OBSERVATION_DIM), torch.ones(OBSERVATION_DIM)):
                trainer.normalizer.update(value)
            checkpoint = trainer.save_checkpoint(
                Path(directory) / "artifacts", "project-freeze", "run-freeze"
            )
            runner = DeterministicEvaluationRunner(
                lambda seed: OutcomeEnvironment(True),
                EvaluationConfig(Path(directory) / "evaluation"),
            )
            runner.run(checkpoint_path=checkpoint)
            loaded = runner._loaded_normalizer
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertTrue(loaded.frozen)
            self.assertEqual(payload["normalizer_state"]["count"], loaded.count)
            self.assertTrue(torch.equal(payload["normalizer_state"]["mean"], loaded.mean))
            self.assertTrue(torch.equal(payload["normalizer_state"]["m2"], loaded.m2))

    def test_evaluation_only_does_not_require_ppo_training_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = PPOTrainer(OutcomeEnvironment(True))
            checkpoint = trainer.save_checkpoint(
                Path(directory) / "artifacts", "project-old-math", "run-old-math"
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            for name in (
                "ppo_training_schema_version", "ppo_training_semantics",
                "ppo_training_config",
            ):
                payload.pop(name)
            torch.save(payload, checkpoint)
            runner = DeterministicEvaluationRunner(
                lambda seed: OutcomeEnvironment(True),
                EvaluationConfig(Path(directory) / "evaluation"),
            )
            summary = runner.run(checkpoint_path=checkpoint)
            self.assertEqual(100, summary["success_count"])

    def test_wilson_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            wilson_confidence_interval(2, 1)


if __name__ == "__main__":
    unittest.main()
