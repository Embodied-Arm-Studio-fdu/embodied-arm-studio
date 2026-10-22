import unittest
import hashlib

from simulation_core.reward_extension import (
    PROTECTED_TERMS, TASK_TERMS, RewardExpression, starter_source,
)
from simulation_core.domain import TrainingTaskConfig


class RewardExpressionTests(unittest.TestCase):
    def test_starter_preserves_baseline_and_protected_terms(self):
        for task_id, names in TASK_TERMS.items():
            terms = {name: 0.2 for name in names}
            terms.update({"success_bonus": 2.0, "joint_limit_penalty": -0.1})
            self.assertAlmostEqual(RewardExpression(task_id, starter_source(task_id)).evaluate(terms), sum(terms.values()))

    def test_custom_expression_cannot_remove_safety_terms(self):
        expression = RewardExpression("s1_reach", "def combine_reward(terms):\n    return 0\n")
        self.assertEqual(expression.evaluate({"success_bonus": 2, "joint_limit_penalty": -1}), 1)

    def test_rejects_executable_python_and_protected_access(self):
        for source in (
            "import os\ndef combine_reward(terms):\n    return 0",
            "def combine_reward(terms):\n    return __import__('os').system('true')",
            "def combine_reward(terms):\n    return terms.__class__",
            "def combine_reward(terms):\n    return terms['success_bonus']",
            "def combine_reward(terms):\n    return 100000000 ** 100000000",
            "def combine_reward(terms):\n    return [x for x in terms]",
            "def combine_reward(terms):\n    return 11 * terms['progress_reward']",
        ):
            with self.assertRaises(ValueError):
                RewardExpression("s1_reach", source)

    def test_runtime_rejects_invalid_numeric_results(self):
        for formula in ("1 / 0", "terms['progress_reward']", "10 * terms['progress_reward']"):
            expression = RewardExpression("s1_reach", "def combine_reward(terms):\n    return " + formula)
            with self.assertRaises(ValueError):
                expression.evaluate({"progress_reward": float('inf'), **{name: 0 for name in PROTECTED_TERMS}})

    def test_training_config_serializes_the_frozen_extension(self):
        source = starter_source("s1_push")
        extension = {"schema": "reward-expression/v1", "source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}
        config = TrainingTaskConfig(
            task_type="training", environment_type="s1_push", seed=7,
            rollout_steps=16, num_updates=2, checkpoint_every=1,
            episode_steps=32, physics_steps_per_action=2,
            reward_extension=extension,
        )
        self.assertEqual(config.to_dict()["reward_extension"], extension)


if __name__ == "__main__":
    unittest.main()