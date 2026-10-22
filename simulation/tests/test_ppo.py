import tempfile
import unittest
import json
from pathlib import Path

import torch

from simulation_core.ppo import (
    ACTION_DIM,
    ACTION_SEMANTICS,
    OBSERVATION_DIM,
    OBSERVATION_FIELDS,
    OBSERVATION_SLICES,
    PUSH_OBSERVATION_SCHEMA_VERSION,
    PUSH_OBSERVATION_DIM,
    PUSH_OBSERVATION_SEMANTICS,
    MLPActorCritic,
    ObservationNormalizer,
    PPOConfig,
    PPOTrainer,
    PPOTrainingConfig,
    RolloutBuffer,
    TrainingCancelled,
    flatten_observation,
)
from simulation_core.ppo.trainer import (
    normalize_advantages,
    ppo_clipped_surrogate,
)


def observation(value=0.0):
    return {
        "arm_q": (value,) * 6,
        "arm_dq": (value,) * 6,
        "ee_to_object_xy": (value,) * 2,
        "object_to_goal_xy": (value,) * 2,
        "ee_velocity_xy": (value,) * 2,
        "object_velocity_xy": (value,) * 2,
        "target_tracking_error_xy": (value,) * 2,
        "previous_action_xy": (value,) * 2,
    }


class FakeEnvironment:
    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0
        return observation()

    def step(self, action):
        self.steps += 1
        done = self.steps >= 5
        distance = 1.0 - self.steps * 0.1
        return observation(self.steps * 0.001), 0.01, done, {
            "success": done,
            "timeout": False,
            "failure": False,
            "distance_to_goal": distance,
            "initial_distance_to_goal": 1.0,
            "object_displacement": self.steps * 0.1,
            "action_norm": 0.25,
            "saturation_rate": 0.2,
        }


class PPOTestCase(unittest.TestCase):
    def test_policy_output_shape_and_bounds(self):
        policy = MLPActorCritic()
        action, log_prob, value = policy.act(torch.zeros(OBSERVATION_DIM))
        self.assertEqual((ACTION_DIM,), tuple(action.shape))
        self.assertEqual((), tuple(log_prob.shape))
        self.assertEqual((), tuple(value.shape))
        self.assertTrue(torch.all(action <= policy.action_bias + policy.action_scale))
        self.assertTrue(torch.all(action >= policy.action_bias - policy.action_scale))
        self.assertEqual((2,), tuple(policy.log_std.shape))
        self.assertTrue(torch.allclose(policy.action_scale, torch.tensor((0.005, 0.005))))
        self.assertTrue(torch.equal(policy.action_bias, torch.zeros(2)))

    def test_observation_vector_has_stable_shape(self):
        vector = flatten_observation(observation())
        self.assertEqual(24, OBSERVATION_DIM)
        self.assertEqual(PUSH_OBSERVATION_DIM, OBSERVATION_DIM)
        self.assertEqual((OBSERVATION_DIM,), tuple(vector.shape))
        self.assertEqual(tuple(observation()), OBSERVATION_FIELDS)
        self.assertEqual(slice(0, 6), OBSERVATION_SLICES["arm_q"])
        self.assertEqual(slice(22, 24), OBSERVATION_SLICES["previous_action_xy"])
        with self.assertRaisesRegex(ValueError, "missing field"):
            flatten_observation({})

    def test_buffer_stores_and_samples(self):
        buffer = RolloutBuffer()
        for _ in range(4):
            buffer.add(
                torch.zeros(3), torch.zeros(2), 1.0, False,
                torch.tensor(0.5), torch.tensor(-0.1), torch.tensor(0.5), 1.0, 1.0,
            )
        buffer.compute_returns_and_advantages(0.99, 0.95)
        batches = list(buffer.batches(2))
        self.assertEqual(4, len(buffer))
        self.assertEqual(2, len(batches))
        self.assertEqual((2, 3), tuple(batches[0]["observations"].shape))
        self.assertEqual((2, 2), tuple(batches[0]["actions"].shape))
        with self.assertRaisesRegex(ValueError, "shape \(2,\)"):
            buffer.add(
                torch.zeros(3), torch.zeros(6), 1.0, False,
                torch.tensor(0.5), torch.tensor(-0.1), torch.tensor(0.5), 1.0, 1.0,
            )

    def test_hand_computed_gae_no_terminal(self):
        buffer = RolloutBuffer()
        for reward, value, next_value in zip((1.0, 1.0, 1.0), (0.5, 0.6, 0.7), (0.6, 0.7, 0.8)):
            buffer.add(torch.zeros(3), torch.zeros(2), reward, False, torch.tensor(value), torch.tensor(0.0), torch.tensor(next_value), 1.0, 1.0)
        buffer.compute_returns_and_advantages(0.9, 0.8)
        self.assertTrue(torch.allclose(buffer.advantages, torch.tensor((2.310368, 1.7644, 1.02)), atol=1e-6))
        self.assertTrue(torch.allclose(buffer.returns, torch.tensor((2.810368, 2.3644, 1.72)), atol=1e-6))

    def test_success_and_failure_do_not_bootstrap(self):
        for outcome in ("success", "failure"):
            with self.subTest(outcome=outcome):
                buffer = RolloutBuffer()
                buffer.add(torch.zeros(3), torch.zeros(2), 0.3, True, torch.tensor(0.7), torch.tensor(0.0), torch.tensor(99.0), 0.0, 0.0)
                buffer.compute_returns_and_advantages(0.9, 0.8)
                self.assertAlmostEqual(-0.4, float(buffer.advantages[0]), places=6)
                self.assertAlmostEqual(0.3, float(buffer.returns[0]), places=6)

    def test_timeout_bootstraps_but_does_not_cross_reset_episode(self):
        buffer = RolloutBuffer()
        buffer.add(torch.zeros(3), torch.zeros(2), 0.0, True, torch.tensor(1.0), torch.tensor(0.0), torch.tensor(10.0), 1.0, 0.0)
        buffer.add(torch.zeros(3), torch.zeros(2), 0.0, False, torch.tensor(100.0), torch.tensor(0.0), torch.tensor(100.0), 1.0, 1.0)
        buffer.compute_returns_and_advantages(0.9, 0.8)
        self.assertAlmostEqual(8.0, float(buffer.advantages[0]), places=6)
        self.assertAlmostEqual(-10.0, float(buffer.advantages[1]), places=6)
        self.assertAlmostEqual(9.0, float(buffer.returns[0]), places=6)

    def test_advantage_normalization_and_constant_input_are_finite(self):
        normalized, mean, std = normalize_advantages(torch.tensor((1.0, 2.0, 3.0)))
        self.assertAlmostEqual(0.0, float(normalized.mean()), places=6)
        self.assertAlmostEqual(1.0, float(normalized.std(unbiased=False)), places=6)
        constant, _, constant_std = normalize_advantages(torch.ones(4))
        self.assertEqual(0.0, float(constant_std))
        self.assertTrue(torch.isfinite(constant).all())
        self.assertTrue(torch.equal(constant, torch.zeros(4)))

    def test_same_policy_log_prob_ratio_and_physical_scaling(self):
        torch.manual_seed(4)
        policy = MLPActorCritic()
        observations = torch.zeros((8, OBSERVATION_DIM))
        actions, old_log_prob, _ = policy.act(observations)
        new_log_prob, entropy, _ = policy.evaluate_actions(observations, actions)
        ratio = (new_log_prob - old_log_prob).exp()
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=2e-5))
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(torch.all(actions.abs() <= 0.005))

    def test_clipped_surrogate_handles_positive_and_negative_advantages(self):
        ratio = torch.tensor((1.3, 0.7))
        positive = ppo_clipped_surrogate(ratio, torch.ones(2), 0.2)
        negative = ppo_clipped_surrogate(ratio, -torch.ones(2), 0.2)
        self.assertTrue(torch.allclose(positive, torch.tensor((1.2, 0.7))))
        self.assertTrue(torch.allclose(negative, torch.tensor((-1.3, -0.8))))

    def test_one_cpu_update_and_checkpoint(self):
        trainer = PPOTrainer(
            FakeEnvironment(),
            config=PPOConfig(rollout_steps=8, update_epochs=1, batch_size=4),
        )
        metrics = trainer.train_one_update()
        self.assertEqual(8, trainer.training_steps)
        self.assertEqual(1, trainer.update_steps)
        self.assertIn("policy_loss", metrics)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save_checkpoint(
                Path(directory) / "artifacts", "project-test", "run-test"
            )
            self.assertEqual("model_latest.pt", checkpoint.name)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("policy", payload)
            self.assertIn("optimizer", payload)
            self.assertEqual(4, payload["checkpoint_schema_version"])
            self.assertEqual(2, payload["action_dim"])
            self.assertEqual(ACTION_SEMANTICS, payload["action_semantics"])
            self.assertEqual(OBSERVATION_DIM, payload["observation_dim"])
            self.assertEqual(PUSH_OBSERVATION_SCHEMA_VERSION, payload["observation_schema_version"])
            self.assertEqual(PUSH_OBSERVATION_SEMANTICS, payload["observation_semantics"])
            self.assertIn("normalizer_state", payload)
            self.assertGreater(payload["normalizer_state"]["count"], 0)
            self.assertEqual(2, payload["reward_schema_version"])
            self.assertIn("directional_push_reward_scale", payload["reward_weights"])
            self.assertEqual(2, payload["ppo_training_schema_version"])
            self.assertTrue(payload["ppo_training_config"]["advantage_normalization"])
            self.assertTrue(payload["ppo_training_config"]["timeout_bootstrap"])
            self.assertEqual(8, payload["training_steps"])

    def test_training_config_validation_and_metrics_log(self):
        with tempfile.TemporaryDirectory() as directory:
            training = PPOTrainingConfig(
                rollout_steps=4,
                num_updates=2,
                project_id="project-config",
                run_id="run-config",
                artifacts_root=Path(directory) / "artifacts",
            )
            trainer = PPOTrainer(
                FakeEnvironment(),
                config=PPOConfig(rollout_steps=4, update_epochs=1, batch_size=4),
            )
            published = []
            records = trainer.run_training(training, metric_callback=published.append)
            metrics_path = (
                training.artifacts_root
                / training.project_id
                / training.run_id
                / "metrics"
                / "training.jsonl"
            )
            lines = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            self.assertEqual(2, len(records))
            self.assertEqual(2, len(lines))
            self.assertEqual(2, len(published))
            self.assertNotIn("success_rate", published[0])
            self.assertEqual([4, 8], [record["training_steps"] for record in published])
            self.assertEqual([1, 2], [record["update_index"] for record in lines])
            self.assertEqual([4, 8], [record["training_steps"] for record in lines])
            self.assertEqual(
                {
                    "success_rate", "timeout_rate", "failure_rate",
                    "final_distance", "minimum_distance", "object_displacement",
                    "episode_return", "action_norm", "saturation_rate",
                },
                {
                    key for key in lines[0]
                    if key not in {
                        "update_index", "training_steps", "average_reward",
                        "episode_length", "episodes", "policy_loss", "value_loss",
                        "entropy",
                        "total_loss", "approx_kl", "clip_fraction",
                        "advantage_mean_raw", "advantage_std_raw",
                        "advantage_mean_normalized", "advantage_std_normalized",
                        "return_mean", "value_mean", "ratio_mean", "ratio_min",
                        "ratio_max", "actor_grad_norm", "critic_grad_norm",
                        "explained_variance", "samples_collected",
                        "samples_consumed_per_epoch", "total_sample_uses",
                        "minibatches",
                    }
                },
            )
            episodes_path = metrics_path.with_name("episodes.jsonl")
            episode_lines = [
                json.loads(line) for line in episodes_path.read_text().splitlines()
            ]
            self.assertEqual(1, len(episode_lines))
            episode = episode_lines[0]
            self.assertTrue(episode["success"])
            self.assertFalse(episode["timeout"])
            self.assertFalse(episode["failure"])
            self.assertEqual(0.5, episode["final_distance"])
            self.assertEqual(0.5, episode["minimum_distance"])
            self.assertEqual(0.5, episode["object_displacement"])
            self.assertAlmostEqual(0.05, episode["return"])
            self.assertEqual(5, episode["episode_length"])
            self.assertEqual(0.25, episode["action_norm"])
            self.assertAlmostEqual(0.2, episode["saturation_rate"])
            self.assertIn("precontact_progress_return", episode)
            self.assertIn("directional_push_return", episode)
            self.assertIn("lateral_push_penalty_return", episode)
            self.assertIn("contact_bonus_return", episode)
            self.assertIn("success_bonus_return", episode)
            self.assertIn("action_penalty_return", episode)
            self.assertIn("joint_limit_penalty_return", episode)
            self.assertIn("minimum_precontact_error", episode)
            self.assertIn("behind_fraction", episode)
            self.assertIn("near_precontact_steps", episode)
            self.assertIn("dynamic_contact_steps", episode)
            self.assertIn("contact_rate", episode)
            self.assertIn("positive_directional_push_steps", episode)
            self.assertIn("backward_push_steps", episode)
            self.assertIn("directional_object_displacement", episode)
            self.assertIn("lateral_object_displacement", episode)
            self.assertEqual(1.0, lines[1]["success_rate"])
            self.assertTrue(
                (
                    training.artifacts_root
                    / training.project_id
                    / training.run_id
                    / "checkpoints"
                    / "model_best.pt"
                ).is_file()
            )

    def test_episode_outcomes_produce_real_success_timeout_and_failure_rates(self):
        class OutcomeEnvironment:
            def __init__(self):
                self.episode = -1

            def reset(self):
                self.episode += 1
                return observation()

            def step(self, action):
                outcome = ("success", "timeout", "failure")[self.episode]
                return observation(), 1.0, True, {
                    "success": outcome == "success",
                    "timeout": outcome == "timeout",
                    "failure": outcome == "failure",
                    "distance_to_goal": 0.1 + self.episode,
                    "initial_distance_to_goal": 2.0,
                    "object_displacement": 0.2 + self.episode,
                    "action_norm": 0.3,
                    "saturation_rate": 0.4,
                }

        trainer = PPOTrainer(
            OutcomeEnvironment(),
            config=PPOConfig(rollout_steps=3, update_epochs=1, batch_size=3),
        )
        metrics = trainer.collect_rollout()
        self.assertEqual(3.0, metrics["episodes"])
        self.assertAlmostEqual(1.0 / 3.0, metrics["success_rate"])
        self.assertAlmostEqual(1.0 / 3.0, metrics["timeout_rate"])
        self.assertAlmostEqual(1.0 / 3.0, metrics["failure_rate"])
        self.assertEqual(
            [(True, False, False), (False, True, False), (False, False, True)],
            [
                (row["success"], row["timeout"], row["failure"])
                for row in trainer._rollout_episode_records
            ],
        )

    def test_higher_average_reward_atomically_updates_best_checkpoint(self):
        class RewardSequenceTrainer(PPOTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rewards = iter((1.0, 0.5, 2.0))
                self.checkpoint_events = []

            def train_one_update(self, cancel_requested=None):
                self.training_steps += self.config.rollout_steps
                self.update_steps += 1
                reward = next(self.rewards)
                return {
                    "average_reward": reward,
                    "average_episode_length": 1.0,
                    "episodes": 1.0,
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                }

            def save_checkpoint(
                self,
                artifacts_root,
                project_id,
                run_id,
                filename="model_latest.pt",
            ):
                self.checkpoint_events.append((filename, self.update_steps))
                return super().save_checkpoint(
                    artifacts_root, project_id, run_id, filename=filename
                )

        with tempfile.TemporaryDirectory() as directory:
            training = PPOTrainingConfig(
                rollout_steps=4,
                num_updates=3,
                project_id="project-best",
                run_id="run-best",
                artifacts_root=Path(directory) / "artifacts",
            )
            trainer = RewardSequenceTrainer(
                FakeEnvironment(),
                config=PPOConfig(rollout_steps=4, update_epochs=1, batch_size=4),
            )

            trainer.run_training(training)

            best_events = [
                step
                for filename, step in trainer.checkpoint_events
                if filename == "model_best.pt"
            ]
            self.assertEqual([1, 3], best_events)
            best_path = (
                training.artifacts_root
                / training.project_id
                / training.run_id
                / "checkpoints/model_best.pt"
            )
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            self.assertEqual(3, best_payload["update_steps"])

    def test_evaluation_success_rate_overrides_training_reward_for_best_checkpoint(self):
        class EvaluatedTrainer(PPOTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rewards = iter((100.0, -100.0))

            def train_one_update(self, cancel_requested=None):
                self.training_steps += self.config.rollout_steps
                self.update_steps += 1
                return {
                    "average_reward": next(self.rewards),
                    "average_episode_length": 1.0,
                    "episodes": 1.0,
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                }

        evaluations = iter((
            {
                "success_rate": 0.1,
                "confidence_interval_95": {"lower": 0.05},
                "mean_final_distance": 0.1,
            },
            {
                "success_rate": 0.9,
                "confidence_interval_95": {"lower": 0.8},
                "mean_final_distance": 0.2,
            },
        ))
        with tempfile.TemporaryDirectory() as directory:
            training = PPOTrainingConfig(
                rollout_steps=4,
                num_updates=2,
                project_id="project-evaluated-best",
                run_id="run-evaluated-best",
                artifacts_root=Path(directory) / "artifacts",
            )
            trainer = EvaluatedTrainer(
                FakeEnvironment(),
                config=PPOConfig(rollout_steps=4, update_epochs=1, batch_size=4),
            )
            trainer.run_training(
                training, evaluation_callback=lambda checkpoint: next(evaluations)
            )
            best_payload = torch.load(
                training.artifacts_root
                / training.project_id
                / training.run_id
                / "checkpoints/model_best.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(2, best_payload["update_steps"])

    def test_checkpoint_load_restores_and_continues(self):
        config = PPOConfig(rollout_steps=4, update_epochs=1, batch_size=4)
        original = PPOTrainer(FakeEnvironment(), config=config)
        original.train_one_update()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = original.save_checkpoint(
                Path(directory) / "artifacts", "project-resume", "run-resume"
            )
            expected_parameter = next(original.policy.parameters()).detach().clone()
            resumed = PPOTrainer(FakeEnvironment(), config=config)
            resumed.load_checkpoint(checkpoint)
            self.assertTrue(torch.equal(expected_parameter, next(resumed.policy.parameters())))
            self.assertEqual(original.training_steps, resumed.training_steps)
            self.assertEqual(original.update_steps, resumed.update_steps)
            self.assertTrue(resumed.optimizer.state)
            self.assertEqual(original.normalizer.count, resumed.normalizer.count)
            self.assertTrue(torch.equal(original.normalizer.mean, resumed.normalizer.mean))
            self.assertTrue(torch.equal(original.normalizer.m2, resumed.normalizer.m2))
            resumed.train_one_update()
            self.assertEqual(original.training_steps + 4, resumed.training_steps)
            self.assertEqual(original.update_steps + 1, resumed.update_steps)

    def test_minibatches_use_every_sample_once_per_epoch_and_update_parameters(self):
        trainer = PPOTrainer(
            FakeEnvironment(),
            config=PPOConfig(rollout_steps=5, update_epochs=3, batch_size=2),
        )
        before = [parameter.detach().clone() for parameter in trainer.policy.parameters()]
        metrics = trainer.train_one_update()
        self.assertEqual(5, metrics["samples_collected"])
        self.assertEqual([5, 5, 5], metrics["samples_consumed_per_epoch"])
        self.assertEqual(15, metrics["total_sample_uses"])
        self.assertEqual(9, metrics["minibatches"])
        self.assertTrue(0.0 <= metrics["clip_fraction"] <= 1.0)
        for name in (
            "approx_kl", "explained_variance", "actor_grad_norm",
            "critic_grad_norm", "ratio_mean", "ratio_min", "ratio_max",
            "total_loss",
        ):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[name])))
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, trainer.policy.parameters())
        ))
        self.assertTrue(all(torch.isfinite(p).all() for p in trainer.policy.parameters()))

    def test_timeout_next_value_uses_final_observation_before_reset(self):
        class ScalarCritic(torch.nn.Module):
            def forward(self, values):
                return values[..., :1]

        class TimeoutEnvironment:
            def __init__(self): self.reset_calls = 0
            def reset(self):
                self.reset_calls += 1
                return observation(0.0 if self.reset_calls == 1 else -0.0005)
            def step(self, action):
                return observation(0.0005), 0.0, True, {
                    "success": False, "failure": False, "timeout": True,
                    "distance_to_goal": 1.0, "initial_distance_to_goal": 1.0,
                    "object_displacement": 0.0,
                }

        environment = TimeoutEnvironment()
        policy = MLPActorCritic()
        policy.critic = ScalarCritic()
        trainer = PPOTrainer(
            environment, policy=policy,
            config=PPOConfig(rollout_steps=1, update_epochs=1, batch_size=1),
        )
        trainer.collect_rollout()
        self.assertAlmostEqual(
            0.0005 / ((1.0e-6 + 1.0e-8) ** 0.5),
            float(trainer.buffer.next_values[0]), places=5,
        )
        self.assertGreater(float(trainer.buffer.next_values[0]), 0.0)
        self.assertEqual(1.0, trainer.buffer.bootstrap_masks[0])
        self.assertEqual(0.0, trainer.buffer.continuation_masks[0])
        self.assertEqual(2, environment.reset_calls)

    def test_old_6d_checkpoint_is_rejected_with_action_schema_error(self):
        trainer = PPOTrainer(
            FakeEnvironment(),
            config=PPOConfig(rollout_steps=4, update_epochs=1, batch_size=4),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy-6d.pt"
            torch.save(
                {
                    "policy": trainer.policy.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "training_steps": 0,
                    "update_steps": 0,
                    "checkpoint_schema_version": 2,
                    "action_dim": 6,
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "action schema incompatible"):
                trainer.load_checkpoint(checkpoint)

    def test_old_2d_action_but_48d_observation_checkpoint_is_rejected(self):
        trainer = PPOTrainer(FakeEnvironment())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "p1-1-48d.pt"
            torch.save({
                "checkpoint_schema_version": 3,
                "action_dim": 2,
                "action_semantics": ACTION_SEMANTICS,
                "policy": trainer.policy.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "training_steps": 0,
                "update_steps": 0,
            }, checkpoint)
            with self.assertRaisesRegex(ValueError, "observation schema incompatible"):
                trainer.load_checkpoint(checkpoint)

    def test_incompatible_reward_schema_cannot_resume_training(self):
        trainer = PPOTrainer(FakeEnvironment())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save_checkpoint(
                Path(directory) / "artifacts", "project-reward", "run-reward"
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["reward_schema_version"] = 1
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "reward schema incompatible"):
                trainer.load_checkpoint(checkpoint)

    def test_old_ppo_training_schema_cannot_resume(self):
        trainer = PPOTrainer(FakeEnvironment())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = trainer.save_checkpoint(
                Path(directory) / "artifacts", "project-math", "run-math"
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload.pop("ppo_training_schema_version")
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "training schema incompatible"):
                trainer.load_checkpoint(checkpoint)

    def test_running_normalizer_welford_constant_and_freeze(self):
        normalizer = ObservationNormalizer()
        first = torch.zeros(OBSERVATION_DIM)
        second = torch.full((OBSERVATION_DIM,), 2.0)
        out0 = normalizer.normalize(first, update=True)
        out1 = normalizer.normalize(second, update=True)
        self.assertEqual(2, normalizer.count)
        self.assertTrue(torch.isfinite(out0).all())
        self.assertTrue(torch.isfinite(out1).all())
        self.assertTrue(torch.allclose(normalizer.mean, torch.ones(OBSERVATION_DIM, dtype=torch.float64)))
        # previous_action is deterministic passthrough, not running-normalized.
        self.assertTrue(torch.equal(out1[OBSERVATION_SLICES["previous_action_xy"]], torch.full((2,), 2.0)))
        state = normalizer.state_dict()
        frozen = ObservationNormalizer(frozen=True)
        frozen.load_state_dict(state, frozen=True)
        before = frozen.state_dict()
        frozen.normalize(torch.full((OBSERVATION_DIM,), 7.0), update=False)
        after = frozen.state_dict()
        self.assertEqual(before["count"], after["count"])
        self.assertTrue(torch.equal(before["mean"], after["mean"]))
        self.assertTrue(torch.equal(before["m2"], after["m2"]))
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            frozen.normalize(first, update=True)

    def test_cooperative_cancel_checks_during_rollout_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            training = PPOTrainingConfig(
                rollout_steps=32,
                num_updates=10,
                project_id="project-cancel",
                run_id="run-cancel",
                artifacts_root=Path(directory) / "artifacts",
            )
            trainer = PPOTrainer(
                FakeEnvironment(),
                config=PPOConfig(rollout_steps=32, update_epochs=1, batch_size=8),
            )

            with self.assertRaises(TrainingCancelled) as raised:
                trainer.run_training(
                    training, cancel_requested=lambda: trainer.training_steps >= 3
                )

            checkpoint = (
                training.artifacts_root
                / training.project_id
                / training.run_id
                / "checkpoints"
                / "model_latest.pt"
            )
            metrics = (
                training.artifacts_root
                / training.project_id
                / training.run_id
                / "metrics"
                / "training.jsonl"
            )
            self.assertEqual(3, trainer.training_steps)
            self.assertEqual(0, trainer.update_steps)
            self.assertEqual([], raised.exception.records)
            self.assertTrue(checkpoint.is_file())
            self.assertFalse(metrics.exists())


if __name__ == "__main__":
    unittest.main()
