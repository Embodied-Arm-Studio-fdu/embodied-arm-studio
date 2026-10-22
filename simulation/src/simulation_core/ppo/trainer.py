"""Single-process, single-environment PPO training runner."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch

from ..artifacts import ArtifactPaths
from ..push_adapter import (
    PUSH_REWARD_SCHEMA_VERSION,
    PUSH_REWARD_SEMANTICS,
    PUSH_REWARD_TERM_NAMES,
    PushConfig,
    reward_schema_metadata,
)
from .buffer import RolloutBuffer
from .normalizer import ObservationNormalizer
from .policy import (
    ACTION_DIM,
    ACTION_SEMANTICS,
    OBSERVATION_DIM,
    PUSH_OBSERVATION_SCHEMA_VERSION,
    PUSH_OBSERVATION_SEMANTICS,
    MLPActorCritic,
    PUSH_POLICY_SCHEMA,
    PolicySchema,
    flatten_observation,
)


CHECKPOINT_SCHEMA_VERSION = 4
PPO_TRAINING_SCHEMA_VERSION = 2
PPO_TRAINING_SEMANTICS = (
    "per_transition_next_value",
    "true_terminal_bootstrap_mask_zero",
    "timeout_bootstrap_final_state_value",
    "episode_boundary_continuation_mask_zero",
    "global_rollout_advantage_normalization_actor_only",
    "raw_gae_return_critic_target",
    "tanh_gaussian_logprob_with_action_scale_jacobian",
)
ADVANTAGE_NORMALIZATION_EPSILON = 1.0e-8
LIVE_METRIC_FIELDS = frozenset({
    "update_index", "training_steps", "average_reward", "policy_loss", "value_loss", "entropy",
    "episode_length", "episodes",
})


@dataclass(frozen=True)
class PPOTaskContract:
    policy_schema: PolicySchema = PUSH_POLICY_SCHEMA
    reward_schema_version: int = PUSH_REWARD_SCHEMA_VERSION
    reward_semantics: tuple = PUSH_REWARD_SEMANTICS
    reward_term_names: tuple = PUSH_REWARD_TERM_NAMES
    reward_metadata_factory: Callable[[Any], Mapping[str, Any]] = reward_schema_metadata

    @property
    def task_id(self):
        return self.policy_schema.task_id


PUSH_PPO_CONTRACT = PPOTaskContract()


class TrainingCancelled(Exception):
    """Internal control signal for a cooperative, non-failure training stop."""

    def __init__(self, records: Optional[List[Dict[str, float]]] = None) -> None:
        super().__init__("training was cooperatively stopped")
        self.records = list(records or [])


def validate_checkpoint_action_schema(
    payload: Mapping[str, Any], contract: PPOTaskContract = PUSH_PPO_CONTRACT
) -> None:
    """Reject checkpoints whose actor action contract is not exactly Push 2D."""
    action_dim = payload.get("action_dim")
    semantics = payload.get("action_semantics")
    if (
        payload.get("task_id", "s1_push") != contract.task_id
        or action_dim != contract.policy_schema.action_dim
        or semantics != contract.policy_schema.action_semantics
    ):
        raise ValueError(
            "PPO checkpoint action schema incompatible: expected action_dim=%d, "
            "action_semantics=%r; got action_dim=%r, action_semantics=%r. "
            "Old 6D/18D checkpoints cannot "
            "be truncated or padded into the 2D policy."
            % (
                contract.policy_schema.action_dim,
                contract.policy_schema.action_semantics,
                action_dim,
                semantics,
            )
        )


def validate_checkpoint_observation_schema(
    payload: Mapping[str, Any], contract: PPOTaskContract = PUSH_PPO_CONTRACT
) -> None:
    """Require the exact observation contract and stored training statistics."""
    dimension = payload.get("observation_dim")
    schema_version = payload.get("observation_schema_version")
    semantics = payload.get("observation_semantics")
    normalizer_state = payload.get("normalizer_state")
    if (
        payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("task_id", "s1_push") != contract.task_id
        or dimension != contract.policy_schema.observation_dim
        or schema_version != contract.policy_schema.observation_schema_version
        or tuple(semantics or ()) != contract.policy_schema.observation_semantics
        or not isinstance(normalizer_state, Mapping)
    ):
        raise ValueError(
            "PPO checkpoint observation schema incompatible: expected checkpoint "
            "version %d, observation_dim=%d, observation_schema_version=%d and "
            "stored normalizer statistics; got version %r, dim=%r, schema=%r. "
            "Old 48D checkpoints cannot be truncated, padded, or evaluated with "
            "an identity normalizer."
            % (
                CHECKPOINT_SCHEMA_VERSION,
                contract.policy_schema.observation_dim,
                contract.policy_schema.observation_schema_version,
                payload.get("checkpoint_schema_version"),
                dimension,
                schema_version,
            )
        )


def validate_checkpoint_reward_schema(
    payload: Mapping[str, Any], config: Any,
    contract: PPOTaskContract = PUSH_PPO_CONTRACT,
) -> None:
    expected = contract.reward_metadata_factory(config)
    if (
        payload.get("reward_schema_version") != contract.reward_schema_version
        or tuple(payload.get("reward_semantics") or ()) != contract.reward_semantics
        or payload.get("reward_weights") != expected["reward_weights"]
    ):
        raise ValueError(
            "PPO checkpoint reward schema incompatible for training resume: "
            "expected reward_schema_version=%d and current reward weights. "
            "Evaluation-only loading may ignore reward metadata, but training "
            "cannot silently resume across reward definitions."
            % contract.reward_schema_version
        )


def ppo_clipped_surrogate(ratio, advantage, clip_ratio: float):
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    return torch.minimum(unclipped, clipped)


def normalize_advantages(advantages, epsilon: float = ADVANTAGE_NORMALIZATION_EPSILON):
    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    normalized = (advantages - mean) / std.clamp_min(epsilon)
    return normalized, mean, std


def explained_variance(returns, values, epsilon: float = 1.0e-8) -> float:
    return_variance = returns.var(unbiased=False)
    if float(return_variance) <= epsilon:
        return 0.0
    value = 1.0 - (returns - values).var(unbiased=False) / return_variance
    return float(value)


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 32
    update_epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if self.rollout_steps <= 0 or self.update_epochs <= 0 or self.batch_size <= 0:
            raise ValueError("rollout_steps, update_epochs, and batch_size must be positive")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1:
            raise ValueError("learning_rate and gamma are invalid")
        if not 0 <= self.gae_lambda <= 1 or self.clip_ratio <= 0:
            raise ValueError("gae_lambda or clip_ratio is invalid")


def ppo_training_metadata(config: PPOConfig) -> Dict[str, Any]:
    return {
        "ppo_training_schema_version": PPO_TRAINING_SCHEMA_VERSION,
        "ppo_training_semantics": PPO_TRAINING_SEMANTICS,
        "ppo_training_config": {
            "advantage_normalization": True,
            "advantage_normalization_epsilon": ADVANTAGE_NORMALIZATION_EPSILON,
            "timeout_bootstrap": True,
            "gae_terminal_semantics": "separate_bootstrap_and_continuation_masks",
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_ratio": config.clip_ratio,
            "entropy_coefficient": config.entropy_coefficient,
            "learning_rate": config.learning_rate,
            "update_epochs": config.update_epochs,
            "batch_size": config.batch_size,
        },
    }


def validate_checkpoint_training_schema(payload: Mapping[str, Any], config: PPOConfig) -> None:
    expected = ppo_training_metadata(config)
    if (
        payload.get("ppo_training_schema_version") != PPO_TRAINING_SCHEMA_VERSION
        or tuple(payload.get("ppo_training_semantics") or ()) != PPO_TRAINING_SEMANTICS
        or payload.get("ppo_training_config") != expected["ppo_training_config"]
    ):
        raise ValueError(
            "PPO checkpoint training schema incompatible for resume: expected "
            "training schema version %d with timeout bootstrap, separated GAE "
            "masks, and advantage normalization. Evaluation-only loading does "
            "not require this training metadata."
            % PPO_TRAINING_SCHEMA_VERSION
        )


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Artifact-aware configuration for a finite sequence of PPO updates."""

    rollout_steps: int
    num_updates: int
    project_id: str
    run_id: str
    artifacts_root: Path
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root))
        if self.rollout_steps <= 0 or self.num_updates <= 0:
            raise ValueError("rollout_steps and num_updates must be positive")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        # Reuse the path component validation at config acceptance time.
        ArtifactPaths(self.artifacts_root).run_root(self.project_id, self.run_id)


class PPOTrainer:
    """Collect short PushAdapter rollouts and perform standard clipped PPO updates."""

    def __init__(
        self,
        environment: Any,
        policy: Optional[MLPActorCritic] = None,
        config: Optional[PPOConfig] = None,
        device: str = "cpu",
        normalizer: Optional[ObservationNormalizer] = None,
        task_contract: PPOTaskContract = PUSH_PPO_CONTRACT,
    ) -> None:
        self.environment = environment
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.task_contract = task_contract
        schema = task_contract.policy_schema
        self.policy = (policy or MLPActorCritic(
            observation_dim=schema.observation_dim,
            action_dim=schema.action_dim,
            action_low=schema.action_low,
            action_high=schema.action_high,
        )).to(self.device)
        self.normalizer = normalizer or ObservationNormalizer(
            dimension=task_contract.policy_schema.observation_dim
        )
        if self.normalizer.frozen:
            raise ValueError("training requires an updateable observation normalizer")
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )
        self.buffer = RolloutBuffer(action_dim=schema.action_dim)
        self.training_steps = 0
        self.update_steps = 0
        self._observation = None
        self._episode_length = 0
        self._episode_return = 0.0
        self._episode_min_distance = None
        self._episode_action_norm_sum = 0.0
        self._episode_saturation_sum = 0.0
        self._episode_object_displacement = 0.0
        self._episode_reward_term_returns = dict.fromkeys(task_contract.reward_term_names, 0.0)
        self._episode_minimum_precontact_error = None
        self._episode_behind_steps = 0
        self._episode_near_precontact_steps = 0
        self._episode_dynamic_contact_steps = 0
        self._episode_positive_directional_push_steps = 0
        self._episode_backward_push_steps = 0
        self._episode_directional_object_displacement = 0.0
        self._episode_lateral_object_displacement = 0.0
        self._episode_index = 0
        self._rollout_episode_records = []

    @staticmethod
    def _check_cancel(cancel_requested: Optional[Callable[[], bool]]) -> None:
        if cancel_requested is not None and cancel_requested():
            raise TrainingCancelled()

    def collect_rollout(
        self, cancel_requested: Optional[Callable[[], bool]] = None
    ) -> Dict[str, float]:
        self.buffer.clear()
        self._rollout_episode_records = []
        self._check_cancel(cancel_requested)
        if self._observation is None:
            self._observation = self.environment.reset()
        reward_sum = 0.0
        episodes = 0
        completed_episode_lengths = []
        for _ in range(self.config.rollout_steps):
            # An environment step is the smallest safe cancellation point: no
            # simulator call or buffer mutation is interrupted halfway through.
            self._check_cancel(cancel_requested)
            raw_observation = flatten_observation(
                self._observation, self.task_contract.policy_schema
            ).to(self.device)
            observation_tensor = self.normalizer.normalize(raw_observation, update=True)
            with torch.no_grad():
                action, log_prob, value = self.policy.act(observation_tensor)
            next_observation, reward, done, info = self.environment.step(
                tuple(float(item) for item in action.detach().cpu())
            )
            with torch.no_grad():
                raw_next = flatten_observation(
                    next_observation, self.task_contract.policy_schema
                ).to(self.device)
                normalized_next = self.normalizer.normalize(raw_next, update=False)
                next_value = self.policy.critic(normalized_next).squeeze(-1)
            timeout = bool(info.get("timeout", False))
            true_terminal = bool(done) and not timeout
            bootstrap_mask = 0.0 if true_terminal else 1.0
            continuation_mask = 0.0 if done else 1.0
            self.buffer.add(
                observation_tensor, action, reward, done, value, log_prob,
                next_value, bootstrap_mask, continuation_mask,
            )
            reward_sum += float(reward)
            self.training_steps += 1
            self._episode_length += 1
            self._episode_return += float(reward)
            distance = float(info.get("distance_to_goal", 0.0))
            initial_distance = float(info.get("initial_distance_to_goal", distance))
            if self._episode_min_distance is None:
                self._episode_min_distance = min(initial_distance, distance)
            else:
                self._episode_min_distance = min(self._episode_min_distance, distance)
            self._episode_action_norm_sum += float(info.get("action_norm", 0.0))
            self._episode_saturation_sum += float(info.get("saturation_rate", 0.0))
            self._episode_object_displacement = float(
                info.get("object_displacement", self._episode_object_displacement)
            )
            for name, value in info.get("reward_terms", {}).items():
                self._episode_reward_term_returns[name] = (
                    self._episode_reward_term_returns.get(name, 0.0) + float(value)
                )
            precontact_error = float(info.get("precontact_error", float("inf")))
            if math.isfinite(precontact_error):
                self._episode_minimum_precontact_error = (
                    precontact_error
                    if self._episode_minimum_precontact_error is None
                    else min(self._episode_minimum_precontact_error, precontact_error)
                )
            self._episode_behind_steps += int(float(info.get("longitudinal", 0.0)) < 0.0)
            self._episode_near_precontact_steps += int(bool(info.get("near_precontact", False)))
            self._episode_dynamic_contact_steps += int(bool(info.get("contact", False)))
            directional_push = float(info.get("directional_push", 0.0))
            lateral_push = float(info.get("lateral_push", 0.0))
            self._episode_positive_directional_push_steps += int(directional_push > 1.0e-9)
            self._episode_backward_push_steps += int(directional_push < -1.0e-9)
            self._episode_directional_object_displacement += directional_push
            self._episode_lateral_object_displacement += abs(lateral_push)
            if done:
                episodes += 1
                completed_episode_lengths.append(self._episode_length)
                self._rollout_episode_records.append(self._finish_episode(info, distance))
                self._observation = self.environment.reset()
            else:
                self._observation = next_observation

        self.buffer.compute_returns_and_advantages(
            self.config.gamma, self.config.gae_lambda
        )
        average_episode_length = (
            sum(completed_episode_lengths) / len(completed_episode_lengths)
            if completed_episode_lengths
            else float(self._episode_length)
        )
        episode_metrics = self._aggregate_episode_records(self._rollout_episode_records)
        return {
            "rollout_reward": reward_sum,
            "average_reward": reward_sum / len(self.buffer),
            "episodes": float(episodes),
            "average_episode_length": average_episode_length,
            "rollout_steps": float(len(self.buffer)),
            **episode_metrics,
        }

    def _finish_episode(self, info, final_distance):
        self._episode_index += 1
        length = self._episode_length
        record = {
            "episode_index": self._episode_index,
            "training_steps": self.training_steps,
            "success": bool(info.get("success", False)),
            "timeout": bool(info.get("timeout", False)),
            "failure": bool(info.get("failure", False)),
            "final_distance": float(final_distance),
            "minimum_distance": float(
                final_distance
                if self._episode_min_distance is None
                else self._episode_min_distance
            ),
            "object_displacement": float(self._episode_object_displacement),
            "return": float(self._episode_return),
            "episode_length": int(length),
            "action_norm": self._episode_action_norm_sum / max(1, length),
            "saturation_rate": self._episode_saturation_sum / max(1, length),
            "minimum_precontact_error": float(
                0.0 if self._episode_minimum_precontact_error is None
                else self._episode_minimum_precontact_error
            ),
            "behind_fraction": self._episode_behind_steps / max(1, length),
            "near_precontact_steps": self._episode_near_precontact_steps,
            "dynamic_contact_steps": self._episode_dynamic_contact_steps,
            "contact_rate": self._episode_dynamic_contact_steps / max(1, length),
            "positive_directional_push_steps": self._episode_positive_directional_push_steps,
            "backward_push_steps": self._episode_backward_push_steps,
            "directional_object_displacement": self._episode_directional_object_displacement,
            "lateral_object_displacement": self._episode_lateral_object_displacement,
        }
        for name, value in self._episode_reward_term_returns.items():
            metric_name = {
                "directional_push_progress": "directional_push",
                "contact_sustain_bonus": "contact_bonus",
            }.get(name, name)
            record["%s_return" % metric_name] = value
        self._episode_length = 0
        self._episode_return = 0.0
        self._episode_min_distance = None
        self._episode_action_norm_sum = 0.0
        self._episode_saturation_sum = 0.0
        self._episode_object_displacement = 0.0
        self._episode_reward_term_returns = dict.fromkeys(self.task_contract.reward_term_names, 0.0)
        self._episode_minimum_precontact_error = None
        self._episode_behind_steps = 0
        self._episode_near_precontact_steps = 0
        self._episode_dynamic_contact_steps = 0
        self._episode_positive_directional_push_steps = 0
        self._episode_backward_push_steps = 0
        self._episode_directional_object_displacement = 0.0
        self._episode_lateral_object_displacement = 0.0
        return record

    @staticmethod
    def _aggregate_episode_records(records):
        if not records:
            return {
                "success_rate": 0.0,
                "timeout_rate": 0.0,
                "failure_rate": 0.0,
                "final_distance": 0.0,
                "minimum_distance": 0.0,
                "object_displacement": 0.0,
                "episode_return": 0.0,
                "action_norm": 0.0,
                "saturation_rate": 0.0,
            }
        count = len(records)
        return {
            "success_rate": sum(item["success"] for item in records) / count,
            "timeout_rate": sum(item["timeout"] for item in records) / count,
            "failure_rate": sum(item["failure"] for item in records) / count,
            "final_distance": sum(item["final_distance"] for item in records) / count,
            "minimum_distance": sum(item["minimum_distance"] for item in records) / count,
            "object_displacement": sum(item["object_displacement"] for item in records) / count,
            "episode_return": sum(item["return"] for item in records) / count,
            "action_norm": sum(item["action_norm"] for item in records) / count,
            "saturation_rate": sum(item["saturation_rate"] for item in records) / count,
        }

    def update(self) -> Dict[str, float]:
        if not len(self.buffer) or self.buffer.returns is None:
            raise RuntimeError("collect_rollout() must run before update()")
        raw_advantages = self.buffer.advantages
        normalized_advantages, advantage_mean, advantage_std = normalize_advantages(
            raw_advantages
        )
        values_old = torch.stack(self.buffer.values)
        explained = explained_variance(self.buffer.returns, values_old)
        totals = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "total_loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0,
            "ratio": 0.0, "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
        }
        ratio_min = float("inf")
        ratio_max = float("-inf")
        samples_consumed_per_epoch = []
        total_sample_uses = 0
        batches = 0
        for _ in range(self.config.update_epochs):
            epoch_samples = 0
            for batch in self.buffer.batches(
                self.config.batch_size, advantages=normalized_advantages
            ):
                log_prob, entropy, value = self.policy.evaluate_actions(
                    batch["observations"], batch["actions"]
                )
                log_ratio = log_prob - batch["old_log_probs"]
                ratio = log_ratio.exp()
                policy_loss = -ppo_clipped_surrogate(
                    ratio, batch["advantages"], self.config.clip_ratio
                ).mean()
                value_loss = (value - batch["returns"]).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_parameters = list(self.policy.actor.parameters()) + [self.policy.log_std]
                critic_parameters = list(self.policy.critic.parameters())
                actor_grad_norm = self._gradient_norm(actor_parameters)
                critic_grad_norm = self._gradient_norm(critic_parameters)
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                batch_samples = int(batch["actions"].shape[0])
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > self.config.clip_ratio
                ).float().mean()
                batch_values = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy_mean,
                    "total_loss": loss,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "ratio": ratio.mean(),
                    "actor_grad_norm": actor_grad_norm,
                    "critic_grad_norm": critic_grad_norm,
                }
                for name, metric in batch_values.items():
                    totals[name] += float(metric.detach()) * batch_samples
                ratio_min = min(ratio_min, float(ratio.detach().min()))
                ratio_max = max(ratio_max, float(ratio.detach().max()))
                epoch_samples += batch_samples
                total_sample_uses += batch_samples
                batches += 1
            samples_consumed_per_epoch.append(epoch_samples)
        self.update_steps += 1
        metrics = {name: value / total_sample_uses for name, value in totals.items()}
        metrics.update({
            "ratio_mean": metrics.pop("ratio"),
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "advantage_mean_raw": float(advantage_mean),
            "advantage_std_raw": float(advantage_std),
            "advantage_mean_normalized": float(normalized_advantages.mean()),
            "advantage_std_normalized": float(normalized_advantages.std(unbiased=False)),
            "return_mean": float(self.buffer.returns.mean()),
            "value_mean": float(values_old.mean()),
            "explained_variance": explained,
            "samples_collected": len(self.buffer),
            "samples_consumed_per_epoch": samples_consumed_per_epoch,
            "total_sample_uses": total_sample_uses,
            "minibatches": batches,
        })
        if not all(
            math.isfinite(float(value))
            for value in metrics.values()
            if isinstance(value, (int, float))
        ):
            raise RuntimeError("PPO update produced non-finite diagnostics")
        return metrics

    @staticmethod
    def _gradient_norm(parameters):
        squared = None
        for parameter in parameters:
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().pow(2).sum()
            squared = value if squared is None else squared + value
        return torch.zeros(()) if squared is None else squared.sqrt()

    def train_one_update(
        self, cancel_requested: Optional[Callable[[], bool]] = None
    ) -> Dict[str, float]:
        metrics = self.collect_rollout(cancel_requested)
        # Finish an optimizer update once it starts. This avoids checkpointing
        # partially-applied minibatches while still checking at every rollout.
        self._check_cancel(cancel_requested)
        metrics.update(self.update())
        return metrics

    def save_checkpoint(
        self,
        artifacts_root: Path,
        project_id: str,
        run_id: str,
        filename: str = "model_latest.pt",
    ) -> Path:
        if filename not in ("model_latest.pt", "model_best.pt"):
            raise ValueError("unsupported checkpoint filename: %s" % filename)
        run_root = ArtifactPaths(Path(artifacts_root)).run_root(project_id, run_id)
        checkpoint_directory = run_root / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_directory / filename
        temporary_path = checkpoint_directory / (".%s.tmp" % filename)
        reward_metadata = self.task_contract.reward_metadata_factory(
            getattr(self.environment, "config", PushConfig())
        )
        schema = self.task_contract.policy_schema
        training_metadata = ppo_training_metadata(self.config)
        torch.save(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "task_id": self.task_contract.task_id,
                "action_dim": schema.action_dim,
                "action_semantics": schema.action_semantics,
                "observation_dim": schema.observation_dim,
                "observation_schema_version": schema.observation_schema_version,
                "observation_semantics": schema.observation_semantics,
                "normalizer_state": self.normalizer.state_dict(),
                **reward_metadata,
                **training_metadata,
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "update_steps": self.update_steps,
                "ppo_config": asdict(self.config),
                "environment_config": (
                    asdict(self.environment.config)
                    if hasattr(self.environment, "config") else {}
                ),
            },
            temporary_path,
        )
        temporary_path.replace(checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Restore model, optimizer, and counters for a clean continuation."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError("PPO checkpoint does not exist: %s" % checkpoint_path)
        payload = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        required = {"policy", "optimizer", "training_steps", "update_steps"}
        missing = required.difference(payload)
        if missing:
            raise ValueError("PPO checkpoint is missing: %s" % ", ".join(sorted(missing)))
        validate_checkpoint_action_schema(payload, self.task_contract)
        validate_checkpoint_observation_schema(payload, self.task_contract)
        validate_checkpoint_reward_schema(
            payload, getattr(self.environment, "config", PushConfig()), self.task_contract
        )
        validate_checkpoint_training_schema(payload, self.config)
        self.normalizer.load_state_dict(payload["normalizer_state"], frozen=False)
        self.policy.load_state_dict(payload["policy"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.training_steps = int(payload["training_steps"])
        self.update_steps = int(payload["update_steps"])
        self.buffer.clear()
        self._observation = None
        self._episode_length = 0
        self._episode_return = 0.0
        self._episode_min_distance = None
        self._episode_action_norm_sum = 0.0
        self._episode_saturation_sum = 0.0
        self._episode_object_displacement = 0.0
        self._episode_reward_term_returns = dict.fromkeys(self.task_contract.reward_term_names, 0.0)
        self._episode_minimum_precontact_error = None
        self._episode_behind_steps = 0
        self._episode_near_precontact_steps = 0
        self._episode_dynamic_contact_steps = 0
        self._episode_positive_directional_push_steps = 0
        self._episode_backward_push_steps = 0
        self._episode_directional_object_displacement = 0.0
        self._episode_lateral_object_displacement = 0.0
        self._episode_index = 0
        self._rollout_episode_records = []
        return payload

    def run_training(
        self,
        training: PPOTrainingConfig,
        cancel_requested: Optional[Callable[[], bool]] = None,
        metric_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
        evaluation_callback: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    ) -> List[Dict[str, float]]:
        """Run a finite training cycle with JSONL metrics and latest checkpoints."""
        if training.rollout_steps != self.config.rollout_steps:
            raise ValueError("training rollout_steps must match PPOConfig.rollout_steps")
        run_root = ArtifactPaths(training.artifacts_root).run_root(
            training.project_id, training.run_id
        )
        metrics_directory = run_root / "metrics"
        metrics_directory.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_directory / "training.jsonl"
        episodes_path = metrics_directory / "episodes.jsonl"
        records = []
        best_checkpoint_rank = getattr(
            self, "initial_best_checkpoint_rank", None
        )
        if best_checkpoint_rank is not None:
            self.save_checkpoint(
                training.artifacts_root,
                training.project_id,
                training.run_id,
                filename="model_best.pt",
            )
        try:
            for local_update in range(1, training.num_updates + 1):
                self._check_cancel(cancel_requested)
                metrics = self.train_one_update(cancel_requested)
                record = {
                    "update_index": self.update_steps,
                    "training_steps": self.training_steps,
                    "average_reward": metrics["average_reward"],
                    "episode_length": metrics["average_episode_length"],
                    "episodes": metrics["episodes"],
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "success_rate": metrics.get("success_rate", 0.0),
                    "timeout_rate": metrics.get("timeout_rate", 0.0),
                    "failure_rate": metrics.get("failure_rate", 0.0),
                    "final_distance": metrics.get("final_distance", 0.0),
                    "minimum_distance": metrics.get("minimum_distance", 0.0),
                    "object_displacement": metrics.get("object_displacement", 0.0),
                    "episode_return": metrics.get("episode_return", 0.0),
                    "action_norm": metrics.get("action_norm", 0.0),
                    "saturation_rate": metrics.get("saturation_rate", 0.0),
                    **{
                        name: metrics.get(
                            name, [] if name == "samples_consumed_per_epoch" else 0.0
                        ) for name in (
                            "total_loss", "approx_kl", "clip_fraction",
                            "advantage_mean_raw", "advantage_std_raw",
                            "advantage_mean_normalized", "advantage_std_normalized",
                            "return_mean", "value_mean", "ratio_mean", "ratio_min",
                            "ratio_max", "actor_grad_norm", "critic_grad_norm",
                            "explained_variance", "samples_collected",
                            "samples_consumed_per_epoch", "total_sample_uses",
                            "minibatches",
                        )
                    },
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                if self._rollout_episode_records:
                    with episodes_path.open("a", encoding="utf-8") as stream:
                        for episode_record in self._rollout_episode_records:
                            stream.write(
                                json.dumps(episode_record, sort_keys=True) + "\n"
                            )
                records.append(record)
                if metric_callback is not None:
                    metric_callback({
                        name: value for name, value in record.items()
                        if name in LIVE_METRIC_FIELDS
                    })
                checkpoint_due = (
                    local_update % training.checkpoint_every == 0
                    or local_update == training.num_updates
                )
                if checkpoint_due:
                    latest_path = self.save_checkpoint(
                        training.artifacts_root, training.project_id, training.run_id
                    )
                    evaluation = (
                        dict(evaluation_callback(latest_path))
                        if evaluation_callback is not None
                        else None
                    )
                    from ..evaluation import checkpoint_rank_key
                    rank = checkpoint_rank_key(
                        evaluation, float(record["average_reward"])
                    )
                    if best_checkpoint_rank is None or rank > best_checkpoint_rank:
                        self.save_checkpoint(
                            training.artifacts_root,
                            training.project_id,
                            training.run_id,
                            filename="model_best.pt",
                        )
                        best_checkpoint_rank = rank
                self._check_cancel(cancel_requested)
        except TrainingCancelled as exc:
            # A model is usable after at least one simulator step. Do not create
            # an empty checkpoint when cancellation happened before training.
            if self.training_steps > 0:
                self.save_checkpoint(
                    training.artifacts_root, training.project_id, training.run_id
                )
            raise TrainingCancelled(records) from exc
        return records
