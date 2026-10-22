"""Deterministic, checkpoint-oriented evaluation for S1 Push policies."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch

from .ppo.policy import MLPActorCritic, flatten_observation
from .ppo.normalizer import ObservationNormalizer
from .ppo.trainer import (
    validate_checkpoint_action_schema,
    validate_checkpoint_observation_schema,
)


DEFAULT_EVALUATION_SEEDS = tuple(range(100))


def wilson_confidence_interval(
    success_count: int, episode_count: int, z: float = 1.959963984540054
) -> Tuple[float, float]:
    """Return the two-sided Wilson score interval (95% for the default z)."""
    if isinstance(success_count, bool) or isinstance(episode_count, bool):
        raise ValueError("success_count and episode_count must be integers")
    if episode_count <= 0 or success_count < 0 or success_count > episode_count:
        raise ValueError("success_count must be within a positive episode_count")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    proportion = success_count / episode_count
    z_squared = z * z
    denominator = 1.0 + z_squared / episode_count
    center = (proportion + z_squared / (2.0 * episode_count)) / denominator
    half_width = z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / episode_count
        + z_squared / (4.0 * episode_count * episode_count)
    )
    lower = 0.0 if success_count == 0 else max(0.0, center - half_width)
    upper = 1.0 if success_count == episode_count else min(1.0, center + half_width)
    return lower, upper


@dataclass(frozen=True)
class EvaluationConfig:
    output_directory: Path
    seeds: Sequence[int] = DEFAULT_EVALUATION_SEEDS
    max_episode_steps: int = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        seeds = tuple(self.seeds)
        object.__setattr__(self, "seeds", seeds)
        if len(seeds) < 100:
            raise ValueError("deterministic evaluation requires at least 100 seeds")
        if len(set(seeds)) != len(seeds):
            raise ValueError("evaluation seeds must be unique")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("evaluation seeds must be non-negative integers")
        if (
            isinstance(self.max_episode_steps, bool)
            or not isinstance(self.max_episode_steps, int)
            or self.max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer")


def checkpoint_rank_key(
    evaluation: Optional[Mapping[str, Any]], training_average_reward: float = 0.0
) -> Tuple[float, ...]:
    """Rank evaluated checkpoints first; training reward is only a final tie-break."""
    if evaluation is None:
        return (0.0, -1.0, -1.0, float("-inf"), float(training_average_reward))
    interval = evaluation.get("confidence_interval_95", {})
    lower = float(interval.get("lower", 0.0))
    final_distance = float(evaluation.get("mean_final_distance", float("inf")))
    return (
        1.0,
        float(evaluation["success_rate"]),
        lower,
        -final_distance,
        float(training_average_reward),
    )


class DeterministicEvaluationRunner:
    """Evaluate a policy mean over a fixed, versionable seed suite."""

    def __init__(
        self,
        environment_factory: Callable[[int], Any],
        config: EvaluationConfig,
        device: str = "cpu",
    ) -> None:
        self.environment_factory = environment_factory
        self.config = config
        self.device = torch.device(device)
        self._loaded_normalizer = None

    def load_policy(self, checkpoint_path: Path) -> MLPActorCritic:
        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        validate_checkpoint_action_schema(payload)
        validate_checkpoint_observation_schema(payload)
        if "policy" not in payload:
            raise ValueError("evaluation checkpoint is missing policy state")
        policy = MLPActorCritic().to(self.device)
        policy.load_state_dict(payload["policy"])
        normalizer = ObservationNormalizer(frozen=True)
        normalizer.load_state_dict(payload["normalizer_state"], frozen=True)
        self._loaded_normalizer = normalizer
        return policy

    def run(
        self,
        policy: Optional[MLPActorCritic] = None,
        checkpoint_path: Optional[Path] = None,
        normalizer: Optional[ObservationNormalizer] = None,
    ) -> Mapping[str, Any]:
        if (policy is None) == (checkpoint_path is None):
            raise ValueError("provide exactly one of policy or checkpoint_path")
        active_policy = policy or self.load_policy(Path(checkpoint_path))
        active_normalizer = normalizer if policy is not None else self._loaded_normalizer
        if active_normalizer is None or not active_normalizer.frozen:
            raise ValueError("evaluation requires frozen checkpoint observation statistics")
        active_policy.eval()
        episode_records = []
        with torch.no_grad():
            for seed in self.config.seeds:
                episode_records.append(
                    self._run_episode(active_policy, active_normalizer, seed)
                )

        success_count = sum(record["success"] for record in episode_records)
        episode_count = len(episode_records)
        success_rate = success_count / episode_count
        lower, upper = wilson_confidence_interval(success_count, episode_count)
        summary = {
            "episode_count": episode_count,
            "success_count": success_count,
            "success_rate": success_rate,
            "confidence_interval_95": {
                "method": "wilson",
                "lower": lower,
                "upper": upper,
            },
            "mean_final_distance": sum(
                record["final_distance"] for record in episode_records
            ) / episode_count,
            "seeds": list(self.config.seeds),
        }
        self._save(episode_records, summary)
        return summary

    def _run_episode(self, policy, normalizer, seed: int) -> Mapping[str, Any]:
        random.seed(seed)
        torch.manual_seed(seed)
        environment = self.environment_factory(seed)
        try:
            initialize = getattr(environment, "initialize", None)
            if initialize is not None:
                initialize()
            observation = environment.reset()
            episode_return = 0.0
            minimum_distance = None
            final_info = {}
            length = 0
            done = False
            while not done and length < self.config.max_episode_steps:
                raw_observation = flatten_observation(observation).to(self.device)
                observation_tensor = normalizer.normalize(raw_observation, update=False)
                action, _, _ = policy.act(observation_tensor, deterministic=True)
                observation, reward, done, final_info = environment.step(
                    tuple(float(value) for value in action.detach().cpu())
                )
                length += 1
                episode_return += float(reward)
                distance = float(final_info["distance_to_goal"])
                initial_distance = float(
                    final_info.get("initial_distance_to_goal", distance)
                )
                minimum_distance = (
                    min(initial_distance, distance)
                    if minimum_distance is None
                    else min(minimum_distance, distance)
                )
            forced_timeout = not done
            final_distance = float(final_info.get("distance_to_goal", float("inf")))
            return {
                "seed": seed,
                "success": bool(final_info.get("success", False)) and not forced_timeout,
                "timeout": bool(final_info.get("timeout", False)) or forced_timeout,
                "final_distance": final_distance,
                "minimum_distance": float(
                    final_distance if minimum_distance is None else minimum_distance
                ),
                "object_displacement": float(
                    final_info.get("object_displacement", 0.0)
                ),
                "return": episode_return,
                "length": length,
            }
        finally:
            close = getattr(environment, "close", None)
            if close is not None:
                close()

    def _save(self, episode_records, summary) -> None:
        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        episodes_path = output / "evaluation_episodes.jsonl"
        with episodes_path.open("w", encoding="utf-8") as stream:
            for record in episode_records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        summary_path = output / "evaluation_summary.json"
        temporary_path = output / ".evaluation_summary.json.tmp"
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(summary_path)
