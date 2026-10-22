"""PPO/checkpoint/evaluation support for experimental S1 Lift."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch

from ...evaluation import wilson_confidence_interval
from ...ppo.normalizer import ObservationNormalizer
from ...ppo.policy import flatten_observation
from ...ppo.trainer import PPOTaskContract, PPOTrainer
from .lift_env import LIFT_BOOTSTRAP_VERSION, lift_config_digest
from .lift_schema import (
    LIFT_ACTION_SCHEMA_ID, LIFT_OBSERVATION_DIM, LIFT_OBSERVATION_SCHEMA_ID,
    LIFT_POLICY_SCHEMA, LIFT_REWARD_SCHEMA_VERSION, LIFT_REWARD_SEMANTICS,
    LIFT_REWARD_TERM_NAMES, LIFT_REWARD_VERSION,
    LIFT_SUCCESS_CRITERIA_VERSION, LIFT_LONG_HOLD_SUCCESS_CRITERIA_VERSION,
    reward_schema_metadata,
)


LIFT_PPO_CONTRACT = PPOTaskContract(
    LIFT_POLICY_SCHEMA, LIFT_REWARD_SCHEMA_VERSION, LIFT_REWARD_SEMANTICS,
    LIFT_REWARD_TERM_NAMES, reward_schema_metadata,
)


def success_criteria_version(config):
    return (LIFT_LONG_HOLD_SUCCESS_CRITERIA_VERSION
            if float(config.hold_seconds) == 5.0 else LIFT_SUCCESS_CRITERIA_VERSION)


def reward_config_digest(config):
    encoded = json.dumps(reward_schema_metadata(config), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def fixed_observation_normalizer():
    return ObservationNormalizer(
        dimension=LIFT_OBSERVATION_DIM,
        previous_action_slice=slice(0, LIFT_OBSERVATION_DIM),
    )


class ExperimentalLiftPPOTrainer(PPOTrainer):
    def __init__(self, environment, **kwargs):
        kwargs.setdefault("task_contract", LIFT_PPO_CONTRACT)
        kwargs.setdefault("normalizer", fixed_observation_normalizer())
        super().__init__(environment, **kwargs)
        self.parent_checkpoint = None
        self.initial_best_checkpoint_rank = None

    def initialize_from_p3_1(self, checkpoint_path, expected_sha256):
        """Transfer the verified P3-1 policy without legacy optimizer state."""
        checkpoint_path = Path(checkpoint_path)
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("P3-1 checkpoint SHA-256 mismatch")
        payload = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False,
        )
        expected = {
            "task_id": "s1_lift",
            "observation_schema_id": LIFT_OBSERVATION_SCHEMA_ID,
            "action_schema_id": LIFT_ACTION_SCHEMA_ID,
            "action_dim": 1,
            "reward_version": LIFT_REWARD_VERSION,
            "reward_config_digest": reward_config_digest(self.environment.config),
            "success_criteria_version": success_criteria_version(
                self.environment.config
            ),
            "physics_config_digest": lift_config_digest(self.environment.config),
            "bootstrap_version": LIFT_BOOTSTRAP_VERSION,
            "target_lift": self.environment.config.target_lift,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError("P3-1 transfer checkpoint %s incompatible" % name)
        self.policy.load_state_dict(payload["policy"])
        self.normalizer.load_state_dict(payload["normalizer_state"], frozen=False)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate,
        )
        self.training_steps = self.update_steps = 0
        self.parent_checkpoint = {
            "path": str(checkpoint_path),
            "sha256": digest,
            "training_steps": int(payload["training_steps"]),
            "transfer_strategy": (
                "verified P3-1 actor, critic, and normalizer; fresh optimizer and counters"
            ),
        }
        # The transferred policy passed the formal Lift gate. Preserve it as
        # the Run-local best checkpoint unless a future evaluated checkpoint
        # proves better; training-only reward must not overwrite it.
        self.initial_best_checkpoint_rank = (1.0, 1.0, 1.0, 0.0, 0.0)
        return payload

    def _finish_episode(self, info, final_distance):
        record = super()._finish_episode(info, final_distance)
        record.update({name: info.get(name) for name in (
            "failure_reason", "cube_rise", "maximum_cube_rise", "target_reached",
            "hold_duration", "hold_half_second", "hold_1s", "hold_3s", "hold_5s",
            "relative_translation_drift",
            "relative_rotation_drift", "blockage_gap", "jaw_actual",
            "reached_20mm", "reached_40mm", "unsafe", "non_finite", "drop",
            "grasp_lost", "target_overshoot", "bootstrap",
        )})
        return record

    @staticmethod
    def _aggregate_episode_records(records):
        result = PPOTrainer._aggregate_episode_records(records)
        if not records:
            result.update({"target_reached_rate": 0.0, "hold_1s_rate": 0.0,
                           "drop_rate": 0.0, "grasp_lost_rate": 0.0})
            return result
        count = len(records)
        result.update({
            "target_reached_rate": sum(bool(x.get("target_reached")) for x in records) / count,
            "hold_1s_rate": sum(bool(x.get("success")) for x in records) / count,
            "drop_rate": sum(bool(x.get("drop")) for x in records) / count,
            "grasp_lost_rate": sum(bool(x.get("grasp_lost")) for x in records) / count,
            "mean_maximum_cube_rise": sum(float(x.get("maximum_cube_rise", 0.0))
                                            for x in records) / count,
            "mean_hold_duration": sum(float(x.get("hold_duration", 0.0))
                                       for x in records) / count,
        })
        return result

    def save_checkpoint(self, artifacts_root: Path, project_id: str, run_id: str,
                        filename: str = "model_latest.pt") -> Path:
        path = super().save_checkpoint(artifacts_root, project_id, run_id, filename)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update({
            "observation_schema_id": LIFT_OBSERVATION_SCHEMA_ID,
            "action_schema_id": LIFT_ACTION_SCHEMA_ID,
            "reward_version": LIFT_REWARD_VERSION,
            "reward_config_digest": reward_config_digest(self.environment.config),
            "success_criteria_version": success_criteria_version(self.environment.config),
            "physics_config_digest": lift_config_digest(self.environment.config),
            "bootstrap_version": LIFT_BOOTSTRAP_VERSION,
            "bootstrap_config": {
                "stable_grasp_jaw": self.environment.config.stable_grasp_jaw,
                "minimum_blockage": self.environment.config.minimum_blockage,
                "stable_steps": self.environment.config.bootstrap_stable_steps,
            },
            "target_lift": self.environment.config.target_lift,
            "network_config": {"type": "mlp_actor_critic", "hidden_sizes": [64, 64],
                               "activation": "tanh", "recurrent": False},
            "fixed_observation_normalization": True,
            "training_transitions_total": self.training_steps,
            "parent_checkpoint": self.parent_checkpoint,
        })
        temporary = path.with_name(".%s.augment.tmp" % path.name)
        torch.save(payload, temporary); temporary.replace(path)
        return path

    def load_checkpoint(self, checkpoint_path: Path):
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        expected = {
            "task_id": "s1_lift",
            "observation_schema_id": LIFT_OBSERVATION_SCHEMA_ID,
            "action_schema_id": LIFT_ACTION_SCHEMA_ID,
            "action_dim": 1,
            "reward_version": LIFT_REWARD_VERSION,
            "reward_config_digest": reward_config_digest(self.environment.config),
            "success_criteria_version": success_criteria_version(self.environment.config),
            "physics_config_digest": lift_config_digest(self.environment.config),
            "bootstrap_version": LIFT_BOOTSTRAP_VERSION,
            "target_lift": self.environment.config.target_lift,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError("Lift checkpoint %s incompatible" % name)
        self.parent_checkpoint = payload.get("parent_checkpoint")
        return super().load_checkpoint(checkpoint_path)

    def load_hold_extension_checkpoint(self, checkpoint_path: Path,
                                       expected_sha256: str = None):
        """Load P3-1 while permitting only hold duration and horizon changes."""
        checkpoint_path = Path(checkpoint_path)
        if expected_sha256:
            actual = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise ValueError("P3-1 checkpoint SHA-256 mismatch")
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        fixed = {
            "task_id": "s1_lift", "observation_schema_id": LIFT_OBSERVATION_SCHEMA_ID,
            "action_schema_id": LIFT_ACTION_SCHEMA_ID, "action_dim": 1,
            "reward_version": LIFT_REWARD_VERSION,
            "success_criteria_version": LIFT_SUCCESS_CRITERIA_VERSION,
            "bootstrap_version": LIFT_BOOTSTRAP_VERSION, "target_lift": 0.060,
        }
        for name, value in fixed.items():
            if payload.get(name) != value:
                raise ValueError("P3-1 hold-extension checkpoint %s incompatible" % name)
        source = dict(payload.get("environment_config") or {})
        current = asdict(self.environment.config)
        for name in ("hold_seconds", "max_episode_seconds"):
            source.pop(name, None); current.pop(name, None)
        if source != current:
            raise ValueError("P3-2 replay changes more than hold duration/horizon")
        if self.environment.config.hold_seconds != 5.0:
            raise ValueError("hold extension loader requires a 5 second environment")
        return PPOTrainer.load_checkpoint(self, checkpoint_path)


def deterministic_evaluate(environment, trainer: ExperimentalLiftPPOTrainer, episodes=50):
    if episodes < 1:
        raise ValueError("evaluation episodes must be positive")
    trainer.policy.eval(); records = []
    for episode in range(episodes):
        observation = environment.reset(); done = False; reward_sum = 0.0; steps = 0
        while not done:
            raw = flatten_observation(observation, LIFT_POLICY_SCHEMA).to(trainer.device)
            normalized = trainer.normalizer.normalize(raw, update=False)
            with torch.no_grad():
                action, _, _ = trainer.policy.act(normalized, deterministic=True)
            observation, reward, done, info = environment.step(
                tuple(float(v) for v in action.detach().cpu()))
            reward_sum += float(reward); steps += 1
            if steps > environment.max_steps + 1:
                raise RuntimeError("Lift evaluation exceeded horizon")
        records.append({
            "episode": episode, "success": bool(info["success"]),
            "failure_reason": str(info.get("failure_reason", "")),
            "reward": reward_sum, "steps": steps,
            **{name: info.get(name) for name in (
                "cube_rise", "maximum_cube_rise", "target_reached", "hold_duration",
                "hold_half_second", "hold_1s", "hold_3s", "hold_5s",
                "relative_translation_drift", "relative_rotation_drift",
                "drop", "grasp_lost", "target_overshoot", "unsafe", "non_finite",
                "workspace_violation", "jaw_safe", "stable_grasp_valid",
                "blockage_gap", "jaw_actual", "reached_20mm", "reached_40mm", "bootstrap",
            )},
        })
    count = len(records); successes = sum(x["success"] for x in records)
    wilson_lower, wilson_upper = wilson_confidence_interval(successes, count)
    histogram = {}
    for row in records:
        key = row["failure_reason"] or "SUCCESS"
        histogram[key] = histogram.get(key, 0) + 1
    summary = {
        "deterministic": True, "episodes": count,
        "success_count": successes, "success_rate": successes / count,
        "wilson_95_lower": wilson_lower, "wilson_95_upper": wilson_upper,
        "target_reached_count": sum(bool(x["target_reached"]) for x in records),
        "hold_1s_count": successes,
        "drop_count": sum(bool(x["drop"]) for x in records),
        "grasp_lost_count": sum(bool(x["grasp_lost"]) for x in records),
        "timeout_count": histogram.get("TIMEOUT", 0),
        "unsafe_count": sum(bool(x["unsafe"]) for x in records),
        "non_finite_count": sum(bool(x["non_finite"]) for x in records),
        "target_overshoot_count": sum(bool(x["target_overshoot"]) for x in records),
        "workspace_violation_count": sum(bool(x["workspace_violation"]) for x in records),
        "mean_maximum_cube_rise": sum(float(x["maximum_cube_rise"]) for x in records) / count,
        "maximum_cube_rise": max(float(x["maximum_cube_rise"]) for x in records),
        "mean_final_cube_rise": sum(float(x["cube_rise"]) for x in records) / count,
        "mean_hold_duration": sum(float(x["hold_duration"]) for x in records) / count,
        "mean_relative_translation_drift": sum(
            float(x["relative_translation_drift"]) for x in records) / count,
        "max_relative_translation_drift": max(
            float(x["relative_translation_drift"]) for x in records),
        "mean_rotation_drift": sum(float(x["relative_rotation_drift"]) for x in records) / count,
        "max_rotation_drift": max(float(x["relative_rotation_drift"]) for x in records),
        "failure_reason_histogram": histogram,
        "behavior_funnel": {
            "episodes": count, "stable_grasp_reset": count,
            "cube_lifted_20mm": sum(bool(x["reached_20mm"]) for x in records),
            "cube_lifted_40mm": sum(bool(x["reached_40mm"]) for x in records),
            "target_band_reached": sum(bool(x["target_reached"]) for x in records),
            "hold_0p5s": sum(bool(x["hold_half_second"]) for x in records),
            "hold_1p0s": sum(bool(x["hold_1s"]) for x in records),
            "hold_3p0s": sum(bool(x["hold_3s"]) for x in records),
            "hold_5p0s": sum(bool(x["hold_5s"]) for x in records),
            "success": successes,
        },
        "resource_cleanup": True,
    }
    summary["hold_1s_count"] = summary["behavior_funnel"]["hold_1p0s"]
    summary["hold_3s_count"] = summary["behavior_funnel"]["hold_3p0s"]
    summary["hold_5s_count"] = summary["behavior_funnel"]["hold_5p0s"]
    if not all(math.isfinite(float(v)) for v in summary.values()
               if isinstance(v, (int, float))):
        raise RuntimeError("Lift evaluation produced non-finite summary")
    return summary, records
