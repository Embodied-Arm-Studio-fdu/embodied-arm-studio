"""PPO, checkpoint, transfer, and evaluation support for P4-2."""

import hashlib
import json
import math
from pathlib import Path

import torch

from ...evaluation import wilson_confidence_interval
from ...ppo.normalizer import ObservationNormalizer
from ...ppo.policy import flatten_observation
from ...ppo.trainer import PPOTaskContract, PPOTrainer
from .lower_release_env import (
    LOWER_RELEASE_BOOTSTRAP_VERSION, LOWER_RELEASE_TARGET,
    lower_release_config_digest,
)
from .lower_release_schema import (
    LOWER_RELEASE_ACTIVE_ACTION_MASK, LOWER_RELEASE_POLICY_SCHEMA,
    LOWER_RELEASE_REWARD_SCHEMA_VERSION, LOWER_RELEASE_REWARD_SEMANTICS,
    LOWER_RELEASE_REWARD_TERM_NAMES, LOWER_RELEASE_REWARD_VERSION,
    LOWER_RELEASE_SUCCESS_CRITERIA_VERSION, reward_schema_metadata,
)
from .place_schema import PLACE_ACTION_SCHEMA_ID, PLACE_OBSERVATION_SCHEMA_ID


P4_1_CHECKPOINT_SHA256 = "eb9fab2287ffd6a735dc27a1ccccd8eb3c71678617b4aef2bac1561d556cf291"

LOWER_RELEASE_PPO_CONTRACT = PPOTaskContract(
    LOWER_RELEASE_POLICY_SCHEMA, LOWER_RELEASE_REWARD_SCHEMA_VERSION,
    LOWER_RELEASE_REWARD_SEMANTICS, LOWER_RELEASE_REWARD_TERM_NAMES,
    reward_schema_metadata,
)


def reward_config_digest(config):
    encoded = json.dumps(reward_schema_metadata(config), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def fixed_observation_normalizer():
    # Preserve the exact P4-1 v1 observation preprocessing contract.
    return ObservationNormalizer(dimension=32, previous_action_slice=slice(0, 32))


class ExperimentalLowerReleasePPOTrainer(PPOTrainer):
    def __init__(self, environment, **kwargs):
        kwargs.setdefault("task_contract", LOWER_RELEASE_PPO_CONTRACT)
        kwargs.setdefault("normalizer", fixed_observation_normalizer())
        super().__init__(environment, **kwargs)
        self.parent_checkpoint = None

    def initialize_from_p4_1(self, checkpoint_path):
        path = Path(checkpoint_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != P4_1_CHECKPOINT_SHA256:
            raise ValueError("P4-1 checkpoint SHA-256 mismatch")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        expected = {
            "task_id": "s1_place", "observation_schema_id": PLACE_OBSERVATION_SCHEMA_ID,
            "action_schema_id": PLACE_ACTION_SCHEMA_ID, "observation_dim": 32,
            "action_dim": 4, "active_action_mask": (1, 1, 0, 0),
            "target_center": LOWER_RELEASE_TARGET,
        }
        for name, value in expected.items():
            actual = payload.get(name)
            if isinstance(value, tuple) and actual is not None:
                actual = tuple(actual)
            if actual != value:
                raise ValueError("P4-1 transfer checkpoint incompatible: %s" % name)
        self.policy.load_state_dict(payload["policy"])
        self.normalizer.load_state_dict(payload["normalizer_state"], frozen=False)
        # P4-1 masked dz/grip at the environment boundary, so those two actor
        # rows received no task-grounded signal.  Retain the shared trunk,
        # critic, and trained checkpoint exactly, while giving only the newly
        # activated rows a conservative descend/closed-jaw learning prior.
        output = self.policy.actor[-1]
        with torch.no_grad():
            output.weight[2:].zero_()
            output.bias[2] = -0.50
            output.bias[3] = -1.20
            self.policy.log_std[2:] = -0.70
        # New reward/stage means a fresh optimizer and counters. The complete
        # actor/critic weights and observation statistics are transferred.
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.training_steps = self.update_steps = 0
        self.parent_checkpoint = {"path": str(path), "sha256": digest,
                                  "training_steps": int(payload["training_steps"]),
                                  "transfer_strategy": (
                                      "shared actor trunk, critic, normalizer, and dx/dy rows; "
                                      "newly active dz/grip output rows safety-initialized")}
        return payload

    def _finish_episode(self, info, final_distance):
        record = super()._finish_episode(info, final_distance)
        record.update({name: info.get(name) for name in (
            "failure_reason", "xy_error", "descended_to_release_height",
            "transport_reached", "legal_release_height", "release_triggered",
            "auto_release_ready",
            "release_started", "release_latched", "actual_jaw_opened",
            "detached", "table_contact", "inside_target_zone",
            "settled_0p5s", "settled_1p0s", "settle_duration", "release_height",
            "detach_height", "bottom_clearance", "minimum_bottom_clearance",
            "cube_linear_speed", "cube_angular_speed", "cube_tilt", "jaw_actual",
            "grasp_valid", "early_release", "release_too_high",
            "drop_outside_target", "object_unstable", "workspace_violation",
            "table_penetration", "target_overshoot", "joint_limit", "unsafe",
            "non_finite", "active_action_mask", "bootstrap")})
        return record

    @staticmethod
    def _aggregate_episode_records(records):
        result = PPOTrainer._aggregate_episode_records(records)
        if not records:
            result.update({"descended_rate": 0.0, "detached_rate": 0.0,
                           "settled_1s_rate": 0.0})
            return result
        count = len(records)
        result.update({
            "descended_rate": sum(bool(x.get("descended_to_release_height")) for x in records) / count,
            "release_started_rate": sum(bool(x.get("release_started")) for x in records) / count,
            "release_latched_rate": sum(bool(x.get("release_latched")) for x in records) / count,
            "actual_jaw_opened_rate": sum(bool(x.get("actual_jaw_opened")) for x in records) / count,
            "detached_rate": sum(bool(x.get("detached")) for x in records) / count,
            "table_contact_rate": sum(bool(x.get("table_contact")) for x in records) / count,
            "settled_1s_rate": sum(bool(x.get("settled_1p0s")) for x in records) / count,
            "mean_final_xy_error": sum(float(x.get("xy_error", 0.0)) for x in records) / count,
        })
        return result

    def save_checkpoint(self, artifacts_root, project_id, run_id,
                        filename="model_latest.pt"):
        path = super().save_checkpoint(artifacts_root, project_id, run_id, filename)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update({
            "observation_schema_id": PLACE_OBSERVATION_SCHEMA_ID,
            "action_schema_id": PLACE_ACTION_SCHEMA_ID,
            "active_action_mask": LOWER_RELEASE_ACTIVE_ACTION_MASK,
            "reward_version": LOWER_RELEASE_REWARD_VERSION,
            "reward_config_digest": reward_config_digest(self.environment.config),
            "success_criteria_version": LOWER_RELEASE_SUCCESS_CRITERIA_VERSION,
            "physics_config_digest": lower_release_config_digest(self.environment.config),
            "bootstrap_version": LOWER_RELEASE_BOOTSTRAP_VERSION,
            "target_center": LOWER_RELEASE_TARGET,
            "network_config": {"type": "mlp_actor_critic", "hidden_sizes": [64, 64],
                               "activation": "tanh", "recurrent": False},
            "fixed_observation_normalization": True,
            "stage_training_transitions": self.training_steps,
            "release_semantics": "policy_trigger_latched_open",
            "parent_checkpoint": self.parent_checkpoint,
        })
        temporary = path.with_name(".%s.augment.tmp" % path.name)
        torch.save(payload, temporary); temporary.replace(path)
        return path

    def load_checkpoint(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        expected = {
            "task_id": "s1_place", "observation_schema_id": PLACE_OBSERVATION_SCHEMA_ID,
            "action_schema_id": PLACE_ACTION_SCHEMA_ID, "action_dim": 4,
            "active_action_mask": LOWER_RELEASE_ACTIVE_ACTION_MASK,
            "reward_version": LOWER_RELEASE_REWARD_VERSION,
            "reward_config_digest": reward_config_digest(self.environment.config),
            "success_criteria_version": LOWER_RELEASE_SUCCESS_CRITERIA_VERSION,
            "physics_config_digest": lower_release_config_digest(self.environment.config),
            "bootstrap_version": LOWER_RELEASE_BOOTSTRAP_VERSION,
            "target_center": LOWER_RELEASE_TARGET,
            "release_semantics": "policy_trigger_latched_open",
        }
        for name, value in expected.items():
            actual = payload.get(name)
            if isinstance(value, tuple) and actual is not None:
                actual = tuple(actual)
            if actual != value:
                raise ValueError("P4-2 checkpoint %s incompatible" % name)
        self.parent_checkpoint = payload.get("parent_checkpoint")
        return super().load_checkpoint(Path(checkpoint_path))


def deterministic_evaluate(environment, trainer, episodes=50):
    trainer.policy.eval(); records = []
    fields = (
        "xy_error", "transport_reached", "descended_to_release_height",
        "legal_release_height", "auto_release_ready", "release_triggered", "release_started",
        "release_latched", "actual_jaw_opened", "detached",
        "table_contact", "inside_target_zone", "settled_0p5s", "settled_1p0s",
        "settle_duration", "release_height", "detach_height", "bottom_clearance",
        "minimum_bottom_clearance", "cube_linear_speed", "cube_angular_speed",
        "cube_tilt", "jaw_actual", "grasp_valid", "early_release",
        "release_too_high", "drop_outside_target", "object_unstable",
        "workspace_violation", "table_penetration", "target_overshoot",
        "joint_limit", "unsafe", "non_finite", "bootstrap",
    )
    for episode in range(episodes):
        observation = environment.reset(); done = False; reward_sum = 0.0; steps = 0
        while not done:
            raw = flatten_observation(observation, LOWER_RELEASE_POLICY_SCHEMA).to(trainer.device)
            normalized = trainer.normalizer.normalize(raw, update=False)
            with torch.no_grad():
                action, _, _ = trainer.policy.act(normalized, deterministic=True)
            observation, reward, done, info = environment.step(
                tuple(float(value) for value in action.detach().cpu()))
            reward_sum += float(reward); steps += 1
            if steps > environment.max_steps + 1:
                raise RuntimeError("P4-2 evaluation exceeded horizon")
        records.append({"episode": episode, "success": bool(info["success"]),
                        "failure_reason": str(info.get("failure_reason", "")),
                        "reward": reward_sum, "steps": steps,
                        **{name: info.get(name) for name in fields}})
    count = len(records); successes = sum(row["success"] for row in records)
    lower, upper = wilson_confidence_interval(successes, count)
    histogram = {}
    for row in records:
        key = row["failure_reason"] or "SUCCESS"
        histogram[key] = histogram.get(key, 0) + 1
    def tally(name): return sum(bool(row.get(name)) for row in records)
    release_heights = [float(row["release_height"]) for row in records
                       if row.get("release_height") is not None]
    summary = {
        "deterministic": True, "episodes": count, "success_count": successes,
        "success_rate": successes / count,
        "wilson_95_lower": lower, "wilson_95_upper": upper,
        "valid_target_above_reset_count": count,
        "descended_to_release_height_count": tally("descended_to_release_height"),
        "transport_reached_count": tally("transport_reached"),
        "legal_release_height_count": tally("legal_release_height"),
        "auto_release_ready_count": tally("auto_release_ready"),
        "release_triggered_count": tally("release_triggered"),
        "release_started_count": tally("release_started"),
        "release_latched_count": tally("release_latched"),
        "actual_jaw_opened_count": tally("actual_jaw_opened"),
        "detached_count": tally("detached"), "table_contact_count": tally("table_contact"),
        "inside_target_zone_count": tally("inside_target_zone"),
        "settled_0p5s_count": tally("settled_0p5s"),
        "settled_1p0s_count": tally("settled_1p0s"),
        "early_release_count": tally("early_release"),
        "release_too_high_count": tally("release_too_high"),
        "drop_outside_target_count": tally("drop_outside_target"),
        "unstable_placement_count": tally("object_unstable"),
        "table_penetration_count": tally("table_penetration"),
        "workspace_violation_count": tally("workspace_violation"),
        "target_overshoot_count": tally("target_overshoot"),
        "joint_limit_count": tally("joint_limit"), "unsafe_count": tally("unsafe"),
        "non_finite_count": tally("non_finite"),
        "timeout_count": histogram.get("TIMEOUT", 0),
        "mean_release_height": (sum(release_heights) / len(release_heights)
                                if release_heights else None),
        "mean_final_xy_error": sum(float(row["xy_error"]) for row in records) / count,
        "mean_final_linear_velocity": sum(float(row["cube_linear_speed"])
                                           for row in records) / count,
        "mean_final_angular_velocity": sum(float(row["cube_angular_speed"])
                                            for row in records) / count,
        "minimum_bottom_clearance": min(float(row["minimum_bottom_clearance"])
                                         for row in records),
        "failure_reason_histogram": histogram,
        "behavior_funnel": {
            "episodes": count, "valid_target_above_reset": count,
            "transport_reached": tally("transport_reached"),
            "legal_release_height": tally("legal_release_height"),
            "auto_release_ready": tally("auto_release_ready"),
            "release_triggered": tally("release_triggered"),
            "release_latched": tally("release_latched"),
            "actual_jaw_opened": tally("actual_jaw_opened"),
            "descended_to_release_height": tally("descended_to_release_height"),
            "release_started": tally("release_started"), "cube_detached": tally("detached"),
            "cube_contacts_table": tally("table_contact"),
            "inside_target_zone": tally("inside_target_zone"),
            "settled_0p5s": tally("settled_0p5s"),
            "settled_1p0s": tally("settled_1p0s"), "success": successes,
        },
        "target_center": list(LOWER_RELEASE_TARGET), "resource_cleanup": True,
    }
    if not all(math.isfinite(float(value)) for value in summary.values()
               if isinstance(value, (int, float))):
        raise RuntimeError("P4-2 evaluation produced non-finite summary")
    return summary, records
