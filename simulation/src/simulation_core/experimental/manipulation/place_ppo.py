"""PPO/checkpoint/evaluation support for experimental S1 Place transport."""

import hashlib
import json
import math
from pathlib import Path

import torch

from ...evaluation import wilson_confidence_interval
from ...ppo.normalizer import ObservationNormalizer
from ...ppo.policy import flatten_observation
from ...ppo.trainer import PPOTaskContract, PPOTrainer
from .place_env import (PLACE_BOOTSTRAP_VERSION, PLACE_TARGET_SELECTION_VERSION,
                        place_config_digest)
from .place_schema import (
    PLACE_ACTION_SCHEMA_ID, PLACE_ACTIVE_ACTION_MASK, PLACE_OBSERVATION_DIM,
    PLACE_OBSERVATION_SCHEMA_ID, PLACE_POLICY_SCHEMA, PLACE_REWARD_SCHEMA_VERSION,
    PLACE_REWARD_SEMANTICS, PLACE_REWARD_TERM_NAMES, PLACE_REWARD_VERSION,
    PLACE_SUCCESS_CRITERIA_VERSION, reward_schema_metadata,
)


PLACE_PPO_CONTRACT = PPOTaskContract(
    PLACE_POLICY_SCHEMA, PLACE_REWARD_SCHEMA_VERSION, PLACE_REWARD_SEMANTICS,
    PLACE_REWARD_TERM_NAMES, reward_schema_metadata)


def reward_config_digest(config):
    encoded = json.dumps(reward_schema_metadata(config), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def fixed_observation_normalizer():
    return ObservationNormalizer(dimension=PLACE_OBSERVATION_DIM,
                                 previous_action_slice=slice(0, PLACE_OBSERVATION_DIM))


class ExperimentalPlacePPOTrainer(PPOTrainer):
    def __init__(self, environment, **kwargs):
        kwargs.setdefault("task_contract", PLACE_PPO_CONTRACT)
        kwargs.setdefault("normalizer", fixed_observation_normalizer())
        super().__init__(environment, **kwargs)

    def _finish_episode(self, info, final_distance):
        record = super()._finish_episode(info, final_distance)
        record.update({name: info.get(name) for name in (
            "failure_reason", "xy_error", "minimum_xy_error", "target_reached",
            "hold_duration", "hold_half_second", "reached_50mm_error",
            "reached_30mm_error", "transport_time", "cube_rise",
            "cube_height_variation", "relative_translation_drift",
            "relative_rotation_drift", "drop", "grasp_lost", "workspace_violation",
            "table_collision", "target_overshoot", "unsafe", "non_finite",
            "active_action_mask", "bootstrap")})
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
            "mean_minimum_xy_error": sum(float(x.get("minimum_xy_error", 0))
                                          for x in records) / count,
            "mean_final_xy_error": sum(float(x.get("xy_error", 0)) for x in records) / count,
            "mean_hold_duration": sum(float(x.get("hold_duration", 0)) for x in records) / count,
        })
        return result

    def save_checkpoint(self, artifacts_root, project_id, run_id,
                        filename="model_latest.pt"):
        path = super().save_checkpoint(artifacts_root, project_id, run_id, filename)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if self.environment.target_center is None:
            raise RuntimeError("Place target must be selected before checkpoint save")
        payload.update({
            "observation_schema_id": PLACE_OBSERVATION_SCHEMA_ID,
            "action_schema_id": PLACE_ACTION_SCHEMA_ID,
            "active_action_mask": PLACE_ACTIVE_ACTION_MASK,
            "reward_version": PLACE_REWARD_VERSION,
            "reward_config_digest": reward_config_digest(self.environment.config),
            "success_criteria_version": PLACE_SUCCESS_CRITERIA_VERSION,
            "physics_config_digest": place_config_digest(self.environment.config),
            "bootstrap_version": PLACE_BOOTSTRAP_VERSION,
            "target_selection_version": PLACE_TARGET_SELECTION_VERSION,
            "target_center": self.environment.target_center,
            "target_selection": self.environment.target_selection,
            "network_config": {"type": "mlp_actor_critic", "hidden_sizes": [64, 64],
                               "activation": "tanh", "recurrent": False},
            "fixed_observation_normalization": True,
            "training_transitions_total": self.training_steps,
        })
        temporary = path.with_name(".%s.augment.tmp" % path.name)
        torch.save(payload, temporary); temporary.replace(path)
        return path

    def load_checkpoint(self, checkpoint_path, *, evaluation_only=False):
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        expected = {"task_id": "s1_place", "observation_schema_id": PLACE_OBSERVATION_SCHEMA_ID,
                    "action_schema_id": PLACE_ACTION_SCHEMA_ID, "action_dim": 4,
                    "active_action_mask": PLACE_ACTIVE_ACTION_MASK,
                    "reward_version": PLACE_REWARD_VERSION,
                    "reward_config_digest": reward_config_digest(self.environment.config),
                    "success_criteria_version": PLACE_SUCCESS_CRITERIA_VERSION,
                    "physics_config_digest": place_config_digest(self.environment.config),
                    "bootstrap_version": PLACE_BOOTSTRAP_VERSION,
                    "target_selection_version": PLACE_TARGET_SELECTION_VERSION}
        for name, value in expected.items():
            actual = tuple(payload.get(name)) if name == "active_action_mask" else payload.get(name)
            if actual != value:
                raise ValueError("Place checkpoint %s incompatible" % name)
        target = payload.get("target_center")
        if not isinstance(target, (tuple, list)) or len(target) != 3:
            raise ValueError("Place checkpoint target_center incompatible")
        self.environment.target_center = tuple(float(v) for v in target)
        self.environment.target_selection = payload.get("target_selection")
        if evaluation_only:
            from ...ppo.trainer import (validate_checkpoint_action_schema,
                                       validate_checkpoint_observation_schema,
                                       validate_checkpoint_reward_schema)
            validate_checkpoint_action_schema(payload, self.task_contract)
            validate_checkpoint_observation_schema(payload, self.task_contract)
            validate_checkpoint_reward_schema(payload, self.environment.config, self.task_contract)
            self.policy.load_state_dict(payload["policy"])
            self.normalizer.load_state_dict(payload["normalizer_state"], frozen=True)
            self.policy.eval()
            return payload
        return super().load_checkpoint(checkpoint_path)


def deterministic_evaluate(environment, trainer, episodes=50):
    trainer.policy.eval(); records = []
    for episode in range(episodes):
        observation = environment.reset(); done = False; reward_sum = 0.0; steps = 0
        action_norm_sum = saturation_sum = 0.0
        while not done:
            raw = flatten_observation(observation, PLACE_POLICY_SCHEMA).to(trainer.device)
            normalized = trainer.normalizer.normalize(raw, update=False)
            with torch.no_grad():
                action, _, _ = trainer.policy.act(normalized, deterministic=True)
            observation, reward, done, info = environment.step(
                tuple(float(v) for v in action.detach().cpu()))
            reward_sum += float(reward); steps += 1
            action_norm_sum += float(info["action_norm"])
            saturation_sum += float(info["saturation_rate"])
            if steps > environment.max_steps + 1:
                raise RuntimeError("Place evaluation exceeded horizon")
        records.append({"episode": episode, "success": bool(info["success"]),
                        "failure_reason": str(info.get("failure_reason", "")),
                        "reward": reward_sum, "steps": steps,
                        "action_norm": action_norm_sum / steps,
                        "action_saturation_rate": saturation_sum / steps,
                        "object_displacement": float(info["object_displacement"]),
                        **{name: info.get(name) for name in (
                            "xy_error", "minimum_xy_error", "target_reached",
                            "hold_duration", "hold_half_second", "reached_50mm_error",
                            "reached_30mm_error", "transport_time", "cube_rise",
                            "cube_height_variation", "relative_translation_drift",
                            "relative_rotation_drift", "drop", "grasp_lost",
                            "workspace_violation", "table_collision", "target_overshoot",
                            "unsafe", "non_finite", "bootstrap")}})
    count = len(records); successes = sum(x["success"] for x in records)
    lower, upper = wilson_confidence_interval(successes, count)
    histogram = {}
    for row in records:
        key = row["failure_reason"] or "SUCCESS"
        histogram[key] = histogram.get(key, 0) + 1
    reached = sum(bool(x["target_reached"]) for x in records)
    summary = {"deterministic": True, "episodes": count,
               "success_count": successes, "success_rate": successes / count,
               "wilson_95_lower": lower, "wilson_95_upper": upper,
               "target_reached_count": reached,
               "hold_0p5s_count": sum(bool(x["hold_half_second"]) for x in records),
               "hold_1s_count": successes,
               "drop_count": sum(bool(x["drop"]) for x in records),
               "grasp_lost_count": sum(bool(x["grasp_lost"]) for x in records),
               "workspace_violation_count": sum(bool(x["workspace_violation"]) for x in records),
               "table_collision_count": sum(bool(x["table_collision"]) for x in records),
               "target_overshoot_count": sum(bool(x["target_overshoot"]) for x in records),
               "timeout_count": histogram.get("TIMEOUT", 0),
               "unsafe_count": sum(bool(x["unsafe"]) for x in records),
               "non_finite_count": sum(bool(x["non_finite"]) for x in records),
               "mean_minimum_xy_error": sum(float(x["minimum_xy_error"]) for x in records) / count,
               "mean_final_xy_error": sum(float(x["xy_error"]) for x in records) / count,
               "mean_transport_time": (sum(float(x["transport_time"]) for x in records
                                           if x["transport_time"] is not None)
                                       / max(1, sum(x["transport_time"] is not None for x in records))),
               "mean_cube_height_variation": sum(float(x["cube_height_variation"])
                                                  for x in records) / count,
               "mean_hold_duration": sum(float(x["hold_duration"]) for x in records) / count,
               "mean_translation_drift": sum(float(x["relative_translation_drift"])
                                               for x in records) / count,
               "max_translation_drift": max(float(x["relative_translation_drift"])
                                              for x in records),
               "mean_rotation_drift": sum(float(x["relative_rotation_drift"])
                                            for x in records) / count,
               "max_rotation_drift": max(float(x["relative_rotation_drift"])
                                           for x in records),
               "failure_reason_histogram": histogram,
               "behavior_funnel": {"episodes": count, "valid_grasped_lifted_reset": count,
                   "xy_error_lt_50mm": sum(bool(x["reached_50mm_error"]) for x in records),
                   "xy_error_lt_30mm": sum(bool(x["reached_30mm_error"]) for x in records),
                   "target_zone_le_15mm": reached,
                   "hold_0p5s": sum(bool(x["hold_half_second"]) for x in records),
                   "hold_1p0s": successes, "success": successes},
               "target_center": list(environment.target_center), "resource_cleanup": True}
    if not all(math.isfinite(float(v)) for v in summary.values()
               if isinstance(v, (int, float))):
        raise RuntimeError("Place evaluation produced non-finite summary")
    return summary, records


def canonical_place_evaluation(domain_summary, domain_records, checkpoint_sha256):
    """Adapt measured Place episodes to the platform's existing evidence contract.

    As in the manipulation evaluator, seed identifies an episode's position in
    the fixed reset sequence; it is not a randomized scene seed.
    """
    records = []
    for index, row in enumerate(domain_records):
        timeout = str(row["failure_reason"]).startswith("TIMEOUT")
        records.append({**row, "seed": index, "task_id": "s1_place",
                        "timeout": timeout, "failure": not row["success"] and not timeout,
                        "episode_return": row["reward"], "episode_length": row["steps"],
                        "final_distance": row["xy_error"], "minimum_distance": row["minimum_xy_error"]})
    count = len(records)
    if not count:
        raise ValueError("Place evaluation requires episode evidence")
    successes = sum(row["success"] for row in records)
    lower, upper = wilson_confidence_interval(successes, count)
    def average(name):
        return sum(float(row[name]) for row in records) / count
    summary = {
        "task_id": "s1_place", "deterministic": True, "episode_count": count,
        "success_count": successes, "success_rate": successes / count,
        "confidence_interval_95": {"method": "wilson", "lower": lower, "upper": upper},
        "timeout_rate": average("timeout"), "failure_rate": average("failure"),
        "mean_episode_length": average("episode_length"), "mean_episode_return": average("episode_return"),
        "mean_final_distance": average("final_distance"), "mean_minimum_distance": average("minimum_distance"),
        "mean_object_displacement": average("object_displacement"), "mean_action_norm": average("action_norm"),
        "action_saturation_rate": average("action_saturation_rate"),
        "checkpoint_sha256": checkpoint_sha256,
        "selected_success_seed": next((row["seed"] for row in records if row["success"]), None),
        "seed_semantics": "episode_index_in_fixed_reset_sequence",
        "evaluation_video": {"status": "unavailable", "failure_reason": "replay_not_attempted"},
        "task_metrics": domain_summary,
    }
    return summary, records
