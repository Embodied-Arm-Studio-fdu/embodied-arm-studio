"""PPO contract, checkpoint metadata, and deterministic evaluation for P2."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping

import torch

from ...evaluation import wilson_confidence_interval
from ...ppo.normalizer import ObservationNormalizer
from ...ppo.policy import flatten_observation
from ...ppo.trainer import PPOTaskContract, PPOTrainer
from .grasp_env import (
    GRASP_VERIFIER_IMPLEMENTATION_REVISION, GraspEnvConfig,
    physics_config_digest,
)
from .grasp_schema import (
    GRASP_ACTION_SCHEMA_ID, GRASP_OBSERVATION_DIM, GRASP_OBSERVATION_SCHEMA_ID,
    GRASP_POLICY_SCHEMA, GRASP_REWARD_SCHEMA_VERSION, GRASP_REWARD_SEMANTICS,
    GRASP_REWARD_TERM_NAMES, GRASP_REWARD_VERSION,
    GRASP_SUCCESS_CRITERIA_VERSION, reward_schema_metadata,
)


GRASP_PPO_CONTRACT = PPOTaskContract(
    GRASP_POLICY_SCHEMA, GRASP_REWARD_SCHEMA_VERSION,
    GRASP_REWARD_SEMANTICS, GRASP_REWARD_TERM_NAMES, reward_schema_metadata,
)


def reward_config_digest(config):
    encoded = json.dumps(
        reward_schema_metadata(config), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def fixed_observation_normalizer():
    # The environment emits fixed physically normalized values.  An all-false
    # running mask makes the shared normalizer an identity transform while its
    # versioned state remains checkpointed and cross-load validated.
    return ObservationNormalizer(
        dimension=GRASP_OBSERVATION_DIM,
        previous_action_slice=slice(0, GRASP_OBSERVATION_DIM),
    )


class ExperimentalGraspPPOTrainer(PPOTrainer):
    def __init__(self, environment, **kwargs):
        kwargs.setdefault("task_contract", GRASP_PPO_CONTRACT)
        kwargs.setdefault("normalizer", fixed_observation_normalizer())
        super().__init__(environment, **kwargs)
        self.curriculum_parent_checkpoint_sha256 = None
        self.curriculum_stage_start_steps = 0
        self.initial_best_checkpoint_rank = None

    def initialize_from_p2_r8(self, checkpoint_path, expected_sha256):
        """Start production training from the formally verified P2-R8 policy."""
        checkpoint_path = Path(checkpoint_path)
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("P2-R8 Grasp checkpoint SHA-256 mismatch")
        payload = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False,
        )
        expected = {
            "task_id": "s1_grasp",
            "observation_schema_id": GRASP_OBSERVATION_SCHEMA_ID,
            "action_schema_id": GRASP_ACTION_SCHEMA_ID,
            "reward_version": GRASP_REWARD_VERSION,
            "success_criteria_version": GRASP_SUCCESS_CRITERIA_VERSION,
            "verifier_implementation_revision": GRASP_VERIFIER_IMPLEMENTATION_REVISION,
            "curriculum_stage": self.environment.config.curriculum_stage,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError("P2-R8 Grasp checkpoint %s mismatch" % name)
        source = dict(payload.get("environment_config") or {})
        current = asdict(self.environment.config)
        for name in ("randomize_xy", "random_seed"):
            source.pop(name, None)
            current.pop(name, None)
        if source != current:
            raise ValueError(
                "P2-R8 Grasp checkpoint changes more than XY randomization/seed"
            )
        self.policy.load_state_dict(payload["policy"])
        self.normalizer.load_state_dict(payload["normalizer_state"], frozen=False)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate,
        )
        self.training_steps = self.update_steps = 0
        self.curriculum_parent_checkpoint_sha256 = digest
        self.curriculum_stage_start_steps = 0
        # The locked checkpoint passed 98/100 randomized episodes with a
        # 0.92999 Wilson lower bound.  Training-only reward must not replace
        # this Run-local best policy without deterministic evaluation evidence.
        self.initial_best_checkpoint_rank = (1.0, 0.98, 0.92999, 0.0, 0.0)
        return payload

    def _finish_episode(self, info, final_distance):
        record = super()._finish_episode(info, final_distance)
        record.update({
            "failure_reason": str(info.get("failure_reason", "")),
            "grasp_candidate": bool(info.get("grasp_candidate", False)),
            "verified_success": bool(info.get("verified_success", False)),
            "premature_push": bool(info.get("premature_push", False)),
            "unsafe": bool(info.get("unsafe", False)),
            "non_finite": bool(info.get("non_finite", False)),
            "time_to_grasp": float(info.get("time_to_grasp", 0.0)),
            "verification_lift": float(info.get("verification_lift", 0.0)),
            "relative_drift": float(info.get("relative_drift", 0.0)),
            "candidate_local_ee_rise": float(info.get("candidate_local_ee_rise", 0.0)),
            "reset_relative_ee_rise": float(info.get("reset_relative_ee_rise", 0.0)),
            "cube_rise": float(info.get("cube_rise", 0.0)),
            "lift_ratio": float(info.get("lift_ratio", 0.0)),
            "rotation_drift": float(info.get("rotation_drift", 0.0)),
            "verification_clearance": float(info.get("verification_clearance", 0.0)),
            "time_to_verified": info.get("time_to_verified"),
            "grasp_confidence": float(info.get("grasp_confidence", 0.0)),
            "force_max": float(info.get("force_max", 0.0)),
            "reward_term_totals": dict(info.get("reward_term_totals", {})),
            "minimum_xy_error": float(info.get("minimum_xy_error", 0.0)),
            "final_xy_error": float(info.get("final_xy_error", 0.0)),
            "minimum_z_error": float(info.get("minimum_z_error", 0.0)),
            "final_z_error": float(info.get("final_z_error", 0.0)),
            "aligned": bool(info.get("aligned", False)),
            "descended": bool(info.get("descended", False)),
            "clearance_reached": bool(info.get("clearance_reached", False)),
            "post_clearance_descended": bool(info.get("post_clearance_descended", False)),
            "reset_profile": str(info.get("reset_profile", "")),
            "closure_started": bool(info.get("closure_started", False)),
            "candidate_once": bool(info.get("candidate_once", False)),
            "verification_entered": bool(info.get("verification_entered", False)),
            "candidate_evidence_flags": dict(info.get("candidate_evidence_flags", {})),
            "minimum_commanded_jaw": float(info.get("minimum_commanded_jaw", 0.05)),
            "mean_actual_jaw": float(info.get("mean_actual_jaw", 0.05)),
            "maximum_blockage_gap": float(info.get("maximum_blockage_gap", 0.0)),
            "time_to_contact": info.get("time_to_contact"),
            "time_to_candidate": info.get("time_to_candidate"),
            "jaw_below_040": bool(info.get("jaw_below_040", False)),
            "jaw_below_035": bool(info.get("jaw_below_035", False)),
            "jaw_below_030": bool(info.get("jaw_below_030", False)),
            "blockage_ge_0005": bool(info.get("blockage_ge_0005", False)),
            "blockage_ge_0010": bool(info.get("blockage_ge_0010", False)),
            "blockage_ge_0015": bool(info.get("blockage_ge_0015", False)),
            "blockage_ge_0020": bool(info.get("blockage_ge_0020", False)),
        })
        return record

    @staticmethod
    def _aggregate_episode_records(records):
        result = PPOTrainer._aggregate_episode_records(records)
        if not records:
            result.update({"candidate_rate": 0.0, "verified_success_rate": 0.0,
                           "premature_push_rate": 0.0, "unsafe_rate": 0.0})
            return result
        count = len(records)
        result.update({
            "candidate_rate": sum(item.get("grasp_candidate", False) for item in records) / count,
            "verified_success_rate": sum(item.get("verified_success", False) for item in records) / count,
            "premature_push_rate": sum(item.get("premature_push", False) for item in records) / count,
            "unsafe_rate": sum(item.get("unsafe", False) for item in records) / count,
            "mean_time_to_grasp": sum(item.get("time_to_grasp", 0.0) for item in records) / count,
            "mean_verification_lift": sum(item.get("verification_lift", 0.0) for item in records) / count,
            "mean_relative_drift": sum(item.get("relative_drift", 0.0) for item in records) / count,
            "aligned_rate": sum(item.get("aligned", False) for item in records) / count,
            "descended_rate": sum(item.get("descended", False) for item in records) / count,
            "clearance_reached_rate": sum(
                item.get("clearance_reached", False) for item in records) / count,
            "post_clearance_descended_rate": sum(
                item.get("post_clearance_descended", False) for item in records) / count,
            "closure_started_rate": sum(item.get("closure_started", False) for item in records) / count,
            "verification_entered_rate": sum(item.get("verification_entered", False) for item in records) / count,
            "mean_minimum_xy_error": sum(item.get("minimum_xy_error", 0.0) for item in records) / count,
            "mean_final_xy_error": sum(item.get("final_xy_error", 0.0) for item in records) / count,
            "mean_minimum_z_error": sum(item.get("minimum_z_error", 0.0) for item in records) / count,
            "jaw_below_040_rate": sum(item.get("jaw_below_040", False) for item in records) / count,
            "jaw_below_035_rate": sum(item.get("jaw_below_035", False) for item in records) / count,
            "jaw_below_030_rate": sum(item.get("jaw_below_030", False) for item in records) / count,
            "blockage_ge_0010_rate": sum(item.get("blockage_ge_0010", False) for item in records) / count,
            "blockage_ge_0020_rate": sum(item.get("blockage_ge_0020", False) for item in records) / count,
            "mean_minimum_commanded_jaw": sum(item.get("minimum_commanded_jaw", 0.05) for item in records) / count,
            "mean_actual_jaw": sum(item.get("mean_actual_jaw", 0.05) for item in records) / count,
            "mean_maximum_blockage_gap": sum(item.get("maximum_blockage_gap", 0.0) for item in records) / count,
        })
        return result

    def save_checkpoint(self, artifacts_root: Path, project_id: str, run_id: str,
                        filename: str = "model_latest.pt") -> Path:
        path = super().save_checkpoint(artifacts_root, project_id, run_id, filename)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update({
            "observation_schema_id": GRASP_OBSERVATION_SCHEMA_ID,
            "action_schema_id": GRASP_ACTION_SCHEMA_ID,
            "reward_version": GRASP_REWARD_VERSION,
            "reward_config_digest": reward_config_digest(self.environment.config),
            "success_criteria_version": GRASP_SUCCESS_CRITERIA_VERSION,
            "verifier_implementation_revision": GRASP_VERIFIER_IMPLEMENTATION_REVISION,
            "physics_config_digest": physics_config_digest(self.environment.config),
            "network_config": {"type": "mlp_actor_critic", "hidden_sizes": [64, 64],
                               "activation": "tanh", "recurrent": False},
            "fixed_observation_normalization": True,
            "curriculum_stage": self.environment.config.curriculum_stage,
            "parent_checkpoint_sha256": self.curriculum_parent_checkpoint_sha256,
            "training_transitions_total": self.training_steps,
            "training_transitions_stage": (
                self.training_steps - self.curriculum_stage_start_steps),
            "curriculum_stage_start_steps": self.curriculum_stage_start_steps,
        })
        temporary = path.with_name(".%s.augment.tmp" % path.name)
        torch.save(payload, temporary); temporary.replace(path)
        return path

    def load_checkpoint(self, checkpoint_path: Path):
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        expected = {
            "observation_schema_id": GRASP_OBSERVATION_SCHEMA_ID,
            "action_schema_id": GRASP_ACTION_SCHEMA_ID,
            "reward_version": GRASP_REWARD_VERSION,
            "reward_config_digest": reward_config_digest(self.environment.config),
            "success_criteria_version": GRASP_SUCCESS_CRITERIA_VERSION,
            "physics_config_digest": physics_config_digest(self.environment.config),
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError("Grasp checkpoint %s incompatible" % name)
        checkpoint_stage = payload.get("curriculum_stage", "C_full")
        if checkpoint_stage != self.environment.config.curriculum_stage:
            raise ValueError("Grasp checkpoint curriculum_stage incompatible")
        loaded = super().load_checkpoint(checkpoint_path)
        self.curriculum_parent_checkpoint_sha256 = payload.get("parent_checkpoint_sha256")
        self.curriculum_stage_start_steps = int(payload.get(
            "curriculum_stage_start_steps",
            self.training_steps - int(payload.get("training_transitions_stage", self.training_steps)),
        ))
        return loaded

    def load_staged_curriculum_checkpoint(self, checkpoint_path: Path):
        """Continue optimizer state while allowing only the reset stage to change."""
        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        source = dict(payload.get("environment_config") or {})
        current = asdict(self.environment.config)
        source_stage = source.pop("curriculum_stage", payload.get("curriculum_stage", "C_full"))
        current.pop("curriculum_stage", None)
        if source != current:
            raise ValueError("staged curriculum checkpoint changes more than reset profile")
        loaded = PPOTrainer.load_checkpoint(self, checkpoint_path)
        if source_stage == self.environment.config.curriculum_stage:
            self.curriculum_parent_checkpoint_sha256 = payload.get("parent_checkpoint_sha256")
            self.curriculum_stage_start_steps = int(payload.get(
                "curriculum_stage_start_steps",
                self.training_steps - int(payload.get("training_transitions_stage", self.training_steps)),
            ))
        else:
            self.curriculum_parent_checkpoint_sha256 = hashlib.sha256(
                checkpoint_path.read_bytes()).hexdigest()
            self.curriculum_stage_start_steps = self.training_steps
        return loaded

    def load_curriculum_checkpoint(self, checkpoint_path: Path):
        """Transfer Fixed -> ±6 mm while keeping physics/reward/PPO identical."""
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        source = dict(payload.get("environment_config") or {})
        current = asdict(self.environment.config)
        for name in ("randomize_xy", "random_seed"):
            source.pop(name, None); current.pop(name, None)
        if source != current:
            raise ValueError("curriculum checkpoint changes more than XY randomization/seed")
        return PPOTrainer.load_checkpoint(self, checkpoint_path)


def deterministic_evaluate(environment, trainer: ExperimentalGraspPPOTrainer,
                           episodes: int = 100):
    if episodes < 1:
        raise ValueError("evaluation episodes must be positive")
    trainer.policy.eval()
    records = []
    for episode in range(episodes):
        observation = environment.reset()
        done = False; total_reward = 0.0; steps = 0; info = {}
        while not done:
            raw = flatten_observation(observation, GRASP_POLICY_SCHEMA).to(trainer.device)
            normalized = trainer.normalizer.normalize(raw, update=False)
            with torch.no_grad():
                action, _, _ = trainer.policy.act(normalized, deterministic=True)
            observation, reward, done, info = environment.step(
                tuple(float(v) for v in action.detach().cpu()))
            total_reward += float(reward); steps += 1
            if steps > environment.config.learning_horizon + 1:
                raise RuntimeError("evaluation episode exceeded environment horizon")
        record = {
            "episode": episode, "success": bool(info["success"]),
            "failure_reason": str(info.get("failure_reason", "")),
            "reward": total_reward, "learning_steps": int(info.get("learning_steps", steps)),
            "time_to_grasp": float(info.get("time_to_grasp", 0.0)),
            "verification_lift": float(info.get("verification_lift", 0.0)),
            "relative_drift": float(info.get("relative_drift", 0.0)),
            "candidate_local_ee_rise": float(info.get("candidate_local_ee_rise", 0.0)),
            "reset_relative_ee_rise": float(info.get("reset_relative_ee_rise", 0.0)),
            "cube_rise": float(info.get("cube_rise", 0.0)),
            "candidate_local_cube_rise": float(info.get("candidate_local_cube_rise", 0.0)),
            "lift_ratio": float(info.get("lift_ratio", 0.0)),
            "candidate_local_lift_ratio": float(info.get("candidate_local_lift_ratio", 0.0)),
            "rotation_drift": float(info.get("rotation_drift", 0.0)),
            "verification_clearance": float(info.get("verification_clearance", 0.0)),
            "time_to_verified": info.get("time_to_verified"),
            "unsafe": bool(info.get("unsafe", False)),
            "non_finite": bool(info.get("non_finite", False)),
            "force_max": float(info.get("force_max", 0.0)),
            "grasp_confidence": float(info.get("grasp_confidence", 0.0)),
            "grasp_candidate": bool(info.get("candidate_once", info.get("grasp_candidate", False))),
            "verification_entered": bool(info.get("verification_entered", False)),
            "aligned": bool(info.get("aligned", False)),
            "descended": bool(info.get("descended", False)),
            "clearance_reached": bool(info.get("clearance_reached", False)),
            "post_clearance_descended": bool(info.get("post_clearance_descended", False)),
            "closure_started": bool(info.get("closure_started", False)),
            "minimum_distance": float(info.get("minimum_distance", 0.0)),
            "minimum_xy_error": float(info.get("minimum_xy_error", 0.0)),
            "final_xy_error": float(info.get("final_xy_error", 0.0)),
            "minimum_z_error": float(info.get("minimum_z_error", 0.0)),
            "final_z_error": float(info.get("final_z_error", 0.0)),
            "jaw_command": float(info.get("jaw_command", 0.0)),
            "jaw_actual": tuple(float(v) for v in info.get("jaw_actual", ())),
            "corridor_error": tuple(float(v) for v in info.get("corridor_error", ())),
            "candidate_evidence_flags": dict(info.get("candidate_evidence_flags", {})),
            "minimum_commanded_jaw": float(info.get("minimum_commanded_jaw", 0.05)),
            "mean_actual_jaw": float(info.get("mean_actual_jaw", 0.05)),
            "maximum_blockage_gap": float(info.get("maximum_blockage_gap", 0.0)),
            "time_to_contact": info.get("time_to_contact"),
            "time_to_candidate": info.get("time_to_candidate"),
            "jaw_below_040": bool(info.get("jaw_below_040", False)),
            "jaw_below_035": bool(info.get("jaw_below_035", False)),
            "jaw_below_030": bool(info.get("jaw_below_030", False)),
            "blockage_ge_0005": bool(info.get("blockage_ge_0005", False)),
            "blockage_ge_0010": bool(info.get("blockage_ge_0010", False)),
            "blockage_ge_0015": bool(info.get("blockage_ge_0015", False)),
            "blockage_ge_0020": bool(info.get("blockage_ge_0020", False)),
        }
        records.append(record)
    successes = sum(item["success"] for item in records)
    lower, upper = wilson_confidence_interval(successes, episodes)
    histogram = {}
    for item in records:
        reason = item["failure_reason"] or "SUCCESS"
        histogram[reason] = histogram.get(reason, 0) + 1
    times = [item["time_to_grasp"] for item in records]
    successful = [item for item in records if item["success"]]
    verified_records = [item for item in records if item["verification_entered"]]
    def mean_metric(name, source=verified_records):
        return (sum(float(item[name]) for item in source) / len(source)) if source else 0.0
    summary = {
        "deterministic": True, "episodes": episodes,
        "success_count": successes, "success_rate": successes / episodes,
        "wilson_95_lower": lower, "wilson_95_upper": upper,
        "timeout_count": histogram.get("TIMEOUT", 0),
        "verification_fail_count": histogram.get("GRASP_VERIFICATION_FAILED", 0),
        "premature_push_count": histogram.get("PREMATURE_OBJECT_PUSH", 0),
        "unsafe_count": sum(item["unsafe"] for item in records),
        "non_finite_count": sum(item["non_finite"] for item in records),
        "mean_time_to_grasp": sum(times) / len(times),
        "median_time_to_grasp": median(times),
        "mean_verification_lift": (sum(item["verification_lift"] for item in successful) /
                                   max(1, len(successful))),
        "mean_relative_drift": (sum(item["relative_drift"] for item in successful) /
                                max(1, len(successful))),
        "max_relative_drift": max((item["relative_drift"] for item in successful), default=0.0),
        "mean_candidate_local_ee_rise": mean_metric("candidate_local_ee_rise"),
        "median_candidate_local_ee_rise": median(
            [item["candidate_local_ee_rise"] for item in verified_records]) if verified_records else 0.0,
        "mean_reset_relative_ee_rise": mean_metric("reset_relative_ee_rise"),
        "median_reset_relative_ee_rise": median(
            [item["reset_relative_ee_rise"] for item in verified_records]) if verified_records else 0.0,
        "mean_cube_rise": mean_metric("cube_rise"),
        "median_cube_rise": median(
            [item["cube_rise"] for item in verified_records]) if verified_records else 0.0,
        "mean_lift_ratio": mean_metric("lift_ratio"),
        "median_lift_ratio": median(
            [item["lift_ratio"] for item in verified_records]) if verified_records else 0.0,
        "mean_candidate_local_lift_ratio": mean_metric("candidate_local_lift_ratio"),
        "mean_rotation_drift": mean_metric("rotation_drift"),
        "mean_verification_clearance": mean_metric("verification_clearance"),
        "mean_time_to_verified": (
            sum(float(item["time_to_verified"]) for item in successful) / len(successful)
            if successful else 0.0),
        "verifier_implementation_revision": GRASP_VERIFIER_IMPLEMENTATION_REVISION,
        "failure_reason_histogram": histogram,
        "candidate_count": sum(item["grasp_candidate"] for item in records),
        "candidate_rate": sum(item["grasp_candidate"] for item in records) / episodes,
        "verification_entered_count": sum(item["verification_entered"] for item in records),
        "behavior_funnel": {
            "episodes": episodes,
            "clearance_reached": sum(item["clearance_reached"] for item in records),
            "aligned": sum(item["aligned"] for item in records),
            "descended": sum(item["descended"] for item in records),
            "post_clearance_descended": sum(
                item["post_clearance_descended"] for item in records),
            "closure_started": sum(item["closure_started"] for item in records),
            "jaw_below_040": sum(item["jaw_below_040"] for item in records),
            "jaw_below_035": sum(item["jaw_below_035"] for item in records),
            "jaw_below_030": sum(item["jaw_below_030"] for item in records),
            "blockage_ge_0005": sum(item["blockage_ge_0005"] for item in records),
            "blockage_ge_0010": sum(item["blockage_ge_0010"] for item in records),
            "blockage_ge_0015": sum(item["blockage_ge_0015"] for item in records),
            "blockage_ge_0020": sum(item["blockage_ge_0020"] for item in records),
            "candidate": sum(item["grasp_candidate"] for item in records),
            "verifying": sum(item["verification_entered"] for item in records),
            "success": successes,
        },
        "mean_minimum_xy_error": sum(item["minimum_xy_error"] for item in records) / episodes,
        "mean_final_xy_error": sum(item["final_xy_error"] for item in records) / episodes,
        "mean_minimum_z_error": sum(item["minimum_z_error"] for item in records) / episodes,
        "mean_minimum_commanded_jaw": sum(item["minimum_commanded_jaw"] for item in records) / episodes,
        "mean_actual_jaw": sum(item["mean_actual_jaw"] for item in records) / episodes,
        "mean_maximum_blockage_gap": sum(item["maximum_blockage_gap"] for item in records) / episodes,
        "mean_time_to_contact": (sum(item["time_to_contact"] for item in records
                                     if item["time_to_contact"] is not None)
                                 / max(1, sum(item["time_to_contact"] is not None for item in records))),
        "mean_time_to_candidate": (sum(item["time_to_candidate"] for item in records
                                       if item["time_to_candidate"] is not None)
                                   / max(1, sum(item["time_to_candidate"] is not None for item in records))),
        "resource_cleanup": True,
    }
    if not all(math.isfinite(float(value)) for value in summary.values()
               if isinstance(value, (int, float))):
        raise RuntimeError("evaluation produced non-finite summary")
    return summary, records
