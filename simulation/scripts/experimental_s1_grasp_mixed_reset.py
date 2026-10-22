#!/usr/bin/env python3
"""P2-R5 mixed-reset curriculum runner (experimental, never registered)."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import torch

from simulation_core.experimental.manipulation.grasp_env import (
    CURRICULUM_STAGE_B, CURRICULUM_STAGE_C, STAGE_B_GRASP_XY_OFFSET,
    GRASP_VERIFIER_IMPLEMENTATION_REVISION, GraspEnvConfig,
    GraspPPOEnvironment,
)
from simulation_core.experimental.manipulation.grasp_ppo import (
    ExperimentalGraspPPOTrainer, deterministic_evaluate,
)
from simulation_core.experimental.manipulation.grasp_schema import (
    GRASP_ACTION_SCHEMA_ID, GRASP_OBSERVATION_SCHEMA_ID, GRASP_POLICY_SCHEMA,
    GRASP_REWARD_VERSION, GRASP_SUCCESS_CRITERIA_VERSION,
)
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document
from simulation_core.task_registry import PRODUCTION_TASK_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p2_r5_s1_grasp"
STAGE_B_BEST = (ROOT / "artifacts" / "experimental" / "p2_r4_s1_grasp" /
                "stage_B" / "checkpoints" / "model_best.pt")
PHASES = {
    1: (0.70, 0.30),
    2: (0.40, 0.60),
    3: (0.10, 0.90),
    4: (0.00, 1.00),
}
EXPECTED_TASK_IDS = [
    "demo_s1_circle_tracking", "demo_s1_joint_position", "demo_s1_joint_sweep",
    "demo_s1_line_tracking", "demo_s1_target_pose", "s1_push", "s1_reach",
]
EXPECTED_CATALOG_DIGEST = (
    "sha256:c0f3feb0771f4ab636c61fa53ba0d49d5f2a489a224b6890ae38ed798423f791"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def seed_everything(seed=20261002):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locked_contract():
    return {
        "observation": GRASP_OBSERVATION_SCHEMA_ID,
        "observation_dim": GRASP_POLICY_SCHEMA.observation_dim,
        "action": GRASP_ACTION_SCHEMA_ID,
        "action_dim": GRASP_POLICY_SCHEMA.action_dim,
        "reward": GRASP_REWARD_VERSION,
        "candidate": "locked_6_step_composite_evidence",
        "success": GRASP_SUCCESS_CRITERIA_VERSION,
        "verifier_implementation_revision": GRASP_VERIFIER_IMPLEMENTATION_REVISION,
        "ppo_network_physics_geometry": "P2-R4 locked",
    }


class MixedResetGraspEnvironment(GraspPPOEnvironment):
    """Select Stage-B or original-full reset without changing task semantics."""

    def __init__(self, stage_b_probability, *, selector_seed, config=None):
        super().__init__(config or GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_C))
        self._selector_rng = random.Random(selector_seed)
        self.set_stage_b_probability(stage_b_probability)
        self._active_reset_profile = "full"
        self.reset_profile_counts = {"stage_b": 0, "full": 0}

    def set_stage_b_probability(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("stage_b_probability must be in [0, 1]")
        self.stage_b_probability = value

    def reset(self):
        self._active_reset_profile = (
            "stage_b" if self._selector_rng.random() < self.stage_b_probability else "full"
        )
        self.reset_profile_counts[self._active_reset_profile] += 1
        return super().reset()

    def _curriculum_reset_target(self, snapshot):
        if self._active_reset_profile == "full":
            return None
        grasp = self._grasp_point(snapshot)
        return (
            grasp[0] + STAGE_B_GRASP_XY_OFFSET[0],
            grasp[1] + STAGE_B_GRASP_XY_OFFSET[1],
            grasp[2] + self.config.pregrasp_clearance,
        )

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["reset_profile"] = self._active_reset_profile
        return observation, reward, done, info


def ppo_from_checkpoint(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return PPOConfig(**payload["ppo_config"])


def load_for_continuation(trainer, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("curriculum_stage") == trainer.environment.config.curriculum_stage:
        return trainer.load_checkpoint(checkpoint)
    return trainer.load_staged_curriculum_checkpoint(checkpoint)


def train_updates(environment, trainer, updates, output):
    records = []
    start = time.time()
    for local in range(1, updates + 1):
        metrics = trainer.train_one_update()
        record = {
            "update_index": trainer.update_steps,
            "training_steps_total": trainer.training_steps,
            "local_update": local,
            **metrics,
        }
        append_jsonl(output / "metrics.jsonl", record)
        for episode in trainer._rollout_episode_records:
            append_jsonl(output / "episodes.jsonl", episode)
        records.append(record)
        if local % 10 == 0 or local == updates:
            print(
                "update=%d total=%d reward=%.5f candidate=%.3f success=%.3f"
                % (trainer.update_steps, trainer.training_steps, record["average_reward"],
                   record.get("candidate_rate", 0.0),
                   record.get("verified_success_rate", 0.0)),
                flush=True,
            )
    checkpoint = trainer.save_checkpoint(
        output.parents[1], output.parent.name, output.name, "model_latest.pt")
    return {
        "updates": updates,
        "transitions": updates * trainer.config.rollout_steps,
        "training_transitions_total": trainer.training_steps,
        "elapsed_seconds": time.time() - start,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "reset_profile_counts": dict(getattr(environment, "reset_profile_counts", {})),
        "last_metrics": records[-1],
        "resource_cleanup": True,
    }, checkpoint


def augment_evaluation(summary):
    histogram = summary["failure_reason_histogram"]
    summary["verified_count"] = summary["success_count"]
    summary["workspace_violation_count"] = histogram.get("WORKSPACE_VIOLATION", 0)
    summary["behavior_funnel"]["verified"] = summary["success_count"]
    summary["gate_pass"] = bool(
        summary["success_rate"] >= 0.90
        and summary["wilson_95_lower"] >= 0.80
        and summary["unsafe_count"] == 0
        and summary["non_finite_count"] == 0
    )
    return summary


def evaluate(checkpoint, episodes, output, randomize_xy=0.0):
    config = GraspEnvConfig(
        curriculum_stage=CURRICULUM_STAGE_C, randomize_xy=randomize_xy)
    env = GraspPPOEnvironment(config)
    try:
        env.initialize()
        trainer = ExperimentalGraspPPOTrainer(
            env, config=ppo_from_checkpoint(checkpoint), device="cuda:0")
        trainer.load_checkpoint(checkpoint)
        summary, records = deterministic_evaluate(env, trainer, episodes)
        augment_evaluation(summary)
        summary.update({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "reset_distribution": "original_full" if randomize_xy == 0.0 else "xy_uniform_+-0.006m",
            "randomize_xy_m": randomize_xy,
            "resource_cleanup": True,
        })
    finally:
        env.close()
    write_json(output / "summary.json", summary)
    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return summary


def phase(number, resume):
    if number not in PHASES:
        raise ValueError("phase must be 1..4")
    expected_resume = STAGE_B_BEST if number == 1 else (
        OUTPUT / ("phase_%d" % (number - 1)) / "checkpoints" / "model_latest.pt")
    resume = Path(resume or expected_resume)
    if resume.resolve() != expected_resume.resolve():
        raise ValueError("phase %d must resume %s" % (number, expected_resume))
    stage_b_probability, full_probability = PHASES[number]
    output = OUTPUT / ("phase_%d" % number)
    seed_everything(20261002 + number)
    env = MixedResetGraspEnvironment(
        stage_b_probability, selector_seed=20261100 + number,
        config=GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_C),
    )
    try:
        env.initialize()
        trainer = ExperimentalGraspPPOTrainer(
            env, config=ppo_from_checkpoint(resume), device="cuda:0")
        load_for_continuation(trainer, resume)
        config = {
            "experiment": "P2-R5", "phase": number,
            "stage_b_reset_probability": stage_b_probability,
            "full_reset_probability": full_probability,
            "transitions": 10000, "parent_checkpoint": str(resume),
            "parent_checkpoint_sha256": sha256(resume),
            "environment": asdict(env.config), "ppo": asdict(trainer.config),
            "contract": locked_contract(),
        }
        write_json(output / "config.json", config)
        training, checkpoint = train_updates(env, trainer, 50, output)
    finally:
        env.close()
    write_json(output / "training_summary.json", training)
    evaluation = evaluate(checkpoint, 50, output / "evaluation")
    result = {"phase": number, "training": training, "evaluation": evaluation}
    write_json(output / "summary.json", result)
    return result


def final_fixed(checkpoint):
    checkpoint = Path(checkpoint)
    destination = OUTPUT / "final_fixed" / "checkpoints" / "model_best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    result = evaluate(destination, 100, OUTPUT / "final_fixed" / "evaluation")
    result["source_checkpoint"] = str(checkpoint)
    result["source_checkpoint_sha256"] = sha256(checkpoint)
    result["P2_R5_MIXED_RESET_GATE"] = "PASS" if result["gate_pass"] else "FAIL"
    write_json(OUTPUT / "final_fixed" / "summary.json", result)
    return result


def randomized_train(resume, transitions, cumulative):
    if transitions <= 0 or transitions % 200:
        raise ValueError("transitions must be a positive multiple of 200")
    resume = Path(resume)
    output = OUTPUT / "random_xy" / ("train_%05d" % cumulative)
    seed_everything(20261200 + cumulative)
    env = GraspPPOEnvironment(GraspEnvConfig(
        curriculum_stage=CURRICULUM_STAGE_C, randomize_xy=0.006,
        random_seed=20261200 + cumulative,
    ))
    try:
        env.initialize()
        trainer = ExperimentalGraspPPOTrainer(
            env, config=ppo_from_checkpoint(resume), device="cuda:0")
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if float((payload.get("environment_config") or {}).get("randomize_xy", 0.0)) == 0.006:
            trainer.load_checkpoint(resume)
        else:
            trainer.load_curriculum_checkpoint(resume)
            trainer.curriculum_parent_checkpoint_sha256 = sha256(resume)
            trainer.curriculum_stage_start_steps = trainer.training_steps
        write_json(output / "config.json", {
            "experiment": "P2-R5", "randomize_xy_m": 0.006,
            "randomized_transitions_this_segment": transitions,
            "randomized_transitions_cumulative": cumulative,
            "parent_checkpoint": str(resume), "parent_checkpoint_sha256": sha256(resume),
            "environment": asdict(env.config), "ppo": asdict(trainer.config),
            "contract": locked_contract(),
        })
        training, checkpoint = train_updates(env, trainer, transitions // 200, output)
    finally:
        env.close()
    training["randomized_transitions_cumulative"] = cumulative
    write_json(output / "training_summary.json", training)
    return training


def randomized_eval(checkpoint, cumulative):
    output = OUTPUT / "random_xy" / ("evaluation_%05d" % cumulative)
    result = evaluate(Path(checkpoint), 100, output, randomize_xy=0.006)
    result["randomized_transitions_cumulative"] = cumulative
    result["randomized_gate_pass"] = bool(
        result["success_rate"] >= 0.80
        and result["wilson_95_lower"] >= 0.70
        and result["unsafe_count"] == 0
        and result["non_finite_count"] == 0
    )
    write_json(output / "summary.json", result)
    return result


def production_regression():
    document = task_catalog_document()
    ids = [definition.id for definition in PRODUCTION_TASK_REGISTRY.definitions()]
    result = {
        "task_count": len(ids), "task_ids": ids,
        "catalog_version": document["catalog_version"],
        "catalog_digest": document["catalog_digest"],
        "experimental_tasks_registered": sorted(
            set(ids).intersection({"s1_grasp", "s1_lift", "s1_place"})),
    }
    result["passed"] = bool(
        ids == EXPECTED_TASK_IDS
        and result["catalog_version"] == "task-catalog/v1"
        and result["catalog_digest"] == EXPECTED_CATALOG_DIGEST
        and not result["experimental_tasks_registered"]
    )
    write_json(OUTPUT / "production_regression.json", result)
    return result


def finalize():
    production = production_regression()
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
    manifest = {
        "experiment": "P2-R5", "contract": locked_contract(),
        "stage_b_parent": {"path": str(STAGE_B_BEST), "sha256": sha256(STAGE_B_BEST)},
        "phases": {str(key): {"stage_b": value[0], "full": value[1]}
                   for key, value in PHASES.items()},
        "production_regression": production, "files": files,
    }
    write_json(OUTPUT / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=(
        "phase", "final-fixed", "randomized-train", "randomized-eval",
        "production-regression", "finalize"))
    parser.add_argument("--phase", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--transitions", type=int)
    parser.add_argument("--cumulative", type=int)
    args = parser.parse_args()
    os.environ.pop("DISPLAY", None)
    if args.mode == "phase":
        if args.phase is None:
            parser.error("--phase is required")
        result = phase(args.phase, args.resume)
    elif args.mode == "final-fixed":
        if not args.checkpoint:
            parser.error("--checkpoint is required")
        result = final_fixed(args.checkpoint)
    elif args.mode == "randomized-train":
        if not args.resume or args.transitions is None or args.cumulative is None:
            parser.error("--resume, --transitions and --cumulative are required")
        result = randomized_train(args.resume, args.transitions, args.cumulative)
    elif args.mode == "randomized-eval":
        if not args.checkpoint or args.cumulative is None:
            parser.error("--checkpoint and --cumulative are required")
        result = randomized_eval(args.checkpoint, args.cumulative)
    elif args.mode == "production-regression":
        result = production_regression()
    else:
        result = finalize()
    print("P2_R5_%s=%s" % (args.mode.upper().replace("-", "_"),
                             json.dumps(result, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
