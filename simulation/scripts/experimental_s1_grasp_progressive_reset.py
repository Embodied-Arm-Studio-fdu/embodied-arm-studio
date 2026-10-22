#!/usr/bin/env python3
"""P2-R6 progressive reset curriculum (experimental, never registered)."""

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
    GraspPPOEnvironment, progressive_reset_pose,
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
OUTPUT = ROOT / "artifacts" / "experimental" / "p2_r6_s1_grasp"
STAGE_B_BEST = (ROOT / "artifacts" / "experimental" / "p2_r4_s1_grasp" /
                "stage_B" / "checkpoints" / "model_best.pt")
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


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def contract():
    return {
        "observation": GRASP_OBSERVATION_SCHEMA_ID,
        "observation_dim": GRASP_POLICY_SCHEMA.observation_dim,
        "action": GRASP_ACTION_SCHEMA_ID,
        "action_dim": GRASP_POLICY_SCHEMA.action_dim,
        "reward": GRASP_REWARD_VERSION,
        "candidate": "locked_6_step_composite_evidence",
        "success": GRASP_SUCCESS_CRITERIA_VERSION,
        "verifier": GRASP_VERIFIER_IMPLEMENTATION_REVISION,
        "only_changed_variable": "reset_initial_ee_pose",
        "stage_id_in_observation_or_reward": False,
    }


class ProgressiveResetGraspEnvironment(GraspPPOEnvironment):
    """Pre-position EE along B->Full interpolation before rollout begins."""

    def __init__(self, progressive_stage, config=None):
        if progressive_stage not in range(1, 6):
            raise ValueError("progressive_stage must be 1..5")
        super().__init__(config or GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_C))
        self.progressive_stage = progressive_stage
        self.reset_pose_records = []
        self._pose_anchor = None

    def _curriculum_reset_target(self, snapshot):
        full = tuple(float(value) for value in snapshot["ee_state"][:3])
        grasp = self._grasp_point(snapshot)
        stage_b = (
            grasp[0] + STAGE_B_GRASP_XY_OFFSET[0],
            grasp[1] + STAGE_B_GRASP_XY_OFFSET[1],
            grasp[2] + self.config.pregrasp_clearance,
        )
        requested = progressive_reset_pose(self.progressive_stage, stage_b, full)
        self._pose_anchor = {"stage_b": stage_b, "full": full, "requested": requested}
        # R6-5 is the untouched original reset, not a controller replay of it.
        return None if self.progressive_stage == 5 else requested

    def reset(self):
        observation = super().reset()
        record = dict(self._pose_anchor or {})
        if not record:
            # Defensive: reset must always consult the reset target hook.
            raise RuntimeError("PROGRESSIVE_RESET_ANCHOR_MISSING")
        record.update({
            "stage": self.progressive_stage,
            "actual": tuple(float(value) for value in self._initial_ee),
            "jaw_open": self.config.open_safe,
            "orientation_xyzw": self.config.fixed_orientation_xyzw,
        })
        self.reset_pose_records.append(record)
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        info["progressive_reset_stage"] = self.progressive_stage
        return observation, reward, done, info


def ppo_from_checkpoint(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return PPOConfig(**payload["ppo_config"])


def load_continuation(trainer, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("curriculum_stage") == CURRICULUM_STAGE_C:
        return trainer.load_checkpoint(checkpoint)
    return trainer.load_staged_curriculum_checkpoint(checkpoint)


def pose_summary(records):
    if not records:
        return {}
    result = {"samples": len(records)}
    for name in ("stage_b", "full", "requested", "actual"):
        result[name] = [sum(float(row[name][i]) for row in records) / len(records)
                        for i in range(3)]
    result["max_requested_actual_error_m"] = max(
        sum((float(row["requested"][i]) - float(row["actual"][i])) ** 2
            for i in range(3)) ** .5 for row in records)
    return result


def augment_summary(summary):
    histogram = summary["failure_reason_histogram"]
    summary["verified_count"] = summary["success_count"]
    summary["workspace_violation_count"] = histogram.get("WORKSPACE_VIOLATION", 0)
    summary["behavior_funnel"]["verified"] = summary["success_count"]
    summary["stage_gate_pass"] = bool(
        summary["candidate_rate"] >= .80
        and summary["success_rate"] >= .70
        and summary["unsafe_count"] == 0
        and summary["non_finite_count"] == 0)
    return summary


def evaluate(stage, checkpoint, episodes, output, randomize_xy=0.0):
    config = GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_C,
                            randomize_xy=randomize_xy)
    env = (ProgressiveResetGraspEnvironment(stage, config)
           if stage is not None else GraspPPOEnvironment(config))
    try:
        env.initialize()
        trainer = ExperimentalGraspPPOTrainer(
            env, config=ppo_from_checkpoint(checkpoint), device="cuda:0")
        trainer.load_checkpoint(checkpoint)
        summary, records = deterministic_evaluate(env, trainer, episodes)
        augment_summary(summary)
        summary.update({
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "progressive_stage": stage, "randomize_xy_m": randomize_xy,
            "reset_pose_summary": pose_summary(getattr(env, "reset_pose_records", [])),
            "resource_cleanup": True,
        })
    finally:
        env.close()
    write_json(output / "summary.json", summary)
    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8")
    if getattr(env, "reset_pose_records", None):
        write_json(output / "reset_poses.json", env.reset_pose_records)
    return summary


def expected_parent(stage):
    if stage == 1:
        return STAGE_B_BEST
    return OUTPUT / ("stage_%d" % (stage - 1)) / "checkpoints" / "model_best.pt"


def train_stage(stage, resume, transitions, cumulative):
    if stage not in range(1, 6):
        raise ValueError("stage must be 1..5")
    if transitions <= 0 or transitions % 200 or cumulative not in (10000, 20000):
        raise ValueError("stage transition budget must be 10k or cumulative 20k")
    resume = Path(resume or (expected_parent(stage) if cumulative == 10000 else
                             OUTPUT / ("stage_%d" % stage) / "checkpoints" /
                             "model_10000.pt"))
    required = (expected_parent(stage) if cumulative == 10000 else
                OUTPUT / ("stage_%d" % stage) / "checkpoints" / "model_10000.pt")
    if resume.resolve() != required.resolve():
        raise ValueError("stage continuity requires parent %s" % required)
    output = OUTPUT / ("stage_%d" % stage)
    segment = output / ("segment_%05d" % cumulative)
    seed_everything(20262000 + stage * 100 + cumulative)
    env = ProgressiveResetGraspEnvironment(
        stage, GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_C))
    try:
        env.initialize()
        trainer = ExperimentalGraspPPOTrainer(
            env, config=ppo_from_checkpoint(resume), device="cuda:0")
        load_continuation(trainer, resume)
        write_json(segment / "config.json", {
            "experiment": "P2-R6", "progressive_stage": stage,
            "transitions_this_segment": transitions,
            "transitions_stage_cumulative": cumulative,
            "parent_checkpoint": str(resume), "parent_checkpoint_sha256": sha256(resume),
            "environment": asdict(env.config), "ppo": asdict(trainer.config),
            "contract": contract(),
        })
        started = time.time()
        last = {}
        for local in range(1, transitions // 200 + 1):
            metrics = trainer.train_one_update()
            last = {"local_update": local, "update_index": trainer.update_steps,
                    "training_steps_total": trainer.training_steps, **metrics}
            append_jsonl(segment / "metrics.jsonl", last)
            for episode in trainer._rollout_episode_records:
                append_jsonl(segment / "episodes.jsonl", episode)
            if local % 10 == 0 or local == transitions // 200:
                print("stage=%d cumulative=%d update=%d reward=%.5f candidate=%.3f success=%.3f" % (
                    stage, cumulative, trainer.update_steps, last["average_reward"],
                    last.get("candidate_rate", 0.0),
                    last.get("verified_success_rate", 0.0)), flush=True)
        saved = trainer.save_checkpoint(
            OUTPUT, "stage_%d" % stage, "segment_%05d" % cumulative, "model_latest.pt")
        checkpoint = output / "checkpoints" / ("model_%05d.pt" % cumulative)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved, checkpoint)
        training = {
            "stage": stage, "transitions_this_segment": transitions,
            "transitions_stage_cumulative": cumulative,
            "training_transitions_total": trainer.training_steps,
            "elapsed_seconds": time.time() - started,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "reset_pose_summary": pose_summary(env.reset_pose_records),
            "last_metrics": last, "resource_cleanup": True,
        }
    finally:
        env.close()
    write_json(segment / "training_summary.json", training)
    evaluation = evaluate(stage, checkpoint, 50, segment / "evaluation")
    result = {"training": training, "evaluation": evaluation}
    write_json(segment / "summary.json", result)
    return result


def promote(stage, checkpoint, reason):
    checkpoint = Path(checkpoint)
    destination = OUTPUT / ("stage_%d" % stage) / "checkpoints" / "model_best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    result = {"stage": stage, "source": str(checkpoint),
              "source_sha256": sha256(checkpoint), "best": str(destination),
              "best_sha256": sha256(destination), "reason": reason}
    write_json(OUTPUT / ("stage_%d" % stage) / "promotion.json", result)
    return result


def pose_probe():
    stage_b_env = GraspPPOEnvironment(GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_B))
    try:
        stage_b_env.initialize(); stage_b_env.reset()
        stage_b_actual = tuple(float(value) for value in stage_b_env._initial_ee)
    finally:
        stage_b_env.close()
    results = []
    for stage in range(1, 6):
        env = ProgressiveResetGraspEnvironment(stage)
        try:
            env.initialize(); env.reset()
            results.append(env.reset_pose_records[-1])
        finally:
            env.close()
    payload = {"source": "live Isaac Gym reset and code-derived interpolation",
               "stage_b_actual": stage_b_actual,
               "contract": contract(), "records": results}
    write_json(OUTPUT / "reset_pose_probe.json", payload)
    return payload


def final_full(checkpoint):
    checkpoint = Path(checkpoint)
    result = evaluate(None, checkpoint, 100, OUTPUT / "final_full" / "evaluation")
    result["full_gate_pass"] = bool(
        result["success_rate"] >= .90 and result["wilson_95_lower"] >= .80
        and result["unsafe_count"] == 0 and result["non_finite_count"] == 0)
    result["P2_R6_PROGRESSIVE_RESET_GATE"] = "PASS" if result["full_gate_pass"] else "FAIL"
    write_json(OUTPUT / "final_full" / "summary.json", result)
    return result


def randomized_train(resume, transitions, cumulative):
    if transitions <= 0 or transitions % 200 or cumulative < 20000 or cumulative > 50000:
        raise ValueError("randomized budget must be 20k..50k in 200-transition units")
    resume = Path(resume)
    required = (OUTPUT / "stage_5" / "checkpoints" / "model_best.pt"
                if cumulative == 20000 else
                OUTPUT / "random_xy" / "checkpoints" /
                ("model_%05d.pt" % (cumulative - transitions)))
    if resume.resolve() != required.resolve():
        raise ValueError("randomized continuity requires parent %s" % required)
    output = OUTPUT / "random_xy" / ("train_%05d" % cumulative)
    seed_everything(20263000 + cumulative)
    env = GraspPPOEnvironment(GraspEnvConfig(
        curriculum_stage=CURRICULUM_STAGE_C, randomize_xy=0.006))
    try:
        env.initialize()
        trainer = ExperimentalGraspPPOTrainer(
            env, config=ppo_from_checkpoint(resume), device="cuda:0")
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if float((payload.get("environment_config") or {}).get("randomize_xy", 0.0)) == .006:
            trainer.load_checkpoint(resume)
        else:
            trainer.load_curriculum_checkpoint(resume)
            trainer.curriculum_parent_checkpoint_sha256 = sha256(resume)
            trainer.curriculum_stage_start_steps = trainer.training_steps
        write_json(output / "config.json", {
            "experiment": "P2-R6", "randomize_xy_m": .006,
            "randomized_transitions_this_segment": transitions,
            "randomized_transitions_cumulative": cumulative,
            "parent_checkpoint": str(resume), "parent_checkpoint_sha256": sha256(resume),
            "environment": asdict(env.config), "ppo": asdict(trainer.config),
            "contract": contract(),
        })
        started = time.time(); last = {}
        for local in range(1, transitions // 200 + 1):
            metrics = trainer.train_one_update()
            last = {"local_update": local, "update_index": trainer.update_steps,
                    "training_steps_total": trainer.training_steps, **metrics}
            append_jsonl(output / "metrics.jsonl", last)
            for episode in trainer._rollout_episode_records:
                append_jsonl(output / "episodes.jsonl", episode)
            if local % 10 == 0 or local == transitions // 200:
                print("random cumulative=%d update=%d reward=%.5f candidate=%.3f success=%.3f" % (
                    cumulative, trainer.update_steps, last["average_reward"],
                    last.get("candidate_rate", 0.0),
                    last.get("verified_success_rate", 0.0)), flush=True)
        saved = trainer.save_checkpoint(
            OUTPUT, "random_xy", "train_%05d" % cumulative, "model_latest.pt")
        checkpoint = OUTPUT / "random_xy" / "checkpoints" / ("model_%05d.pt" % cumulative)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved, checkpoint)
        result = {
            "randomized_transitions_this_segment": transitions,
            "randomized_transitions_cumulative": cumulative,
            "training_transitions_total": trainer.training_steps,
            "elapsed_seconds": time.time() - started,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "last_metrics": last, "resource_cleanup": True,
        }
    finally:
        env.close()
    write_json(output / "training_summary.json", result)
    return result


def randomized_eval(checkpoint, cumulative):
    result = evaluate(
        None, Path(checkpoint), 100,
        OUTPUT / "random_xy" / ("evaluation_%05d" % cumulative),
        randomize_xy=.006)
    result["randomized_transitions_cumulative"] = cumulative
    result["randomized_gate_pass"] = bool(
        result["success_rate"] >= .80 and result["wilson_95_lower"] >= .70
        and result["unsafe_count"] == 0 and result["non_finite_count"] == 0)
    write_json(OUTPUT / "random_xy" / ("evaluation_%05d" % cumulative) / "summary.json",
               result)
    return result


def production_regression():
    document = task_catalog_document()
    ids = [item.id for item in PRODUCTION_TASK_REGISTRY.definitions()]
    result = {"task_count": len(ids), "task_ids": ids,
              "catalog_version": document["catalog_version"],
              "catalog_digest": document["catalog_digest"],
              "experimental_tasks_registered": sorted(
                  set(ids).intersection({"s1_grasp", "s1_lift", "s1_place"}))}
    result["passed"] = bool(ids == EXPECTED_TASK_IDS
                            and result["catalog_version"] == "task-catalog/v1"
                            and result["catalog_digest"] == EXPECTED_CATALOG_DIGEST
                            and not result["experimental_tasks_registered"])
    write_json(OUTPUT / "production_regression.json", result)
    return result


def finalize():
    production = production_regression()
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"experiment": "P2-R6", "contract": contract(),
                "stage_b_parent": {"path": str(STAGE_B_BEST),
                                   "sha256": sha256(STAGE_B_BEST)},
                "production_regression": production, "files": files}
    write_json(OUTPUT / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=(
        "pose-probe", "train-stage", "promote", "final-full",
        "randomized-train", "randomized-eval", "production-regression", "finalize"))
    parser.add_argument("--stage", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--transitions", type=int, default=10000)
    parser.add_argument("--cumulative", type=int, default=10000)
    parser.add_argument("--reason", default="stage_gate_pass")
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    if args.mode == "pose-probe": result = pose_probe()
    elif args.mode == "train-stage":
        if args.stage is None: parser.error("--stage is required")
        result = train_stage(args.stage, args.resume, args.transitions, args.cumulative)
    elif args.mode == "promote":
        if args.stage is None or not args.checkpoint:
            parser.error("--stage and --checkpoint are required")
        result = promote(args.stage, args.checkpoint, args.reason)
    elif args.mode == "final-full":
        if not args.checkpoint: parser.error("--checkpoint is required")
        result = final_full(args.checkpoint)
    elif args.mode == "randomized-train":
        if not args.resume: parser.error("--resume is required")
        result = randomized_train(args.resume, args.transitions, args.cumulative)
    elif args.mode == "randomized-eval":
        if not args.checkpoint: parser.error("--checkpoint is required")
        result = randomized_eval(args.checkpoint, args.cumulative)
    elif args.mode == "production-regression": result = production_regression()
    else: result = finalize()
    print("P2_R6_%s=%s" % (args.mode.upper().replace("-", "_"),
                             json.dumps(result, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
