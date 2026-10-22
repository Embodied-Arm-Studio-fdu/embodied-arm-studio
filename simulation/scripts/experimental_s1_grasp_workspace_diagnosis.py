#!/usr/bin/env python3
"""P2-R7 read-only policy diagnosis for randomized workspace violations."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import torch

from simulation_core.experimental.manipulation.grasp_env import (
    CURRICULUM_STAGE_C, GraspEnvConfig, GraspPPOEnvironment,
)
from simulation_core.experimental.manipulation.grasp_ppo import ExperimentalGraspPPOTrainer
from simulation_core.ppo.policy import flatten_observation
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.experimental.manipulation.grasp_schema import GRASP_POLICY_SCHEMA
from simulation_core.task_catalog import task_catalog_document
from simulation_core.task_registry import PRODUCTION_TASK_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p2_r7_s1_grasp"
RANDOM_CHECKPOINT = (ROOT / "artifacts" / "experimental" / "p2_r6_s1_grasp" /
                     "random_xy" / "checkpoints" / "model_20000.pt")
FIXED_CHECKPOINT = (ROOT / "artifacts" / "experimental" / "p2_r6_s1_grasp" /
                    "stage_5" / "checkpoints" / "model_best.pt")
EXPECTED_DIGEST = "sha256:c0f3feb0771f4ab636c61fa53ba0d49d5f2a489a224b6890ae38ed798423f791"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def margin(xyz, lower, upper):
    return tuple(min(float(value) - low, high - float(value))
                 for value, low, high in zip(xyz, lower, upper))


def inside(xyz, lower, upper):
    return all(low <= float(value) <= high
               for value, low, high in zip(xyz, lower, upper))


def trace_row(info, control_step):
    lower = tuple(info["workspace_lower"]); upper = tuple(info["workspace_upper"])
    requested = tuple(info["requested_target_preclamp"])
    persistent = tuple(info["persistent_target_postclamp"])
    ee = tuple(info["learning_ee_position"])
    cube = tuple(info["learning_cube_position"])
    return {
        "control_step": control_step,
        "physics_step": control_step * 4,
        "ee_xyz": ee,
        "cube_xyz": cube,
        "cube_relative_ee": tuple(cube[i] - ee[i] for i in range(3)),
        "persistent_target_xyz": persistent,
        "target_preclamp_xyz": requested,
        "target_postclamp_xyz": persistent,
        "action": tuple(info["action"]),
        "workspace_lower": lower, "workspace_upper": upper,
        "workspace_trigger_axes": tuple(info["workspace_trigger_axes"]),
        "workspace_overshoot_mm": dict(info["workspace_overshoot_mm"]),
        "ee_inside_workspace": inside(ee, lower, upper),
        "preclamp_target_inside_workspace": inside(requested, lower, upper),
        "persistent_target_inside_workspace": inside(persistent, lower, upper),
        "ee_workspace_margin_m": margin(ee, lower, upper),
        "target_workspace_margin_m": margin(persistent, lower, upper),
        "jaw_command": float(info["jaw_command"]),
        "jaw_actual": tuple(info["jaw_actual"]),
        "blockage_gap": float(info["blockage_gap"]),
        "left_finger_force": float(info["left_finger_force"]),
        "right_finger_force": float(info["right_finger_force"]),
        "cube_force": float(info["cube_force"]),
        "bilateral_force_min": min(float(info["left_finger_force"]),
                                    float(info["right_finger_force"])),
        "candidate_window_progress": int(info["candidate_window_progress"]),
        "candidate_window_required": int(info["candidate_window_required"]),
        "candidate_only_missing_stability_window": bool(
            info["candidate_only_missing_stability_window"]),
        "candidate_flags": dict(info["candidate_evidence_flags"]),
        "grasp_candidate": bool(info["grasp_candidate"]),
        "candidate_event": bool(info["candidate_event"]),
        "failure_reason": str(info["failure_reason"]),
        "done": bool(info["failure_reason"] or info["success"]),
        "success": bool(info["success"]),
    }


def run_traces(checkpoint, randomize_xy, episodes, require_success=False):
    config = GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_C,
                            randomize_xy=randomize_xy)
    env = GraspPPOEnvironment(config)
    records = []
    try:
        env.initialize()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trainer = ExperimentalGraspPPOTrainer(
            env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
        trainer.load_checkpoint(checkpoint)
        trainer.policy.eval()
        attempts = 0
        while len(records) < episodes:
            attempts += 1
            if attempts > episodes * 3:
                raise RuntimeError("insufficient qualifying trace episodes")
            observation = env.reset()
            cube_xy = tuple(float(value) for value in env._initial_cube[:2])
            trace = []
            done = False; step = 0; info = {}
            while not done:
                raw = flatten_observation(observation, GRASP_POLICY_SCHEMA).to(trainer.device)
                normalized = trainer.normalizer.normalize(raw, update=False)
                with torch.no_grad():
                    action, _, _ = trainer.policy.act(normalized, deterministic=True)
                observation, _, done, info = env.step(
                    tuple(float(value) for value in action.detach().cpu()))
                step += 1
                trace.append(trace_row(info, step))
            if require_success and not info["success"]:
                continue
            record = {
                "episode": len(records), "attempt": attempts,
                "randomized_cube_xy": cube_xy,
                "success": bool(info["success"]),
                "failure_reason": str(info["failure_reason"]),
                "learning_steps": step,
                "trace": trace,
                "terminal": trace[-1],
            }
            records.append(record)
    finally:
        env.close()
    return records


def summarize_violations(records):
    violations = [record for record in records
                  if record["failure_reason"] == "WORKSPACE_VIOLATION"]
    axes = Counter(axis for record in violations
                   for axis in record["terminal"]["workspace_trigger_axes"])
    amounts = [amount for record in violations
               for amount in record["terminal"]["workspace_overshoot_mm"].values()]
    progress = [record["terminal"]["candidate_window_progress"] for record in violations]
    histogram = Counter(progress)
    blockage = [record["terminal"]["blockage_gap"] >= .002 for record in violations]
    base_names = ("finite", "jaw_closed", "jaw_safe", "in_corridor",
                  "jaw_blocked", "bilateral_force", "cube_force")
    base_ready = [all(record["terminal"]["candidate_flags"].get(name, False)
                      for name in base_names) for record in violations]
    post_blockage_actions = []
    for record in violations:
        first = next((index for index, row in enumerate(record["trace"])
                      if row["blockage_gap"] >= .002), None)
        if first is not None:
            post_blockage_actions.extend(row["action"][:3] for row in record["trace"][first:])
    def mean_vector(name, source):
        return [mean(record["terminal"][name][index] for record in source)
                for index in range(3)]
    return {
        "episodes": len(records), "workspace_violation_count": len(violations),
        "axis_histogram": dict(axes),
        "mean_overshoot_mm": mean(amounts) if amounts else 0.0,
        "max_overshoot_mm": max(amounts, default=0.0),
        "ee_actually_outside_count": sum(
            not record["terminal"]["ee_inside_workspace"] for record in violations),
        "preclamp_target_outside_count": sum(
            not record["terminal"]["preclamp_target_inside_workspace"] for record in violations),
        "persistent_target_outside_count": sum(
            not record["terminal"]["persistent_target_inside_workspace"] for record in violations),
        "blockage_ge_2mm_count": sum(blockage),
        "candidate_base_evidence_ready_count": sum(base_ready),
        "candidate_only_missing_stability_window_count": sum(
            record["terminal"]["candidate_only_missing_stability_window"]
            for record in violations),
        "candidate_window_progress_histogram": {
            str(value): histogram.get(value, 0) for value in range(7)},
        "mean_candidate_window_progress": mean(progress) if progress else 0.0,
        "mean_candidate_window_steps_remaining": (
            mean(6 - value for value in progress) if progress else 0.0),
        "mean_abs_xyz_action_after_blockage": [
            mean(abs(action[index]) for action in post_blockage_actions)
            if post_blockage_actions else 0.0 for index in range(3)],
        "post_blockage_action_samples": len(post_blockage_actions),
        "mean_violation_control_step": mean(
            record["terminal"]["control_step"] for record in violations),
        "mean_cube_xy": [mean(record["randomized_cube_xy"][index] for record in violations)
                         for index in range(2)],
        "cube_xy_range": [
            [min(record["randomized_cube_xy"][index] for record in violations),
             max(record["randomized_cube_xy"][index] for record in violations)]
            for index in range(2)],
        "mean_terminal_ee_xyz": mean_vector("ee_xyz", violations),
        "mean_terminal_target_preclamp_xyz": mean_vector(
            "target_preclamp_xyz", violations),
        "mean_terminal_target_postclamp_xyz": mean_vector(
            "target_postclamp_xyz", violations),
        "mean_terminal_cube_relative_ee": mean_vector("cube_relative_ee", violations),
        "mean_terminal_xyz_action": mean_vector("action", violations),
        "mean_terminal_jaw_command": mean(
            record["terminal"]["jaw_command"] for record in violations),
        "mean_terminal_actual_jaw": mean(
            mean(record["terminal"]["jaw_actual"]) for record in violations),
        "mean_terminal_blockage_gap_m": mean(
            record["terminal"]["blockage_gap"] for record in violations),
        "mean_terminal_bilateral_force": mean(
            record["terminal"]["bilateral_force_min"] for record in violations),
    }


def summarize_fixed(records):
    candidates = []
    for record in records:
        index = next(index for index, item in enumerate(record["trace"])
                     if item["candidate_event"])
        row = record["trace"][index]
        candidates.append({
            "episode": record["episode"], "randomized_cube_xy": record["randomized_cube_xy"],
            "candidate_control_step": row["control_step"],
            "candidate_window_duration_steps": row["candidate_window_required"],
            "ee_xyz": row["ee_xyz"], "target_xyz": row["persistent_target_xyz"],
            "cube_relative_ee": row["cube_relative_ee"], "xyz_action": row["action"][:3],
            "ee_workspace_margin_m": row["ee_workspace_margin_m"],
            "target_workspace_margin_m": row["target_workspace_margin_m"],
        })
    return {
        "episodes": len(records), "success_count": sum(record["success"] for record in records),
        "workspace_violation_count": sum(
            record["failure_reason"] == "WORKSPACE_VIOLATION" for record in records),
        "candidate_snapshots": candidates,
        "mean_candidate_window_duration_steps": mean(
            item["candidate_window_duration_steps"] for item in candidates),
        "mean_candidate_target_margin_m": [
            mean(item["target_workspace_margin_m"][index] for item in candidates)
            for index in range(3)],
    }


def analyze_existing():
    randomized = json.loads((OUTPUT / "randomized" / "episodes.json").read_text())
    fixed = json.loads((OUTPUT / "fixed_reference" / "episodes.json").read_text())
    random_summary = summarize_violations(randomized)
    fixed_summary = summarize_fixed(fixed)
    race = (random_summary["candidate_only_missing_stability_window_count"] >= 10
            and random_summary["blockage_ge_2mm_count"] >= 10)
    real = random_summary["preclamp_target_outside_count"] == 20
    detector_bug = (random_summary["preclamp_target_outside_count"] == 0
                    and random_summary["workspace_violation_count"] > 0)
    diagnosis = ("CANDIDATE_WINDOW_RACE" if race else
                 "REAL_WORKSPACE_OVERSHOOT" if real else
                 "WORKSPACE_DETECTOR_BUG" if detector_bug else "OTHER")
    result = {
        "P2_R7_DIAGNOSIS": diagnosis,
        "checkpoint": {"randomized": str(RANDOM_CHECKPOINT),
                       "randomized_sha256": sha256(RANDOM_CHECKPOINT),
                       "fixed": str(FIXED_CHECKPOINT),
                       "fixed_sha256": sha256(FIXED_CHECKPOINT)},
        "locked": True, "training_transitions": 0,
        "randomized_summary": random_summary, "fixed_summary": fixed_summary,
        "implementation_bug_found": detector_bug,
        "workspace_change_required": False,
        "ppo_retraining_required_for_diagnosis": False,
    }
    write_json(OUTPUT / "randomized" / "summary.json", random_summary)
    write_json(OUTPUT / "fixed_reference" / "summary.json", fixed_summary)
    write_json(OUTPUT / "diagnosis.json", result)
    return result


def diagnose():
    randomized = run_traces(RANDOM_CHECKPOINT, .006, 20)
    fixed = run_traces(FIXED_CHECKPOINT, 0.0, 10, require_success=True)
    write_json(OUTPUT / "randomized" / "episodes.json", randomized)
    write_json(OUTPUT / "fixed_reference" / "episodes.json", fixed)
    return analyze_existing()


def finalize():
    diagnosis = json.loads((OUTPUT / "diagnosis.json").read_text())
    catalog = task_catalog_document()
    ids = [item.id for item in PRODUCTION_TASK_REGISTRY.definitions()]
    production = {"task_count": len(ids), "catalog_version": catalog["catalog_version"],
                  "catalog_digest": catalog["catalog_digest"],
                  "experimental_tasks_registered": sorted(
                      set(ids).intersection({"s1_grasp", "s1_lift", "s1_place"}))}
    production["passed"] = (len(ids) == 7 and catalog["catalog_version"] == "task-catalog/v1"
                            and catalog["catalog_digest"] == EXPECTED_DIGEST
                            and not production["experimental_tasks_registered"])
    write_json(OUTPUT / "production_regression.json", production)
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"experiment": "P2-R7", "diagnosis": diagnosis["P2_R7_DIAGNOSIS"],
                "training_transitions": 0, "production_regression": production,
                "files": files}
    write_json(OUTPUT / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("diagnose", "analyze", "finalize"))
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    result = (diagnose() if args.mode == "diagnose" else
              analyze_existing() if args.mode == "analyze" else finalize())
    print("P2_R7_%s=%s" % (args.mode.upper(), json.dumps(result, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
