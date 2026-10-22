#!/usr/bin/env python3
"""P4 completion diagnostic and staged release/full-place repair runner."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import torch

from simulation_core.experimental.manipulation.lower_release_env import LowerReleaseEnvironment
from simulation_core.experimental.manipulation.lower_release_ppo import (
    ExperimentalLowerReleasePPOTrainer, deterministic_evaluate,
)
from simulation_core.experimental.manipulation.lower_release_schema import (
    LOWER_RELEASE_POLICY_SCHEMA,
)
from simulation_core.ppo.policy import flatten_observation
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document
from simulation_core.experimental.manipulation.place_completion import (
    P4_2_20K_SHA256, RELEASE_ONLY_ACTIVE_ACTION_MASK,
    ReleaseOnlyEnvironment, StagePlaceTrainer,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_completion_s1_place"
P4_2_CHECKPOINT = (ROOT / "artifacts" / "experimental" / "p4_2_s1_place" /
                   "train_20k" / "checkpoints" / "model_latest.pt")


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def seed_everything(seed=20261003):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ppo_config():
    return PPOConfig(rollout_steps=200, update_epochs=2, batch_size=32,
                     learning_rate=3e-4, gamma=.99, gae_lambda=.95,
                     clip_ratio=.2, value_coefficient=.5,
                     entropy_coefficient=.01, max_grad_norm=.5)


def load_trainer(env, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = ExperimentalLowerReleasePPOTrainer(
        env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
    trainer.load_checkpoint(Path(checkpoint))
    return trainer


def diagnostic(episodes=20):
    env = LowerReleaseEnvironment(); rows = []; episodes_out = []; cleanup = False
    try:
        env.initialize(); trainer = load_trainer(env, P4_2_CHECKPOINT)
        trainer.policy.eval()
        for episode in range(episodes):
            observation = env.reset(); done = False; legal_rows = []; steps = 0
            while not done:
                vector = flatten_observation(observation, LOWER_RELEASE_POLICY_SCHEMA).to(trainer.device)
                normalized = trainer.normalizer.normalize(vector, update=False)
                with torch.no_grad():
                    action, _, _ = trainer.policy.act(normalized, deterministic=True)
                observation, _, done, info = env.step(
                    tuple(float(value) for value in action.detach().cpu()))
                steps += 1
                if info["bottom_clearance"] <= env.config.legal_release_clearance:
                    row = {"episode": episode, "step": steps,
                           **{name: info[name] for name in (
                               "bottom_clearance", "raw_policy_grip_action",
                               "absolute_grip_request", "rate_limited_jaw_command",
                               "jaw_actual", "blockage_gap", "release_started", "detached")}}
                    legal_rows.append(row); rows.append(row)
            episodes_out.append({"episode": episode, "steps": steps,
                                 "failure_reason": info["failure_reason"],
                                 "legal_height_samples": len(legal_rows),
                                 "release_started": info["release_started"],
                                 "detached": info["detached"],
                                 "minimum_grip_action": min((r["raw_policy_grip_action"]
                                                            for r in legal_rows), default=None),
                                 "maximum_grip_action": max((r["raw_policy_grip_action"]
                                                            for r in legal_rows), default=None),
                                 "maximum_grip_request": max((r["absolute_grip_request"]
                                                             for r in legal_rows), default=None),
                                 "maximum_jaw_command": max((r["rate_limited_jaw_command"]
                                                            for r in legal_rows), default=None),
                                 "maximum_actual_jaw": max((r["jaw_actual"]
                                                           for r in legal_rows), default=None),
                                 "minimum_blockage_gap": min((r["blockage_gap"]
                                                             for r in legal_rows), default=None)})
        cleanup = True
    finally:
        env.close()
    grip = [float(row["raw_policy_grip_action"]) for row in rows]
    request = [float(row["absolute_grip_request"]) for row in rows]
    command = [float(row["rate_limited_jaw_command"]) for row in rows]
    actual = [float(row["jaw_actual"]) for row in rows]
    blockage = [float(row["blockage_gap"]) for row in rows]
    sustained_open = sum(value >= .5 for value in grip) / max(1, len(grip))
    if sustained_open < .8 and max(request, default=0.0) < .040:
        classification = "A_RELEASE_POLICY_NOT_LEARNED"
    elif max(actual, default=0.0) < .032:
        classification = "B_ACTION_MAPPING_OR_RATE_LIMITER"
    else:
        classification = "C_DETACH_VERIFIER"
    summary = {"episodes": episodes, "deterministic": True,
               "checkpoint": str(P4_2_CHECKPOINT),
               "checkpoint_sha256": hashlib.sha256(P4_2_CHECKPOINT.read_bytes()).hexdigest(),
               "legal_height_samples": len(rows),
               "mean_raw_policy_grip_action": sum(grip) / max(1, len(grip)),
               "min_raw_policy_grip_action": min(grip, default=None),
               "max_raw_policy_grip_action": max(grip, default=None),
               "mean_absolute_grip_request": sum(request) / max(1, len(request)),
               "max_absolute_grip_request": max(request, default=None),
               "max_rate_limited_jaw_command": max(command, default=None),
               "max_actual_jaw": max(actual, default=None),
               "min_blockage_gap": min(blockage, default=None),
               "release_started_count": sum(row["release_started"] for row in episodes_out),
               "detached_count": sum(row["detached"] for row in episodes_out),
               "sustained_open_sample_fraction": sustained_open,
               "classification": classification, "training_performed": False,
               "resource_cleanup": cleanup}
    directory = OUTPUT / "diagnostic_20"
    write_json(directory / "summary.json", summary)
    write_json(directory / "episodes.json", episodes_out)
    with (directory / "legal_height_steps.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def release_train(run_name, transitions, resume=None):
    if transitions <= 0 or transitions % 200:
        raise ValueError("transitions must be a positive multiple of 200")
    seed_everything(); output = OUTPUT / run_name; output.mkdir(parents=True, exist_ok=True)
    env = ReleaseOnlyEnvironment(); config = ppo_config(); records = []; best = None
    cleanup = False; start = time.time(); payload = None
    try:
        env.initialize(); env.reset()
        trainer = StagePlaceTrainer(env, active_action_mask=RELEASE_ONLY_ACTIVE_ACTION_MASK,
                                    stage_name="release_only", config=config, device="cuda:0")
        if resume: trainer.load_checkpoint(Path(resume))
        else: trainer.initialize_from_checkpoint(P4_2_CHECKPOINT, P4_2_20K_SHA256)
        write_json(output / "config.json", {
            "stage": "release_only", "active_action_mask": list(RELEASE_ONLY_ACTIVE_ACTION_MASK),
            "transitions_this_run": transitions, "source": str(resume or P4_2_CHECKPOINT),
            "reset_bottom_clearance_m": [0.006, 0.010], "scripted_release": False})
        for local in range(1, transitions // config.rollout_steps + 1):
            metrics = trainer.train_one_update()
            row = {"update_index": trainer.update_steps, "training_steps": trainer.training_steps,
                   **metrics}; records.append(row); append_jsonl(output / "metrics.jsonl", row)
            for episode in trainer._rollout_episode_records:
                append_jsonl(output / "episodes.jsonl", episode)
            rank = (row.get("detached_rate", 0.0), row.get("success_rate", 0.0),
                    row.get("settled_1s_rate", 0.0), row["average_reward"])
            if best is None or rank > best:
                trainer.save_checkpoint(OUTPUT.parent, OUTPUT.name, run_name, "model_best.pt")
                best = rank
            if local % 5 == 0 or local == transitions // config.rollout_steps:
                trainer.save_checkpoint(OUTPUT.parent, OUTPUT.name, run_name, "model_latest.pt")
                print("release update=%d steps=%d detach=%.3f success=%.3f release=%.3f reward=%.4f" % (
                    trainer.update_steps, trainer.training_steps,
                    row.get("detached_rate", 0.0), row.get("success_rate", 0.0),
                    row.get("release_started_rate", 0.0), row["average_reward"]), flush=True)
        latest = output / "checkpoints" / "model_latest.pt"
        reloader = StagePlaceTrainer(env, active_action_mask=RELEASE_ONLY_ACTIVE_ACTION_MASK,
                                     stage_name="release_only", config=config, device="cuda:0")
        payload = reloader.load_checkpoint(latest); cleanup = True
    finally:
        env.close()
    result = {"run": run_name, "passed": bool(records and cleanup),
              "transitions_this_run": transitions,
              "stage_training_steps": int(payload["training_steps"]),
              "elapsed_seconds": time.time() - start, "last_metrics": records[-1],
              "latest_checkpoint": str(latest),
              "latest_sha256": hashlib.sha256(latest.read_bytes()).hexdigest(),
              "best_checkpoint": str(output / "checkpoints" / "model_best.pt"),
              "resource_cleanup": cleanup}
    write_json(output / "training_summary.json", result); return result


def release_eval(run_name, checkpoint, episodes=30):
    env = ReleaseOnlyEnvironment(); cleanup = False
    try:
        env.initialize(); payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trainer = StagePlaceTrainer(env, active_action_mask=RELEASE_ONLY_ACTIVE_ACTION_MASK,
                                    stage_name="release_only",
                                    config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
        trainer.load_checkpoint(Path(checkpoint))
        summary, records = deterministic_evaluate(env, trainer, episodes); cleanup = True
    finally: env.close()
    summary.update({"stage": "release_only", "active_action_mask": [0,0,0,1],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    "resource_cleanup": cleanup})
    directory = OUTPUT / run_name / "evaluation"
    write_json(directory / "summary.json", summary)
    with (directory / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in records: stream.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def finalize():
    diagnostic_summary = json.loads((OUTPUT / "diagnostic_20" / "summary.json").read_text())
    release_3k = json.loads((OUTPUT / "release_3k" / "evaluation" / "summary.json").read_text())
    release_6k = json.loads((OUTPUT / "release_6k" / "evaluation" / "summary.json").read_text())
    catalog = task_catalog_document()
    isolation = {"catalog_version": catalog["catalog_version"],
                 "catalog_digest": catalog["catalog_digest"],
                 "production_task_count": len(catalog["tasks"]),
                 "experimental_registered": [row["id"] for row in catalog["tasks"]
                    if row["id"] in ("s1_grasp", "s1_lift", "s1_place")]}
    isolation["passed"] = (isolation["catalog_version"] == "task-catalog/v1"
                            and isolation["production_task_count"] == 7
                            and not isolation["experimental_registered"])
    write_json(OUTPUT / "production_isolation.json", isolation)
    release_gate = (release_6k["detached_count"] >= math.ceil(.8 * release_6k["episodes"])
                    and release_6k["unsafe_count"] == 0
                    and release_6k["non_finite_count"] == 0
                    and release_6k["release_too_high_count"] == 0)
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    result = {"stage": "P4 Completion", "diagnostic_classification":
              diagnostic_summary["classification"], "release_only_3k": release_3k,
              "release_only_6k": release_6k,
              "release_only_gate": "PASS" if release_gate else "FAIL",
              "repaired_lower_release_run": False, "full_place_run": False,
              "p4_2_repaired_gate": "NOT_REACHED",
              "p4_full_place_gate": "NOT_REACHED",
              "p4_s1_place_ppo_gate": "FAIL",
              "blocker": "SUSTAINED_OPEN_INSUFFICIENT_FOR_DETACH_AFTER_6K",
              "isolation": isolation, "files": files}
    write_json(OUTPUT / "manifest.json", result); return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("diagnostic", "release-train", "release-eval",
                                            "finalize"),
                        required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--transitions", type=int, default=3000)
    parser.add_argument("--run", default="release_3k")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    if args.mode == "diagnostic": result = diagnostic(args.episodes)
    elif args.mode == "release-train":
        result = release_train(args.run, args.transitions, args.resume)
    elif args.mode == "release-eval":
        if not args.checkpoint: parser.error("--checkpoint is required")
        result = release_eval(args.run, args.checkpoint, args.episodes)
    else: result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
