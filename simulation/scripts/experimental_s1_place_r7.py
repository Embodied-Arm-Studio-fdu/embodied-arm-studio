#!/usr/bin/env python3
"""P4-R7 Transport robustness fine-tune and staged Full Place gate."""

import argparse
from collections import Counter
from dataclasses import asdict
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

import experimental_s1_place_staged as staged
from simulation_core.evaluation import wilson_confidence_interval
from simulation_core.experimental.manipulation.place_env import PlaceEnvConfig, PlaceTransportEnvironment
from simulation_core.experimental.manipulation.place_ppo import ExperimentalPlacePPOTrainer
from simulation_core.experimental.manipulation.place_schema import PLACE_POLICY_SCHEMA
from simulation_core.ppo.policy import flatten_observation
from simulation_core.ppo.trainer import PPOConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_r7_s1_place"
ARTIFACTS_ROOT = ROOT / "artifacts" / "experimental"
PROJECT_ID = "p4_r7_s1_place"
SOURCE = (ROOT / "artifacts" / "experimental" / "p4_1_s1_place" /
          "train_20k" / "checkpoints" / "model_latest.pt")
LOWER = (ROOT / "artifacts" / "experimental" / "p4_2_s1_place" /
         "train_20k" / "checkpoints" / "model_latest.pt")
SOURCE_SHA = "eb9fab2287ffd6a735dc27a1ccccd8eb3c71678617b4aef2bac1561d556cf291"
LOWER_SHA = "58ddf5e91a29ac087d37dac9094dad77a3596f5e32693fe5ad3f8e78b9dcba33"


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed=20261003):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ppo_config():
    return PPOConfig(rollout_steps=200, update_epochs=2, batch_size=32,
                     learning_rate=3e-4, gamma=.99, gae_lambda=.95,
                     clip_ratio=.2, value_coefficient=.5,
                     entropy_coefficient=.01, max_grad_norm=.5)


def checkpoint_metadata(path):
    path = Path(path); payload = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path), "sha256": sha256(path), "task_id": payload.get("task_id"),
            "training_steps": int(payload["training_steps"]),
            "training_transitions_total": int(payload.get("training_transitions_total",
                                                           payload["training_steps"])),
            "active_action_mask": list(payload.get("active_action_mask") or ()),
            "normalizer_count": int(payload["normalizer_state"]["count"])}


def train(run_name, transitions, resume):
    if transitions <= 0 or transitions % 200:
        raise ValueError("transitions must be a positive multiple of 200")
    if sha256(SOURCE) != SOURCE_SHA or sha256(LOWER) != LOWER_SHA:
        raise ValueError("locked checkpoint SHA mismatch")
    output = OUTPUT / run_name; output.mkdir(parents=True, exist_ok=True)
    seed_everything(); env = PlaceTransportEnvironment(PlaceEnvConfig())
    records = []; start = time.time(); cleanup = False
    try:
        env.initialize(); env.reset()
        config = ppo_config()
        trainer = ExperimentalPlacePPOTrainer(env, config=config, device="cuda:0")
        source_payload = trainer.load_checkpoint(Path(resume))
        start_steps = trainer.training_steps
        write_json(output / "config.json", {
            "task_id": "s1_place", "phase": "TRANSPORT", "physics_action_mask": [1,1,0,0],
            "reward_modified": False, "environment": asdict(env.config), "ppo": asdict(config),
            "resume": checkpoint_metadata(resume), "locked_source_sha256": SOURCE_SHA})
        best_rank = None
        for local in range(1, transitions // config.rollout_steps + 1):
            metrics = trainer.train_one_update()
            row = {"local_update": local, "update_index": trainer.update_steps,
                   "training_steps": trainer.training_steps, **metrics}
            records.append(row); append_jsonl(output / "metrics.jsonl", row)
            for episode in trainer._rollout_episode_records:
                append_jsonl(output / "episodes.jsonl", episode)
            rank = (row.get("success_rate", 0.), row.get("target_reached_rate", 0.),
                    -row.get("mean_minimum_xy_error", 1.), row["average_reward"])
            if best_rank is None or rank > best_rank:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_best.pt")
                best_rank = rank
            if local % 5 == 0 or local == transitions // config.rollout_steps:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_latest.pt")
                print("update=%d total=%d local=%d reward=%.5f success=%.3f target=%.3f workspace=%.3f" % (
                    trainer.update_steps, trainer.training_steps, trainer.training_steps-start_steps,
                    row["average_reward"], row.get("success_rate", 0.),
                    row.get("target_reached_rate", 0.),
                    sum(bool(x.get("workspace_violation")) for x in trainer._rollout_episode_records)
                    / max(1, len(trainer._rollout_episode_records))), flush=True)
        latest = output / "checkpoints" / "model_latest.pt"
        reloader = ExperimentalPlacePPOTrainer(env, config=config, device="cuda:0")
        loaded = reloader.load_checkpoint(latest)
        reload_ok = reloader.training_steps == trainer.training_steps
        cleanup = True
    finally:
        env.close()
    result = {"task_id": "s1_place", "phase": "TRANSPORT", "run": run_name,
              "transitions_this_run": transitions, "start_training_steps": start_steps,
              "end_training_steps": int(loaded["training_steps"]),
              "source_training_steps": int(source_payload["training_steps"]),
              "reward_modified": False, "checkpoint_reload": reload_ok,
              "resource_cleanup": cleanup, "elapsed_seconds": time.time()-start,
              "latest_checkpoint": checkpoint_metadata(latest),
              "best_checkpoint": checkpoint_metadata(output / "checkpoints" / "model_best.pt"),
              "last_metrics": records[-1]}
    write_json(output / "training_summary.json", result)
    return result


def _margin(point, workspace):
    return min(*(point[i] - workspace.lower[i] for i in range(3)),
               *(workspace.upper[i] - point[i] for i in range(3)))


def transport_evaluate(run_name, checkpoint, episodes=50):
    seed_everything(); env = PlaceTransportEnvironment(); records = []; cleanup = False
    try:
        env.initialize(); trainer = ExperimentalPlacePPOTrainer(env, config=ppo_config(), device="cuda:0")
        trainer.load_checkpoint(Path(checkpoint)); trainer.policy.eval()
        for index in range(episodes):
            observation = env.reset(); done = False; reward_sum = 0.; steps = 0
            minimum_margin = _margin(env._target, env.config.workspace)
            while not done:
                raw = flatten_observation(observation, PLACE_POLICY_SCHEMA).to(trainer.device)
                normalized = trainer.normalizer.normalize(raw, update=False)
                with torch.no_grad(): action, _, _ = trainer.policy.act(normalized, deterministic=True)
                observation, reward, done, info = env.step(tuple(float(x) for x in action.cpu()))
                reward_sum += float(reward); steps += 1
                minimum_margin = min(minimum_margin, _margin(env._target, env.config.workspace))
            records.append({"episode": index, "success": bool(info["success"]),
                "target_reached": bool(info["target_reached"]),
                "workspace_violation": bool(info["workspace_violation"]),
                "failure_reason": info["failure_reason"] or "SUCCESS", "steps": steps,
                "reward": reward_sum, "minimum_xy_error": float(info["minimum_xy_error"]),
                "final_xy_error": float(info["xy_error"]),
                "minimum_persistent_target_workspace_margin": minimum_margin,
                "final_persistent_target_workspace_margin": _margin(env._target, env.config.workspace),
                "unsafe": bool(info["unsafe"]), "non_finite": bool(info["non_finite"])})
        cleanup = True
    finally:
        env.close()
    success = sum(row["success"] for row in records); target = sum(row["target_reached"] for row in records)
    workspace = sum(row["workspace_violation"] for row in records); lower, upper = wilson_confidence_interval(success, episodes)
    summary = {"task_id": "s1_place", "phase": "TRANSPORT", "deterministic": True,
        "episodes": episodes, "success_count": success, "success_rate": success/episodes,
        "target_reached_count": target, "workspace_violation_count": workspace,
        "workspace_violation_rate": workspace/episodes,
        "mean_minimum_xy_error": sum(x["minimum_xy_error"] for x in records)/episodes,
        "max_minimum_xy_error": max(x["minimum_xy_error"] for x in records),
        "mean_final_xy_error": sum(x["final_xy_error"] for x in records)/episodes,
        "max_final_xy_error": max(x["final_xy_error"] for x in records),
        "minimum_workspace_margin": min(x["minimum_persistent_target_workspace_margin"] for x in records),
        "mean_minimum_workspace_margin": sum(x["minimum_persistent_target_workspace_margin"] for x in records)/episodes,
        "mean_final_workspace_margin": sum(x["final_persistent_target_workspace_margin"] for x in records)/episodes,
        "failure_reason_histogram": dict(Counter(x["failure_reason"] for x in records)),
        "unsafe_count": sum(x["unsafe"] for x in records),
        "non_finite_count": sum(x["non_finite"] for x in records),
        "wilson_95_lower": lower, "wilson_95_upper": upper,
        "gate_passed": success/episodes >= .95 and workspace/episodes <= .05,
        "checkpoint": checkpoint_metadata(checkpoint), "resource_cleanup": cleanup}
    directory = OUTPUT / run_name / "transport_eval_50"
    write_json(directory / "summary.json", summary)
    for row in records: append_jsonl(directory / "episodes.jsonl", row)
    return summary


def runtimes(checkpoint):
    transport_sha = sha256(checkpoint)
    return (staged.PolicyRuntime(checkpoint, transport_sha, (1,1,0,0), "transport"),
            staged.PolicyRuntime(LOWER, LOWER_SHA, (0,0,1,1), "lower"))


def full_evaluate(name, checkpoint, episodes):
    staged.OUTPUT = OUTPUT
    transport, lower = runtimes(checkpoint)
    return staged.evaluate(name, episodes, transport, lower)


def make_video(checkpoint):
    staged.OUTPUT = OUTPUT
    transport, lower = runtimes(checkpoint)
    return staged.video(transport, lower)


def finalize(checkpoint, training_transitions, extended, transport_summary,
             replay=None, final=None, video=None):
    result = {"task_id": "s1_place", "training_transitions": training_transitions,
        "extended_to_5k": bool(extended), "transport_checkpoint": checkpoint_metadata(checkpoint),
        "lower_checkpoint": checkpoint_metadata(LOWER), "transport_eval_50": transport_summary,
        "full_place_replay_20": replay, "final_100": final, "video": video,
        "isolation": staged.isolation()}
    result["final_100_run"] = final is not None
    result["video_run"] = video is not None
    if replay is not None and not staged.passed(replay):
        result["minimal_blocker_stage"] = "TRANSPORT_TO_LOWER_SWITCH"
        result["minimal_blocker"] = (
            "Fine-tuned policy satisfies the standalone 15 mm/1 s Transport gate, "
            "but does not enter the staged 10 mm/0.5 s switch condition before "
            "continuing to a workspace clamp.")
    write_json(OUTPUT / "production_isolation.json", result["isolation"])
    passed = bool(final and staged.passed(final, final=True) and result["isolation"]["passed"])
    result["p4_r7_full_place_gate"] = "PASS" if passed else "FAIL"
    result["p4_s1_place_ppo_gate"] = "PASS" if passed else "FAIL"
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    result["files"] = files; write_json(OUTPUT / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
        choices=("train", "transport-eval", "full-eval", "video"))
    parser.add_argument("--run", default="train_3k")
    parser.add_argument("--transitions", type=int, default=3000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--resume", type=Path, default=SOURCE)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    checkpoint = args.checkpoint or args.resume
    if args.mode == "train": result = train(args.run, args.transitions, args.resume)
    elif args.mode == "transport-eval": result = transport_evaluate(args.run, checkpoint, args.episodes)
    elif args.mode == "full-eval": result = full_evaluate(args.run, checkpoint, args.episodes)
    else: result = make_video(checkpoint)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
