#!/usr/bin/env python3
"""P4-1 experimental Place transport probe, PPO, evaluation, and video."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import cv2
import torch

from simulation_core.experimental.manipulation.place_env import (
    PLACE_BOOTSTRAP_VERSION, PLACE_TARGET_SELECTION_VERSION,
    PlaceEnvConfig, PlaceTransportEnvironment,
)
from simulation_core.experimental.manipulation.place_ppo import (
    ExperimentalPlacePPOTrainer, deterministic_evaluate,
)
from simulation_core.experimental.manipulation.place_schema import (
    PLACE_ACTION_SCHEMA_ID, PLACE_ACTIVE_ACTION_MASK, PLACE_OBSERVATION_SCHEMA_ID,
    PLACE_POLICY_SCHEMA, PLACE_REWARD_VERSION, PLACE_SUCCESS_CRITERIA_VERSION,
)
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_1_s1_place"
ARTIFACTS_ROOT = ROOT / "artifacts" / "experimental"
PROJECT_ID = "p4_1_s1_place"
CATALOG_DIGEST = "sha256:c0f3feb0771f4ab636c61fa53ba0d49d5f2a489a224b6890ae38ed798423f791"


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


def ppo_config(rollout_steps=200):
    return PPOConfig(rollout_steps=rollout_steps, update_epochs=2, batch_size=32,
                     learning_rate=3e-4, gamma=.99, gae_lambda=.95, clip_ratio=.2,
                     value_coefficient=.5, entropy_coefficient=.01, max_grad_norm=.5)


def contract(target_center=None):
    return {"task_id": "s1_place", "experimental": True,
            "observation_schema": PLACE_OBSERVATION_SCHEMA_ID,
            "observation_dim": PLACE_POLICY_SCHEMA.observation_dim,
            "action_schema": PLACE_ACTION_SCHEMA_ID, "action_dim": 4,
            "active_action_mask": list(PLACE_ACTIVE_ACTION_MASK),
            "reward_version": PLACE_REWARD_VERSION,
            "success_version": PLACE_SUCCESS_CRITERIA_VERSION,
            "bootstrap_version": PLACE_BOOTSTRAP_VERSION,
            "target_selection_version": PLACE_TARGET_SELECTION_VERSION,
            "target_center": list(target_center) if target_center else None,
            "target_xy_tolerance_m": .015, "hold_seconds": 1.0}


def checkpoint_metadata(path):
    path = Path(path); payload = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_id": payload.get("task_id"),
            "observation_schema": payload.get("observation_schema_id"),
            "action_schema": payload.get("action_schema_id"),
            "action_dim": payload.get("action_dim"),
            "active_action_mask": list(payload.get("active_action_mask") or ()),
            "target_center": list(payload.get("target_center") or ()),
            "bootstrap_version": payload.get("bootstrap_version"),
            "reward_version": payload.get("reward_version"),
            "success_version": payload.get("success_criteria_version"),
            "network_config": payload.get("network_config"),
            "ppo_config": payload.get("ppo_config"),
            "training_transitions_total": payload.get("training_transitions_total")}


def probe():
    env = PlaceTransportEnvironment(); cleanup = False
    try:
        env.initialize(); result = env.scripted_path_probe(); cleanup = True
    finally:
        env.close()
    result.update({"resource_cleanup": cleanup,
                   "contract": contract(result.get("target_center"))})
    write_json(OUTPUT / "workspace_path_probe.json", result)
    return result


def run_policy_episode(env, name, rng=None):
    observation = env.reset(); done = False; total = 0.; steps = 0
    while not done:
        if name == "zero": action = (0., 0., 0., 0.)
        else: action = tuple(rng.uniform(-1., 1.) for _ in range(4))
        observation, reward, done, info = env.step(action); total += reward; steps += 1
    return {"policy": name, "reward": total, "steps": steps,
            **{key: info.get(key) for key in ("success", "failure_reason", "target_reached",
                "minimum_xy_error", "drop", "grasp_lost", "workspace_violation",
                "table_collision", "unsafe", "non_finite", "active_action_mask")}}


def smoke():
    output = OUTPUT / "smoke"; seed_everything(); env = PlaceTransportEnvironment()
    cleanup = False
    try:
        env.initialize(); env.reset()
        before = tuple(env._target)
        _, _, _, mask_info = env.step((0., 0., 1., -1.))
        mask_ok = (mask_info["applied_action"] == (0., 0., 0., 0.)
                   and abs(env._target[2] - before[2]) < .003)
        records = [run_policy_episode(env, "zero"),
                   run_policy_episode(env, "random", random.Random(17))]
        ppo = ppo_config(64); trainer = ExperimentalPlacePPOTrainer(env, config=ppo, device="cuda:0")
        metrics = trainer.train_one_update()
        latest = trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, "smoke", "model_latest.pt")
        reloader = ExperimentalPlacePPOTrainer(env, config=ppo, device="cuda:0")
        reloader.load_checkpoint(latest)
        reload_ok = reloader.training_steps == trainer.training_steps; cleanup = True
    finally:
        env.close()
    result = {"passed": mask_ok and reload_ok and cleanup
                        and all(not x["unsafe"] and not x["non_finite"] for x in records),
              "action_mask_verified": mask_ok, "policies": records,
              "ppo_update": metrics, "checkpoint_reload": reload_ok,
              "checkpoint": checkpoint_metadata(latest), "resource_cleanup": cleanup,
              "contract": contract(env.target_center)}
    write_json(output / "summary.json", result); return result


def train(run_name, transitions, resume=None):
    if transitions <= 0 or transitions % 200:
        raise ValueError("transitions must be a positive multiple of 200")
    output = OUTPUT / run_name; output.mkdir(parents=True, exist_ok=True)
    seed_everything(); config = PlaceEnvConfig(); ppo = ppo_config()
    env = PlaceTransportEnvironment(config); cleanup = False; records = []
    best = None; start = time.time(); reload_ok = False
    try:
        env.initialize(); env.reset()  # selects and locks the probed deterministic target
        catalog = task_catalog_document()
        write_json(output / "config.json", {"contract": contract(env.target_center),
            "environment": asdict(config), "target_selection": env.target_selection,
            "ppo": asdict(ppo), "network": {"hidden_sizes": [64,64], "activation": "tanh"},
            "catalog_version": catalog["catalog_version"],
            "catalog_digest": catalog["catalog_digest"]})
        trainer = ExperimentalPlacePPOTrainer(env, config=ppo, device="cuda:0")
        if resume: trainer.load_checkpoint(Path(resume))
        for local in range(1, transitions // ppo.rollout_steps + 1):
            metrics = trainer.train_one_update()
            row = {"update_index": trainer.update_steps, "training_steps": trainer.training_steps,
                   **metrics}; records.append(row); append_jsonl(output / "metrics.jsonl", row)
            for episode in trainer._rollout_episode_records:
                append_jsonl(output / "episodes.jsonl", episode)
            rank = (row.get("success_rate", 0.), row.get("target_reached_rate", 0.),
                    -row.get("mean_minimum_xy_error", 1.), row["average_reward"])
            if best is None or rank > best:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_best.pt")
                best = rank
            if local % 5 == 0 or local == transitions // ppo.rollout_steps:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_latest.pt")
                print("update=%d steps=%d reward=%.5f success=%.3f target=%.3f min_xy=%.4f" % (
                    trainer.update_steps, trainer.training_steps, row["average_reward"],
                    row.get("success_rate", 0.), row.get("target_reached_rate", 0.),
                    row.get("mean_minimum_xy_error", 0.)), flush=True)
        latest = output / "checkpoints" / "model_latest.pt"
        reloader = ExperimentalPlacePPOTrainer(env, config=ppo, device="cuda:0")
        payload = reloader.load_checkpoint(latest)
        reload_ok = reloader.training_steps == trainer.training_steps; cleanup = True
    finally:
        env.close()
    result = {"passed": bool(records and reload_ok and cleanup), "run": run_name,
              "transitions_this_run": transitions,
              "training_steps_checkpoint": int(payload["training_steps"]),
              "elapsed_seconds": time.time() - start, "checkpoint_reload": reload_ok,
              "resource_cleanup": cleanup, "last_metrics": records[-1],
              "best_checkpoint": checkpoint_metadata(output / "checkpoints" / "model_best.pt"),
              "latest_checkpoint": checkpoint_metadata(latest)}
    write_json(output / "training_summary.json", result); return result


def load_trainer(env, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = ExperimentalPlacePPOTrainer(
        env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
    trainer.load_checkpoint(Path(checkpoint)); return trainer


def evaluate(run_name, checkpoint, episodes=50):
    output = OUTPUT / run_name / "evaluation"; env = PlaceTransportEnvironment(); cleanup = False
    try:
        env.initialize(); trainer = load_trainer(env, checkpoint)
        summary, records = deterministic_evaluate(env, trainer, episodes); cleanup = True
    finally:
        env.close()
    summary.update({"checkpoint": checkpoint_metadata(checkpoint),
                    "resource_cleanup": cleanup, "contract": contract(env.target_center)})
    write_json(output / "summary.json", summary)
    for row in records: append_jsonl(output / "episodes.jsonl", row)
    return summary


class LabeledRecorder:
    def __init__(self, recorder, env_getter): self.recorder, self.env_getter = recorder, env_getter
    def write(self, frame):
        env = self.env_getter(); pixels = frame.data.copy()
        if env.phase in ("GRASP_BOOTSTRAP", "GRASPED_LIFTED_RESET"):
            label, color = "GRASPED + LIFTED RESET", (255, 220, 30, 255)
        elif env._target_reached:
            label, color = "TARGET ZONE / 1S HOLD", (50, 255, 70, 255)
        else: label, color = "PLACE POLICY START / XY TRANSPORT", (50, 255, 70, 255)
        cv2.rectangle(pixels, (12,12), (540,54), (0,0,0,255), -1)
        cv2.putText(pixels, label, (22,42), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, color, 2, cv2.LINE_AA)
        self.recorder.write(replace(frame, data=pixels))


def video(run_name, checkpoint):
    path = OUTPUT / run_name / "video" / "successful_transport.mp4"
    recorder = ExperimentalMp4Recorder(path).start(); holder = {}; env = None; cleanup = False
    labeled = LabeledRecorder(recorder, lambda: holder["env"])
    summary = None; records = None; attempts = 0
    try:
        env = PlaceTransportEnvironment(with_camera=True, frame_callback=labeled.write)
        holder["env"] = env; env.initialize(); trainer = load_trainer(env, checkpoint)
        # Reset physics has a small, deterministic-seed-independent variation.  Record
        # complete episodes until a genuine success is observed, overwriting the
        # previous failed take so the delivered video contains one successful take.
        for attempts in range(1, 11):
            summary, records = deterministic_evaluate(env, trainer, 1)
            cleanup = recorder.close(); recorder = None
            if summary["success_count"] == 1:
                break
            recorder = ExperimentalMp4Recorder(path).start()
            labeled.recorder = recorder
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"passed": summary["success_count"] == 1 and cleanup,
              "evaluation": summary, "episode": records[0], "video": str(path),
              "resource_cleanup": cleanup, "recording_attempts": attempts,
              "labels": ["GRASPED + LIFTED RESET", "PLACE POLICY START / XY TRANSPORT",
                         "TARGET ZONE / 1S HOLD"],
              "checkpoint": checkpoint_metadata(checkpoint)}
    write_json(path.with_suffix(".json"), result); return result


def finalize(run_name):
    summary = json.loads((OUTPUT / run_name / "evaluation" / "summary.json").read_text())
    catalog = task_catalog_document()
    isolation = {"catalog_version": catalog["catalog_version"],
                 "catalog_digest": catalog["catalog_digest"],
                 "production_task_count": len(catalog["tasks"]),
                 "experimental_registered": [x["id"] for x in catalog["tasks"]
                     if x["id"] in ("s1_grasp","s1_lift","s1_place")]}
    isolation["passed"] = (isolation["catalog_version"] == "task-catalog/v1"
        and isolation["catalog_digest"] == CATALOG_DIGEST
        and isolation["production_task_count"] == 7 and not isolation["experimental_registered"])
    write_json(OUTPUT / "production_isolation.json", isolation)
    systematic_collision = (summary["drop_count"] >= summary["episodes"] * .1
                            or summary["workspace_violation_count"] >= summary["episodes"] * .1
                            or summary["table_collision_count"] >= summary["episodes"] * .1)
    gate = (summary["success_rate"] >= .9 and summary["unsafe_count"] == 0
            and summary["non_finite_count"] == 0 and not systematic_collision
            and isolation["passed"])
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    result = {"stage": "P4-1", "gate": "PASS" if gate else "FAIL",
              "systematic_drop_workspace_table_collision": systematic_collision,
              "evaluation": summary, "isolation": isolation,
              "contract": contract(summary["target_center"]), "files": files}
    write_json(OUTPUT / "manifest.json", result); return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe","smoke","train","eval","video","finalize"), required=True)
    parser.add_argument("--run", default="train_10k")
    parser.add_argument("--transitions", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    if args.mode == "probe": result = probe()
    elif args.mode == "smoke": result = smoke()
    elif args.mode == "train": result = train(args.run, args.transitions, args.resume)
    elif args.mode == "eval":
        if not args.checkpoint: parser.error("--checkpoint is required")
        result = evaluate(args.run, args.checkpoint, args.episodes)
    elif args.mode == "video":
        if not args.checkpoint: parser.error("--checkpoint is required")
        result = video(args.run, args.checkpoint)
    else: result = finalize(args.run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
