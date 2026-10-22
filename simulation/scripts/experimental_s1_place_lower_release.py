#!/usr/bin/env python3
"""P4-2 smoke, PPO training, deterministic evaluation, video, and gate."""

import argparse
from dataclasses import asdict, replace
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

import cv2
import numpy as np
import torch

from simulation_core.experimental.manipulation.lower_release_env import (
    LOWER_RELEASE_BOOTSTRAP_VERSION, LOWER_RELEASE_TARGET,
    LowerReleaseEnvConfig, LowerReleaseEnvironment,
)
from simulation_core.experimental.manipulation.lower_release_ppo import (
    P4_1_CHECKPOINT_SHA256, ExperimentalLowerReleasePPOTrainer,
    deterministic_evaluate,
)
from simulation_core.experimental.manipulation.lower_release_schema import (
    LOWER_RELEASE_ACTIVE_ACTION_MASK, LOWER_RELEASE_REWARD_VERSION,
    LOWER_RELEASE_SUCCESS_CRITERIA_VERSION,
)
from simulation_core.experimental.manipulation.place_schema import (
    PLACE_ACTION_SCHEMA_ID, PLACE_OBSERVATION_SCHEMA_ID,
)
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p4_2_s1_place"
ARTIFACTS_ROOT = ROOT / "artifacts" / "experimental"
PROJECT_ID = "p4_2_s1_place"
P4_1_CHECKPOINT = (ROOT / "artifacts" / "experimental" / "p4_1_s1_place" /
                   "train_20k" / "checkpoints" / "model_latest.pt")
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


def contract():
    return {"task_id": "s1_place", "experimental": True,
            "observation_schema": PLACE_OBSERVATION_SCHEMA_ID, "observation_dim": 32,
            "action_schema": PLACE_ACTION_SCHEMA_ID, "action_dim": 4,
            "active_action_mask": list(LOWER_RELEASE_ACTIVE_ACTION_MASK),
            "reward_version": LOWER_RELEASE_REWARD_VERSION,
            "success_version": LOWER_RELEASE_SUCCESS_CRITERIA_VERSION,
            "bootstrap_version": LOWER_RELEASE_BOOTSTRAP_VERSION,
            "target_center": list(LOWER_RELEASE_TARGET),
            "target_xy_tolerance_m": .015, "settle_seconds": 1.0,
            "release_authority": "policy"}


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
            "stage_training_transitions": payload.get("stage_training_transitions"),
            "parent_checkpoint": payload.get("parent_checkpoint")}


def run_policy_episode(env, name, rng=None):
    env.reset(); done = False; total = 0.0; steps = 0
    while not done:
        action = ((0., 0., 0., -1.) if name == "zero"
                  else tuple(rng.uniform(-1., 1.) for _ in range(4)))
        _, reward, done, info = env.step(action); total += reward; steps += 1
    return {"policy": name, "reward": total, "steps": steps,
            **{key: info.get(key) for key in (
                "success", "failure_reason", "descended_to_release_height",
                "release_started", "detached", "table_contact", "early_release",
                "table_penetration", "unsafe", "non_finite", "active_action_mask")}}


def smoke():
    output = OUTPUT / "smoke"; seed_everything(); env = LowerReleaseEnvironment()
    cleanup = False
    try:
        env.initialize(); observation = env.reset(); before = tuple(env._target)
        _, _, _, mask_info = env.step((1., -1., 0., -1.))
        mask_ok = (mask_info["applied_action"] == (0., 0., 0., -1.)
                   and abs(env._target[0] - before[0]) < 1e-9
                   and abs(env._target[1] - before[1]) < 1e-9)
        reset_ok = (len(sum((tuple(v) for v in observation.values()), ())) == 32
                    and env._bootstrap_last["target_above_valid"])
        records = [run_policy_episode(env, "zero"),
                   run_policy_episode(env, "random", random.Random(17))]
        ppo = ppo_config(64)
        trainer = ExperimentalLowerReleasePPOTrainer(env, config=ppo, device="cuda:0")
        trainer.initialize_from_p4_1(P4_1_CHECKPOINT)
        metrics = trainer.train_one_update()
        latest = trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, "smoke")
        reloader = ExperimentalLowerReleasePPOTrainer(env, config=ppo, device="cuda:0")
        reloader.load_checkpoint(latest)
        reload_ok = reloader.training_steps == trainer.training_steps
        cleanup = True
    finally:
        env.close()
    source_sha = hashlib.sha256(P4_1_CHECKPOINT.read_bytes()).hexdigest()
    result = {"passed": bool(mask_ok and reset_ok and reload_ok and cleanup
                              and source_sha == P4_1_CHECKPOINT_SHA256
                              and all(not row["unsafe"] and not row["non_finite"]
                                      for row in records)),
              "gpu": torch.cuda.get_device_name(0), "reset_verified": reset_ok,
              "action_mask_verified": mask_ok, "reward_and_termination_rollouts": records,
              "ppo_update": metrics, "checkpoint_reload": reload_ok,
              "source_checkpoint_sha256": source_sha,
              "checkpoint": checkpoint_metadata(latest), "resource_cleanup": cleanup,
              "contract": contract()}
    write_json(output / "summary.json", result); return result


def train(run_name, transitions, resume=None):
    if transitions <= 0 or transitions % 200:
        raise ValueError("transitions must be a positive multiple of 200")
    output = OUTPUT / run_name; output.mkdir(parents=True, exist_ok=True)
    seed_everything(); config = LowerReleaseEnvConfig(); ppo = ppo_config()
    env = LowerReleaseEnvironment(config); cleanup = False; records = []; best = None
    start = time.time(); payload = None
    try:
        env.initialize(); env.reset()
        catalog = task_catalog_document()
        write_json(output / "config.json", {"contract": contract(),
            "environment": asdict(config), "ppo": asdict(ppo),
            "network": {"hidden_sizes": [64, 64], "activation": "tanh"},
            "initial_checkpoint": str(resume or P4_1_CHECKPOINT),
            "catalog_version": catalog["catalog_version"],
            "catalog_digest": catalog["catalog_digest"]})
        trainer = ExperimentalLowerReleasePPOTrainer(env, config=ppo, device="cuda:0")
        if resume: trainer.load_checkpoint(Path(resume))
        else: trainer.initialize_from_p4_1(P4_1_CHECKPOINT)
        total_updates = transitions // ppo.rollout_steps
        for local in range(1, total_updates + 1):
            metrics = trainer.train_one_update()
            row = {"update_index": trainer.update_steps, "training_steps": trainer.training_steps,
                   **metrics}; records.append(row); append_jsonl(output / "metrics.jsonl", row)
            for episode in trainer._rollout_episode_records:
                append_jsonl(output / "episodes.jsonl", episode)
            rank = (row.get("success_rate", 0.), row.get("settled_1s_rate", 0.),
                    row.get("detached_rate", 0.), row.get("descended_rate", 0.),
                    row["average_reward"])
            if best is None or rank > best:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_best.pt")
                best = rank
            if local % 5 == 0 or local == total_updates:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_latest.pt")
                print("update=%d steps=%d reward=%.5f success=%.3f descend=%.3f "
                      "release=%.3f detach=%.3f" % (
                          trainer.update_steps, trainer.training_steps, row["average_reward"],
                          row.get("success_rate", 0.), row.get("descended_rate", 0.),
                          row.get("release_started_rate", 0.), row.get("detached_rate", 0.)),
                      flush=True)
        latest = output / "checkpoints" / "model_latest.pt"
        reloader = ExperimentalLowerReleasePPOTrainer(env, config=ppo, device="cuda:0")
        payload = reloader.load_checkpoint(latest)
        reload_ok = reloader.training_steps == trainer.training_steps
        cleanup = True
    finally:
        env.close()
    result = {"passed": bool(records and reload_ok and cleanup), "run": run_name,
              "transitions_this_run": transitions,
              "stage_training_steps_checkpoint": int(payload["training_steps"]),
              "elapsed_seconds": time.time() - start, "checkpoint_reload": reload_ok,
              "resource_cleanup": cleanup, "last_metrics": records[-1],
              "best_checkpoint": checkpoint_metadata(output / "checkpoints" / "model_best.pt"),
              "latest_checkpoint": checkpoint_metadata(latest)}
    write_json(output / "training_summary.json", result); return result


def load_trainer(env, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = ExperimentalLowerReleasePPOTrainer(
        env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
    trainer.load_checkpoint(Path(checkpoint)); return trainer


def evaluate(run_name, checkpoint, episodes=50):
    output = OUTPUT / run_name / "evaluation"; env = LowerReleaseEnvironment(); cleanup = False
    try:
        env.initialize(); trainer = load_trainer(env, checkpoint)
        summary, records = deterministic_evaluate(env, trainer, episodes); cleanup = True
    finally:
        env.close()
    summary.update({"checkpoint": checkpoint_metadata(checkpoint),
                    "resource_cleanup": cleanup, "contract": contract()})
    write_json(output / "summary.json", summary)
    for row in records: append_jsonl(output / "episodes.jsonl", row)
    return summary


class TargetZoneRecorder:
    """Camera overlay projected from the true target XY; it never enters physics."""
    def __init__(self, recorder, env_getter):
        self.recorder, self.env_getter = recorder, env_getter

    @staticmethod
    def _project(config, point):
        camera = np.asarray(config.position, dtype=float)
        target = np.asarray(config.target, dtype=float)
        forward = target - camera; forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray((0., 0., 1.))); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = np.asarray(point) - camera; depth = float(relative @ forward)
        if depth <= 1e-6: return None
        focal = config.width / (2 * math.tan(math.radians(config.horizontal_fov) / 2))
        return (int(round(config.width / 2 + focal * float(relative @ right) / depth)),
                int(round(config.height / 2 - focal * float(relative @ up) / depth)))

    def write(self, frame):
        env = self.env_getter(); pixels = frame.data.copy(); cfg = env.scene.camera.config
        z = env.scene.config.table_top + .001
        center = self._project(cfg, (LOWER_RELEASE_TARGET[0], LOWER_RELEASE_TARGET[1], z))
        edge = self._project(cfg, (LOWER_RELEASE_TARGET[0] + .015,
                                   LOWER_RELEASE_TARGET[1], z))
        if center is not None and edge is not None:
            radius = max(5, int(math.hypot(edge[0] - center[0], edge[1] - center[1])))
            cv2.circle(pixels, center, radius, (40, 255, 255, 255), 3, cv2.LINE_AA)
            cv2.drawMarker(pixels, center, (40, 255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(pixels, "TARGET ZONE", (center[0] - 70, center[1] + radius + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 255, 255, 255), 2, cv2.LINE_AA)
        if env.phase in ("GRASP_BOOTSTRAP", "GRASPED_LIFTED_RESET", "PLACE_POLICY"):
            label = "GRASPED + LIFTED RESET / TARGET-ABOVE BOOTSTRAP"
        elif env._maximum_settle_steps > 0:
            label = "STABLE PLACEMENT HOLD (1.0 S)"
        elif env._detached:
            label = "CUBE DETACHED / TABLE CONTACT"
        elif env._release_started:
            label = "POLICY RELEASE"
        else:
            label = "PLACE LOWER/RELEASE POLICY / DESCENT"
        cv2.rectangle(pixels, (8, 8), (632, 48), (0, 0, 0, 255), -1)
        cv2.putText(pixels, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (70, 255, 90, 255), 2, cv2.LINE_AA)
        self.recorder.write(replace(frame, data=pixels))


def video(run_name, checkpoint):
    path = OUTPUT / run_name / "video" / "successful_lower_release.mp4"
    holder = {}; env = None; recorder = None; cleanup = False; summary = records = None
    attempts = 0
    try:
        for attempts in range(1, 11):
            recorder = ExperimentalMp4Recorder(path).start()
            overlay = TargetZoneRecorder(recorder, lambda: holder["env"])
            env = LowerReleaseEnvironment(with_camera=True, frame_callback=overlay.write)
            holder["env"] = env; env.initialize(); trainer = load_trainer(env, checkpoint)
            summary, records = deterministic_evaluate(env, trainer, 1)
            cleanup = recorder.close(); recorder = None; env.close(); env = None
            if summary["success_count"] == 1: break
        passed = summary is not None and summary["success_count"] == 1 and cleanup
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"passed": passed, "evaluation": summary,
              "episode": records[0] if records else None, "video": str(path),
              "resource_cleanup": cleanup, "recording_attempts": attempts,
              "marker": {"implementation": "camera_overlay_world_projection",
                         "target_xy": list(LOWER_RELEASE_TARGET[:2]), "radius_m": .015,
                         "physics_actor": False, "collision_shape": False,
                         "rigid_body": False, "mass": False,
                         "observation_reward_policy_input": False},
              "labels": ["TARGET ZONE", "GRASPED + LIFTED RESET",
                         "PLACE LOWER/RELEASE POLICY / DESCENT", "POLICY RELEASE",
                         "CUBE DETACHED / TABLE CONTACT", "STABLE PLACEMENT HOLD (1.0 S)"],
              "checkpoint": checkpoint_metadata(checkpoint)}
    write_json(path.with_suffix(".json"), result); return result


def finalize(run_name):
    summary = json.loads((OUTPUT / run_name / "evaluation" / "summary.json").read_text())
    catalog = task_catalog_document()
    isolation = {"catalog_version": catalog["catalog_version"],
                 "catalog_digest": catalog["catalog_digest"],
                 "production_task_count": len(catalog["tasks"]),
                 "experimental_registered": [row["id"] for row in catalog["tasks"]
                    if row["id"] in ("s1_grasp", "s1_lift", "s1_place")]}
    isolation["passed"] = (isolation["catalog_version"] == "task-catalog/v1"
        and isolation["catalog_digest"] == CATALOG_DIGEST
        and isolation["production_task_count"] == 7 and not isolation["experimental_registered"])
    write_json(OUTPUT / "production_isolation.json", isolation)
    systematic = any(summary[name] >= summary["episodes"] * .1 for name in (
        "early_release_count", "release_too_high_count", "drop_outside_target_count",
        "unstable_placement_count", "table_penetration_count"))
    gate = (summary["success_rate"] >= .9 and summary["unsafe_count"] == 0
            and summary["non_finite_count"] == 0 and not systematic
            and isolation["passed"])
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    result = {"stage": "P4-2", "gate": "PASS" if gate else "FAIL",
              "systematic_high_drop_penetration_unstable": systematic,
              "evaluation": summary, "isolation": isolation,
              "contract": contract(), "files": files}
    write_json(OUTPUT / "manifest.json", result); return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train", "eval", "video", "finalize"),
                        required=True)
    parser.add_argument("--run", default="train_10k")
    parser.add_argument("--transitions", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    if args.mode == "smoke": result = smoke()
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
