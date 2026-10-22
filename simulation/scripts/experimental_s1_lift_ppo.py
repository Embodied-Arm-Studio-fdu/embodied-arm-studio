#!/usr/bin/env python3
"""P3-1 experimental-only S1 Lift PPO gates and artifacts."""

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

from simulation_core.experimental.manipulation.lift_env import (
    LIFT_BOOTSTRAP_VERSION, LiftEnvConfig, LiftPPOEnvironment,
)
from simulation_core.experimental.manipulation.lift_ppo import (
    ExperimentalLiftPPOTrainer, deterministic_evaluate,
)
from simulation_core.experimental.manipulation.lift_schema import (
    LIFT_ACTION_SCHEMA_ID, LIFT_OBSERVATION_SCHEMA_ID, LIFT_POLICY_SCHEMA,
    LIFT_REWARD_VERSION, LIFT_SUCCESS_CRITERIA_VERSION,
)
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p3_1_s1_lift"
ARTIFACTS_ROOT = ROOT / "artifacts" / "experimental"
PROJECT_ID = "p3_1_s1_lift"
EXPECTED_CATALOG_DIGEST = "sha256:c0f3feb0771f4ab636c61fa53ba0d49d5f2a489a224b6890ae38ed798423f791"


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
    return PPOConfig(
        rollout_steps=rollout_steps, update_epochs=2, batch_size=32,
        learning_rate=3.0e-4, gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, value_coefficient=0.5,
        entropy_coefficient=0.01, max_grad_norm=0.5,
    )


def contract_payload(config=None):
    config = config or LiftEnvConfig()
    return {
        "task_id": "s1_lift", "experimental": True,
        "observation_schema": LIFT_OBSERVATION_SCHEMA_ID,
        "observation_dim": LIFT_POLICY_SCHEMA.observation_dim,
        "action_schema": LIFT_ACTION_SCHEMA_ID, "action_dim": 1,
        "action": "dz_only_persistent_target_delta",
        "fixed_targets": ["x", "y", "orientation", "jaw"],
        "target_lift_m": config.target_lift,
        "target_band_m": list(config.target_band),
        "hold_seconds": config.hold_seconds,
        "reward_version": LIFT_REWARD_VERSION,
        "success_version": LIFT_SUCCESS_CRITERIA_VERSION,
        "bootstrap_version": LIFT_BOOTSTRAP_VERSION,
    }


def config_payload(env_config, ppo):
    catalog = task_catalog_document()
    return {
        **contract_payload(env_config), "environment": asdict(env_config),
        "ppo": asdict(ppo), "network": {"hidden_sizes": [64, 64], "activation": "tanh"},
        "catalog_version": catalog["catalog_version"],
        "catalog_digest": catalog["catalog_digest"],
    }


def checkpoint_metadata(path):
    path = Path(path); payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "task_id": payload.get("task_id"),
        "observation_schema": payload.get("observation_schema_id"),
        "action_schema": payload.get("action_schema_id"),
        "action_dim": payload.get("action_dim"),
        "target_lift": payload.get("target_lift"),
        "bootstrap_version": payload.get("bootstrap_version"),
        "reward_version": payload.get("reward_version"),
        "success_version": payload.get("success_criteria_version"),
        "network_config": payload.get("network_config"),
        "ppo_config": payload.get("ppo_config"),
        "training_transitions_total": payload.get("training_transitions_total"),
    }


def run_policy_episode(env, policy_name, rng=None):
    observation = env.reset(); done = False; total = 0.0; steps = 0
    while not done:
        action = (0.0,) if policy_name == "zero" else (rng.uniform(-1.0, 1.0),)
        observation, reward, done, info = env.step(action)
        total += reward; steps += 1
    return {"policy": policy_name, "steps": steps, "reward": total,
            **{key: info.get(key) for key in (
                "success", "failure_reason", "cube_rise", "maximum_cube_rise",
                "target_reached", "hold_duration", "unsafe", "non_finite", "bootstrap")}}


def smoke():
    output = OUTPUT / "smoke"; seed_everything(); env = LiftPPOEnvironment()
    cleanup = False; records = []
    try:
        env.initialize(); probe = env.reset() and env.workspace_probe()
        records.append(run_policy_episode(env, "zero"))
        records.append(run_policy_episode(env, "random", random.Random(17)))
        ppo = ppo_config(64)
        trainer = ExperimentalLiftPPOTrainer(env, config=ppo, device="cuda:0")
        metrics = trainer.train_one_update()
        latest = trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, "smoke", "model_latest.pt")
        reloader = ExperimentalLiftPPOTrainer(env, config=ppo, device="cuda:0")
        reloader.load_checkpoint(latest)
        reload_ok = reloader.training_steps == trainer.training_steps
        cleanup = True
    finally:
        env.close()
    result = {
        "passed": cleanup and reload_ok and probe["passed"]
                  and all(not x["unsafe"] and not x["non_finite"] for x in records),
        "policies": records, "workspace_probe": probe, "ppo_update": metrics,
        "checkpoint": checkpoint_metadata(latest), "checkpoint_reload": reload_ok,
        "resource_cleanup": cleanup, "contract": contract_payload(),
    }
    write_json(output / "summary.json", result)
    write_json(OUTPUT / "workspace_probe.json", probe)
    return result


def train(run_name, transitions, resume=None):
    if transitions <= 0 or transitions % 200:
        raise ValueError("training transitions must be a positive multiple of 200")
    output = OUTPUT / run_name; output.mkdir(parents=True, exist_ok=True)
    seed_everything(); env_config = LiftEnvConfig(); ppo = ppo_config()
    write_json(output / "config.json", config_payload(env_config, ppo))
    env = LiftPPOEnvironment(env_config); records = []; cleanup = False
    start = time.time(); best_rank = None; reload_ok = False
    try:
        env.initialize(); trainer = ExperimentalLiftPPOTrainer(env, config=ppo, device="cuda:0")
        if resume: trainer.load_checkpoint(Path(resume))
        updates = transitions // ppo.rollout_steps
        for local in range(1, updates + 1):
            metrics = trainer.train_one_update()
            row = {"update_index": trainer.update_steps,
                   "training_steps": trainer.training_steps, **metrics}
            append_jsonl(output / "metrics.jsonl", row)
            for episode in trainer._rollout_episode_records:
                append_jsonl(output / "episodes.jsonl", episode)
            records.append(row)
            rank = (row.get("success_rate", 0.0), row.get("target_reached_rate", 0.0),
                    row["average_reward"])
            if best_rank is None or rank > best_rank:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_best.pt")
                best_rank = rank
            if local % 5 == 0 or local == updates:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_latest.pt")
                print("update=%d steps=%d reward=%.5f success=%.3f target=%.3f hold=%.3f" % (
                    trainer.update_steps, trainer.training_steps, row["average_reward"],
                    row.get("success_rate", 0.0), row.get("target_reached_rate", 0.0),
                    row.get("hold_1s_rate", 0.0)), flush=True)
        latest = output / "checkpoints" / "model_latest.pt"
        reloader = ExperimentalLiftPPOTrainer(env, config=ppo, device="cuda:0")
        payload = reloader.load_checkpoint(latest)
        reload_ok = reloader.training_steps == trainer.training_steps
        cleanup = True
    finally:
        env.close()
    result = {
        "passed": bool(records and reload_ok and cleanup), "run": run_name,
        "transitions_this_run": transitions,
        "training_steps_checkpoint": int(payload["training_steps"]),
        "elapsed_seconds": time.time() - start, "checkpoint_reload": reload_ok,
        "resource_cleanup": cleanup, "last_metrics": records[-1],
        "best_checkpoint": checkpoint_metadata(output / "checkpoints" / "model_best.pt"),
        "latest_checkpoint": checkpoint_metadata(output / "checkpoints" / "model_latest.pt"),
    }
    write_json(output / "training_summary.json", result)
    return result


def load_trainer(env, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = ExperimentalLiftPPOTrainer(
        env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
    trainer.load_checkpoint(Path(checkpoint)); return trainer


def evaluate(run_name, checkpoint, episodes):
    output = OUTPUT / run_name / "evaluation"; env = LiftPPOEnvironment(); cleanup = False
    try:
        env.initialize(); trainer = load_trainer(env, checkpoint)
        summary, records = deterministic_evaluate(env, trainer, episodes)
        summary.update({"checkpoint": checkpoint_metadata(checkpoint),
                        "contract": contract_payload()})
        cleanup = True
    finally:
        env.close()
    summary["resource_cleanup"] = cleanup
    write_json(output / "summary.json", summary)
    for record in records: append_jsonl(output / "episodes.jsonl", record)
    return summary


class LabeledRecorder:
    def __init__(self, recorder, env_getter):
        self.recorder, self.env_getter = recorder, env_getter
    def write(self, frame):
        pixels = frame.data.copy()
        phase = self.env_getter().phase
        label = "GRASP BOOTSTRAP" if phase == "GRASP_BOOTSTRAP" else "LIFT POLICY START / PPO"
        color = (255, 220, 30, 255) if phase == "GRASP_BOOTSTRAP" else (50, 255, 70, 255)
        cv2.rectangle(pixels, (12, 12), (430, 54), (0, 0, 0, 255), -1)
        cv2.putText(pixels, label, (22, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, color, 2, cv2.LINE_AA)
        self.recorder.write(replace(frame, data=pixels))


def video(run_name, checkpoint, filename):
    path = OUTPUT / run_name / "video" / filename
    recorder = ExperimentalMp4Recorder(path).start(); holder = {}; env = None; cleanup = False
    labeled = LabeledRecorder(recorder, lambda: holder["env"])
    try:
        env = LiftPPOEnvironment(with_camera=True, frame_callback=labeled.write)
        holder["env"] = env; env.initialize(); trainer = load_trainer(env, checkpoint)
        summary, records = deterministic_evaluate(env, trainer, 1)
        cleanup = recorder.close(); recorder = None
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"passed": summary["success_count"] == 1 and cleanup,
              "evaluation": summary, "episode": records[0],
              "video": str(path), "resource_cleanup": cleanup,
              "labels": ["GRASP BOOTSTRAP", "LIFT POLICY START / PPO"],
              "checkpoint": checkpoint_metadata(checkpoint)}
    write_json(path.with_suffix(".json"), result); return result


def finalize(run_name):
    evaluation = json.loads((OUTPUT / run_name / "evaluation" / "summary.json").read_text())
    catalog = task_catalog_document()
    gate = (evaluation["success_rate"] >= .90 and evaluation["unsafe_count"] == 0
            and evaluation["non_finite_count"] == 0)
    isolation = {
        "catalog_version": catalog["catalog_version"],
        "catalog_digest": catalog["catalog_digest"],
        "expected_digest": EXPECTED_CATALOG_DIGEST,
        "production_task_count": len(catalog["tasks"]),
        "experimental_registered": [x["id"] for x in catalog["tasks"]
                                    if x["id"] in ("s1_grasp", "s1_lift", "s1_place")],
    }
    isolation["passed"] = (isolation["catalog_version"] == "task-catalog/v1"
                           and isolation["catalog_digest"] == EXPECTED_CATALOG_DIGEST
                           and isolation["production_task_count"] == 7
                           and not isolation["experimental_registered"])
    write_json(OUTPUT / "production_isolation.json", isolation)
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    result = {"stage": "P3-1", "gate": "PASS" if gate and isolation["passed"] else "FAIL",
              "evaluation": evaluation, "isolation": isolation,
              "contract": contract_payload(), "files": files}
    write_json(OUTPUT / "manifest.json", result); return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train", "eval", "video", "finalize"), required=True)
    parser.add_argument("--run", default="train_10k")
    parser.add_argument("--transitions", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--filename", default="successful_lift.mp4")
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    if args.mode == "smoke": result = smoke()
    elif args.mode == "train": result = train(args.run, args.transitions, args.resume)
    elif args.mode == "eval":
        if not args.checkpoint: parser.error("--checkpoint is required")
        result = evaluate(args.run, args.checkpoint, args.episodes)
    elif args.mode == "video":
        if not args.checkpoint: parser.error("--checkpoint is required")
        result = video(args.run, args.checkpoint, args.filename)
    else: result = finalize(args.run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
