#!/usr/bin/env python3
"""P3-2 direct 5-second hold replay, optional continuation, and evidence."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import cv2
import torch

from simulation_core.experimental.manipulation.lift_env import LiftEnvConfig, LiftPPOEnvironment
from simulation_core.experimental.manipulation.lift_ppo import (
    ExperimentalLiftPPOTrainer, deterministic_evaluate,
)
from simulation_core.experimental.manipulation.lift_schema import (
    LIFT_ACTION_SCHEMA_ID, LIFT_LONG_HOLD_SUCCESS_CRITERIA_VERSION,
    LIFT_OBSERVATION_SCHEMA_ID, LIFT_REWARD_VERSION,
)
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p3_2_s1_lift"
ARTIFACTS_ROOT = ROOT / "artifacts" / "experimental"
PROJECT_ID = "p3_2_s1_lift"
P3_1_CHECKPOINT = (ROOT / "artifacts" / "experimental" / "p3_1_s1_lift" /
                   "train_20k" / "checkpoints" / "model_best.pt")
P3_1_SHA256 = "3a564c8effb7b06cad2f9e624e57732ba209198c70267569d26a40def180727b"
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


def environment_config():
    return replace(LiftEnvConfig(), hold_seconds=5.0, max_episode_seconds=10.0)


def checkpoint_metadata(path):
    path = Path(path); payload = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_id": payload.get("task_id"),
            "observation_schema": payload.get("observation_schema_id"),
            "action_schema": payload.get("action_schema_id"),
            "action_dim": payload.get("action_dim"),
            "reward_version": payload.get("reward_version"),
            "success_version": payload.get("success_criteria_version"),
            "training_transitions_total": payload.get("training_transitions_total")}


def contract():
    config = environment_config()
    return {"task_id": "s1_lift", "experimental": True,
            "observation_schema": LIFT_OBSERVATION_SCHEMA_ID, "observation_dim": 18,
            "action_schema": LIFT_ACTION_SCHEMA_ID, "action_dim": 1,
            "reward_version": LIFT_REWARD_VERSION,
            "success_version": LIFT_LONG_HOLD_SUCCESS_CRITERIA_VERSION,
            "target_lift_m": config.target_lift, "target_band_m": list(config.target_band),
            "hold_seconds": config.hold_seconds,
            "episode_horizon_seconds": config.max_episode_seconds,
            "only_changes_from_p3_1": ["success_hold_seconds", "episode_horizon_seconds"]}


def make_trainer(env, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = ExperimentalLiftPPOTrainer(
        env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
    if payload.get("success_criteria_version") == LIFT_LONG_HOLD_SUCCESS_CRITERIA_VERSION:
        trainer.load_checkpoint(checkpoint)
    else:
        trainer.load_hold_extension_checkpoint(checkpoint, P3_1_SHA256)
    return trainer


def replay(checkpoint=P3_1_CHECKPOINT, episodes=50):
    checkpoint = Path(checkpoint)
    if checkpoint.resolve() != P3_1_CHECKPOINT.resolve():
        raise ValueError("direct replay must use the locked P3-1 best checkpoint")
    env = LiftPPOEnvironment(environment_config()); cleanup = False
    try:
        env.initialize(); trainer = make_trainer(env, checkpoint)
        summary, records = deterministic_evaluate(env, trainer, episodes)
        cleanup = True
    finally:
        env.close()
    summary.update({"resource_cleanup": cleanup, "checkpoint": checkpoint_metadata(checkpoint),
                    "contract": contract(), "training_transitions_added": 0,
                    "replay_first": True,
                    "gate": "PASS" if summary["success_rate"] >= .90
                    and summary["unsafe_count"] == 0
                    and summary["non_finite_count"] == 0 else "FAIL"})
    output = OUTPUT / "replay_50"
    write_json(output / "summary.json", summary)
    for row in records: append_jsonl(output / "episodes.jsonl", row)
    return summary


class LabeledRecorder:
    def __init__(self, recorder, env_getter):
        self.recorder, self.env_getter = recorder, env_getter
    def write(self, frame):
        pixels = frame.data.copy(); phase = self.env_getter().phase
        if phase == "GRASP_BOOTSTRAP":
            label, color = "STABLE GRASP RESET", (255, 220, 30, 255)
        else:
            label, color = "LIFT POLICY / 5 SECOND HOLD", (50, 255, 70, 255)
        cv2.rectangle(pixels, (12, 12), (500, 54), (0, 0, 0, 255), -1)
        cv2.putText(pixels, label, (22, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    .68, color, 2, cv2.LINE_AA)
        self.recorder.write(replace(frame, data=pixels))


def video(checkpoint=P3_1_CHECKPOINT):
    path = OUTPUT / "video" / "successful_5s_hold.mp4"
    recorder = ExperimentalMp4Recorder(path).start(); holder = {}; env = None; cleanup = False
    labeled = LabeledRecorder(recorder, lambda: holder["env"])
    try:
        env = LiftPPOEnvironment(environment_config(), with_camera=True,
                                 frame_callback=labeled.write)
        holder["env"] = env; env.initialize(); trainer = make_trainer(env, Path(checkpoint))
        summary, records = deterministic_evaluate(env, trainer, 1)
        cleanup = recorder.close(); recorder = None
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"passed": summary["success_count"] == 1 and cleanup,
              "evaluation": summary, "episode": records[0], "video": str(path),
              "resource_cleanup": cleanup,
              "labels": ["STABLE GRASP RESET", "LIFT POLICY / 5 SECOND HOLD"],
              "checkpoint": checkpoint_metadata(checkpoint)}
    write_json(path.with_suffix(".json"), result); return result


def train(run_name, transitions, resume=P3_1_CHECKPOINT):
    """Allowed only after a failed replay; keeps all P3-1 task/reward settings."""
    replay_summary = json.loads((OUTPUT / "replay_50" / "summary.json").read_text())
    if replay_summary["gate"] == "PASS":
        raise RuntimeError("P3-2 replay already passed; training is forbidden")
    if transitions not in (10000, 20000):
        raise ValueError("P3-2 continuation must be 10k, or at most 20k, transitions")
    output = OUTPUT / run_name; config = environment_config()
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    ppo = PPOConfig(**payload["ppo_config"]); env = LiftPPOEnvironment(config)
    records = []; start = time.time(); cleanup = False; best = None
    write_json(output / "config.json", {"contract": contract(), "environment": asdict(config),
                                         "ppo": asdict(ppo), "resume": checkpoint_metadata(resume)})
    try:
        env.initialize(); trainer = make_trainer(env, Path(resume))
        for _ in range(transitions // ppo.rollout_steps):
            metrics = trainer.train_one_update()
            row = {"update_index": trainer.update_steps, "training_steps": trainer.training_steps,
                   **metrics}; records.append(row); append_jsonl(output / "metrics.jsonl", row)
            rank = (row.get("success_rate", 0.0), row.get("target_reached_rate", 0.0),
                    row["average_reward"])
            if best is None or rank > best:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_best.pt")
                best = rank
            if len(records) % 5 == 0:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_latest.pt")
        cleanup = True
    finally:
        env.close()
    result = {"transitions_added": transitions, "resource_cleanup": cleanup,
              "elapsed_seconds": time.time() - start, "last_metrics": records[-1],
              "best_checkpoint": checkpoint_metadata(output / "checkpoints" / "model_best.pt")}
    write_json(output / "training_summary.json", result); return result


def finalize():
    replay_summary = json.loads((OUTPUT / "replay_50" / "summary.json").read_text())
    catalog = task_catalog_document()
    isolation = {"catalog_version": catalog["catalog_version"],
                 "catalog_digest": catalog["catalog_digest"],
                 "production_task_count": len(catalog["tasks"]),
                 "experimental_registered": [x["id"] for x in catalog["tasks"]
                                             if x["id"] in ("s1_grasp", "s1_lift", "s1_place")]}
    isolation["passed"] = (isolation["catalog_version"] == "task-catalog/v1"
                           and isolation["catalog_digest"] == CATALOG_DIGEST
                           and isolation["production_task_count"] == 7
                           and not isolation["experimental_registered"])
    write_json(OUTPUT / "production_isolation.json", isolation)
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size}
    gate = replay_summary["gate"] == "PASS" and isolation["passed"]
    result = {"stage": "P3-2", "gate": "PASS" if gate else "FAIL",
              "replay": replay_summary, "training_performed": False,
              "training_transitions_added": 0, "contract": contract(),
              "isolation": isolation, "files": files}
    write_json(OUTPUT / "manifest.json", result); return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "video", "train", "finalize"), required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=P3_1_CHECKPOINT)
    parser.add_argument("--run", default="train_10k")
    parser.add_argument("--transitions", type=int, default=10000)
    parser.add_argument("--resume", type=Path, default=P3_1_CHECKPOINT)
    args = parser.parse_args(); os.environ.pop("DISPLAY", None)
    if args.mode == "replay": result = replay(args.checkpoint, args.episodes)
    elif args.mode == "video": result = video(args.checkpoint)
    elif args.mode == "train": result = train(args.run, args.transitions, args.resume)
    else: result = finalize()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
