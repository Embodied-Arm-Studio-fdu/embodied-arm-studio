#!/usr/bin/env python3
"""P2-R1 reset-only staged curriculum runner (experimental, never registered)."""

import argparse
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

from simulation_core.experimental.manipulation.grasp_env import (
    CURRICULUM_STAGES, GraspEnvConfig, GraspPPOEnvironment,
)
from simulation_core.experimental.manipulation.grasp_ppo import (
    ExperimentalGraspPPOTrainer, deterministic_evaluate,
)
from simulation_core.experimental.manipulation.grasp_schema import (
    GRASP_ACTION_SCHEMA_ID, GRASP_OBSERVATION_SCHEMA_ID, GRASP_POLICY_SCHEMA,
    GRASP_REWARD_VERSION, GRASP_SUCCESS_CRITERIA_VERSION,
)
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.trainer import PPOConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "experimental" / "p2_r1_s1_grasp"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def seed_everything(seed=20261002):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ppo_config():
    return PPOConfig(
        rollout_steps=200, update_epochs=1, batch_size=16,
        learning_rate=3.0e-4, gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, value_coefficient=0.5,
        entropy_coefficient=0.01, max_grad_norm=0.5,
    )


def stage_directory(stage):
    return ARTIFACT_ROOT / ("stage_" + stage.split("_", 1)[0])


def config_payload(stage, env_config, ppo, parent):
    return {
        "task": "s1_grasp", "experiment": "P2-R1", "curriculum_stage": stage,
        "only_changed_variable": "training_reset_profile",
        "parent_checkpoint": str(parent) if parent else None,
        "observation_schema": GRASP_OBSERVATION_SCHEMA_ID,
        "observation_dim": GRASP_POLICY_SCHEMA.observation_dim,
        "action_schema": GRASP_ACTION_SCHEMA_ID,
        "action_dim": GRASP_POLICY_SCHEMA.action_dim,
        "reward_version": GRASP_REWARD_VERSION,
        "success_criteria_version": GRASP_SUCCESS_CRITERIA_VERSION,
        "environment": asdict(env_config), "ppo": asdict(ppo),
    }


def _existing_best_rank(path):
    if not path.is_file():
        return None
    best = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rank = (record.get("verified_success_rate", 0.0),
                record.get("candidate_rate", 0.0), record.get("average_reward", 0.0))
        if best is None or rank > best:
            best = rank
    return best


def train(stage, updates, resume=None):
    output = stage_directory(stage)
    output.mkdir(parents=True, exist_ok=True)
    seed_everything()
    env_config = GraspEnvConfig(curriculum_stage=stage)
    ppo = ppo_config()
    write_json(output / "config.json", config_payload(stage, env_config, ppo, resume))
    env = GraspPPOEnvironment(env_config)
    start = time.time()
    records = []
    reload_ok = False
    payload = {}
    metrics_path = output / "metrics.jsonl"
    best_rank = _existing_best_rank(metrics_path)
    try:
        env.initialize()
        trainer = ExperimentalGraspPPOTrainer(env, config=ppo, device="cuda:0")
        if resume:
            trainer.load_staged_curriculum_checkpoint(Path(resume))
        for local in range(1, updates + 1):
            metrics = trainer.train_one_update()
            record = {
                "curriculum_stage": stage, "update_index": trainer.update_steps,
                "training_steps_total": trainer.training_steps,
                "training_steps_stage": trainer.training_steps - trainer.curriculum_stage_start_steps,
                **metrics,
            }
            append_jsonl(metrics_path, record)
            for episode in trainer._rollout_episode_records:
                append_jsonl(output / "episodes.jsonl", episode)
            records.append(record)
            rank = (record.get("verified_success_rate", 0.0),
                    record.get("candidate_rate", 0.0), record["average_reward"])
            if best_rank is None or rank > best_rank:
                trainer.save_checkpoint(
                    ARTIFACT_ROOT.parent, ARTIFACT_ROOT.name, output.name, "model_best.pt")
                best_rank = rank
            if local % 10 == 0 or local == updates:
                trainer.save_checkpoint(
                    ARTIFACT_ROOT.parent, ARTIFACT_ROOT.name, output.name, "model_latest.pt")
                print("stage=%s update=%d total=%d stage_steps=%d reward=%.5f candidate=%.3f success=%.3f" % (
                    stage, trainer.update_steps, trainer.training_steps,
                    trainer.training_steps - trainer.curriculum_stage_start_steps,
                    record["average_reward"], record.get("candidate_rate", 0.0),
                    record.get("verified_success_rate", 0.0)), flush=True)
        latest = output / "checkpoints" / "model_latest.pt"
        reloader = ExperimentalGraspPPOTrainer(env, config=ppo, device="cuda:0")
        payload = reloader.load_checkpoint(latest)
        reload_ok = reloader.training_steps == trainer.training_steps
    finally:
        env.close()
    summary = {
        "passed": bool(records and reload_ok), "curriculum_stage": stage,
        "updates_this_run": updates, "transitions_this_run": updates * ppo.rollout_steps,
        "training_transitions_total": int(payload.get("training_transitions_total", 0)),
        "training_transitions_stage": int(payload.get("training_transitions_stage", 0)),
        "parent_checkpoint_sha256": payload.get("parent_checkpoint_sha256"),
        "checkpoint_reload": reload_ok, "resource_cleanup": True,
        "elapsed_seconds": time.time() - start,
        "last_metrics": records[-1] if records else {},
    }
    write_json(output / "training_summary.json", summary)
    return summary


def evaluate(stage, checkpoint, episodes):
    output = stage_directory(stage)
    env = GraspPPOEnvironment(GraspEnvConfig(curriculum_stage=stage))
    try:
        env.initialize()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trainer = ExperimentalGraspPPOTrainer(
            env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
        trainer.load_checkpoint(Path(checkpoint))
        summary, records = deterministic_evaluate(env, trainer, episodes)
        summary.update({
            "curriculum_stage": stage, "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "resource_cleanup": True,
        })
    finally:
        env.close()
    write_json(output / "evaluation" / "summary.json", summary)
    records_path = output / "evaluation" / "episodes.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n"
                                    for record in records), encoding="utf-8")
    return summary


def video(stage, checkpoint, filename):
    output = stage_directory(stage) / "video" / filename
    recorder = ExperimentalMp4Recorder(output).start()
    env = None
    cleanup = False
    try:
        env = GraspPPOEnvironment(
            GraspEnvConfig(curriculum_stage=stage), with_camera=True,
            frame_callback=recorder.write).initialize()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trainer = ExperimentalGraspPPOTrainer(
            env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
        trainer.load_checkpoint(Path(checkpoint))
        summary, records = deterministic_evaluate(env, trainer, 1)
        cleanup = recorder.close()
        recorder = None
    finally:
        if recorder is not None:
            cleanup = recorder.close()
        if env is not None:
            env.close()
    result = {
        "curriculum_stage": stage, "evaluation": summary,
        "episode": records[0], "resource_cleanup": cleanup, "video": str(output),
    }
    write_json(output.with_suffix(".json"), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval", "video"), required=True)
    parser.add_argument("--stage", choices=CURRICULUM_STAGES, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--filename", default="policy.mp4")
    args = parser.parse_args()
    os.environ.pop("DISPLAY", None)
    if args.mode == "train":
        result = train(args.stage, args.updates, args.resume)
    elif args.mode == "eval":
        if not args.checkpoint:
            parser.error("--checkpoint is required")
        result = evaluate(args.stage, args.checkpoint, args.episodes)
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required")
        result = video(args.stage, args.checkpoint, args.filename)
    print("P2_R1_%s=%s" % (args.mode.upper(), json.dumps(result, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
