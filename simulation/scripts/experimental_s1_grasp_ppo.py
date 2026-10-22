#!/usr/bin/env python3
"""Direct experimental-only P2 Grasp PPO gates; never touches task registry."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

# Isaac Gym Preview 4 requires its extension to be imported before torch in
# real GPU runs.  CPU-only contract tests do not place Isaac Gym on PYTHONPATH.
try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import torch

from simulation_core.experimental.manipulation.grasp_env import GraspEnvConfig, GraspPPOEnvironment
from simulation_core.experimental.manipulation.grasp_ppo import (
    ExperimentalGraspPPOTrainer, deterministic_evaluate,
)
from simulation_core.experimental.manipulation.grasp_schema import (
    GRASP_ACTION_SCHEMA_ID, GRASP_OBSERVATION_SCHEMA_ID, GRASP_POLICY_SCHEMA,
    GRASP_REWARD_VERSION, GRASP_SUCCESS_CRITERIA_VERSION, compute_grasp_reward,
)
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.trainer import PPOConfig


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = ROOT / "artifacts" / "experimental"
PROJECT_ID = "p2_s1_grasp"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def seed_everything(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_dir(name):
    return ARTIFACTS_ROOT / PROJECT_ID / name


def ppo_config():
    return PPOConfig(
        rollout_steps=200, update_epochs=1, batch_size=16,
        learning_rate=3.0e-4, gamma=0.99, gae_lambda=0.95,
        clip_ratio=0.2, value_coefficient=0.5,
        entropy_coefficient=0.01, max_grad_norm=0.5,
    )


def config_payload(env_config, ppo):
    return {
        "task": "s1_grasp", "experimental": True,
        "observation_schema": GRASP_OBSERVATION_SCHEMA_ID,
        "observation_dim": GRASP_POLICY_SCHEMA.observation_dim,
        "action_schema": GRASP_ACTION_SCHEMA_ID, "action_dim": 4,
        "reward_version": GRASP_REWARD_VERSION,
        "success_criteria_version": GRASP_SUCCESS_CRITERIA_VERSION,
        "environment": asdict(env_config), "ppo": asdict(ppo),
    }


def random_gate(output, episodes=3, video=True):
    recorder = None
    env = None
    records = []
    try:
        if video:
            recorder = ExperimentalMp4Recorder(output / "video" / "early_failure.mp4").start()
        env = GraspPPOEnvironment(GraspEnvConfig(), with_camera=video,
                                  frame_callback=recorder.write if recorder else None).initialize()
        rng = random.Random(17)
        for episode in range(episodes):
            observation = env.reset(); done = False; steps = 0; reward_sum = 0.0
            while not done:
                # First episode is a reproducible zero-policy failure video;
                # later episodes exercise the complete normalized action range.
                action = (0.0, 0.0, 0.0, 1.0) if episode == 0 else tuple(
                    rng.uniform(-1.0, 1.0) for _ in range(4))
                observation, reward, done, info = env.step(action)
                reward_sum += reward; steps += 1
                flat = [v for name in GRASP_POLICY_SCHEMA.observation_fields for v in observation[name]]
                if len(flat) != 40 or not all(math.isfinite(v) for v in flat):
                    raise RuntimeError("random gate observation invalid")
                if not math.isfinite(reward): raise RuntimeError("random gate reward invalid")
            records.append({"episode": episode, "steps": steps, "reward": reward_sum, **info})
        passed = all(not item["non_finite"] for item in records)
    finally:
        cleanup = True
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    summary = {"passed": passed and cleanup, "episodes": records,
               "resource_cleanup": cleanup, "video": str(output / "video" / "early_failure.mp4")}
    write_json(output / "random_policy_summary.json", summary)
    return summary


def train(run_name, updates, randomize, resume=None):
    output = run_dir(run_name); output.mkdir(parents=True, exist_ok=True)
    seed_everything(20261002)
    env_config = GraspEnvConfig(randomize_xy=randomize)
    ppo = ppo_config()
    write_json(output / "config" / "config.json", config_payload(env_config, ppo))
    env = GraspPPOEnvironment(env_config)
    start = time.time(); cleanup = False
    records = []; best_rank = None
    try:
        env.initialize()
        trainer = ExperimentalGraspPPOTrainer(env, config=ppo, device="cuda:0")
        if resume:
            trainer.load_curriculum_checkpoint(Path(resume))
        for local in range(1, updates + 1):
            metrics = trainer.train_one_update()
            record = {"update_index": trainer.update_steps,
                      "training_steps": trainer.training_steps, **metrics}
            append_jsonl(output / "metrics" / "training.jsonl", record)
            for episode in trainer._rollout_episode_records:
                append_jsonl(output / "metrics" / "episodes.jsonl", episode)
            records.append(record)
            rank = (record.get("verified_success_rate", 0.0), record["average_reward"])
            if best_rank is None or rank > best_rank:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_best.pt")
                best_rank = rank
            if local % 10 == 0 or local == updates:
                trainer.save_checkpoint(ARTIFACTS_ROOT, PROJECT_ID, run_name, "model_latest.pt")
                print("update=%d steps=%d reward=%.5f success=%.3f candidate=%.3f" % (
                    trainer.update_steps, trainer.training_steps, record["average_reward"],
                    record.get("verified_success_rate", 0.0), record.get("candidate_rate", 0.0)),
                    flush=True)
        # Prove exact reload into a fresh trainer before releasing the scene.
        latest = output / "checkpoints" / "model_latest.pt"
        reloader = ExperimentalGraspPPOTrainer(env, config=ppo, device="cuda:0")
        payload = reloader.load_checkpoint(latest)
        reload_ok = reloader.training_steps == trainer.training_steps
        cleanup = True
    finally:
        env.close()
    summary = {
        "passed": bool(records and reload_ok and cleanup), "run": run_name,
        "updates": updates, "transitions_this_run": updates * ppo.rollout_steps,
        "training_steps_checkpoint": int(payload["training_steps"]),
        "elapsed_seconds": time.time() - start, "checkpoint_reload": reload_ok,
        "resource_cleanup": cleanup, "last_metrics": records[-1] if records else {},
    }
    write_json(output / "training_summary.json", summary)
    return summary


def evaluate(run_name, checkpoint, randomize, episodes):
    output = run_dir(run_name); output.mkdir(parents=True, exist_ok=True)
    env_config = GraspEnvConfig(randomize_xy=randomize)
    env = GraspPPOEnvironment(env_config)
    cleanup = False
    try:
        env.initialize(); payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = PPOConfig(**payload["ppo_config"])
        trainer = ExperimentalGraspPPOTrainer(env, config=config, device="cuda:0")
        trainer.load_checkpoint(Path(checkpoint))
        summary, records = deterministic_evaluate(env, trainer, episodes)
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        summary["checkpoint"] = str(checkpoint); summary["checkpoint_sha256"] = digest
        cleanup = True
    finally:
        env.close()
    summary["resource_cleanup"] = cleanup
    write_json(output / "evaluation" / "summary.json", summary)
    for record in records: append_jsonl(output / "evaluation" / "episodes.jsonl", record)
    return summary


def video(run_name, checkpoint, randomize, filename):
    output = run_dir(run_name) / "video" / filename
    recorder = ExperimentalMp4Recorder(output).start(); env = None; cleanup = False
    try:
        env = GraspPPOEnvironment(GraspEnvConfig(randomize_xy=randomize), with_camera=True,
                                  frame_callback=recorder.write).initialize()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trainer = ExperimentalGraspPPOTrainer(env, config=PPOConfig(**payload["ppo_config"]),
                                               device="cuda:0")
        trainer.load_checkpoint(Path(checkpoint))
        summary, records = deterministic_evaluate(env, trainer, 1)
        cleanup = recorder.close(); recorder = None
    finally:
        if recorder is not None: cleanup = recorder.close()
        if env is not None: env.close()
    result = {"passed": summary["success_count"] == 1 and cleanup,
              "evaluation": summary, "episode": records[0],
              "resource_cleanup": cleanup, "video": str(output)}
    write_json(output.with_suffix(".json"), result)
    return result


def exploit_audit(output):
    cfg = GraspEnvConfig().reward
    def reward(**kw):
        defaults = dict(approach_progress=0.0, alignment_progress=0.0,
            descend_progress=0.0, xy_aligned=False,
            closure_potential_progress=0.0, blockage_potential_progress=0.0,
            candidate_event=False, verified_event=False,
            action=(0.0,0.0,0.0,0.0), saturated=0.0, clamped=False, failure=False)
        defaults.update(kw); return compute_grasp_reward(config=cfg, **defaults)[0]
    cases = {
        "far_close_no_reward": reward(closure_potential_progress=0.0,
                                       action=(0,0,0,-1)),
        "push_without_grasp_penalized": reward(failure=True),
        "oscillation_progress_cancels": reward(approach_progress=.01) + reward(approach_progress=-.01),
        "repeat_close_no_credit": reward(closure_potential_progress=0.0),
        "verifier_bonus_not_available_without_event": reward(candidate_event=False, verified_event=False),
        "table_clamp_penalized": reward(clamped=True),
    }
    passed = (cases["far_close_no_reward"] <= 0 and cases["push_without_grasp_penalized"] < 0
              and abs(cases["oscillation_progress_cancels"]) < 1e-12
              and cases["repeat_close_no_credit"] == 0
              and cases["verifier_bonus_not_available_without_event"] == 0
              and cases["table_clamp_penalized"] < 0)
    result = {"passed": passed, "cases": cases}
    write_json(output / "reward_exploit_audit.json", result); return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("random", "smoke", "train", "eval", "video", "exploit"), required=True)
    parser.add_argument("--run", default="fixed")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--randomize", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--filename", default="policy.mp4")
    args = parser.parse_args()
    os.environ.pop("DISPLAY", None)
    if args.mode == "random": result = random_gate(run_dir(args.run))
    elif args.mode == "smoke": result = train(args.run, 10, args.randomize, args.resume)
    elif args.mode == "train": result = train(args.run, args.updates, args.randomize, args.resume)
    elif args.mode == "eval":
        if not args.checkpoint: parser.error("--checkpoint is required")
        result = evaluate(args.run, args.checkpoint, args.randomize, args.episodes)
    elif args.mode == "video":
        if not args.checkpoint: parser.error("--checkpoint is required")
        result = video(args.run, args.checkpoint, args.randomize, args.filename)
    else: result = exploit_audit(run_dir(args.run))
    print("P2_%s=%s" % (args.mode.upper(), json.dumps(result, sort_keys=True)), flush=True)
    if not result.get("passed", True): raise SystemExit(2)


if __name__ == "__main__": main()
