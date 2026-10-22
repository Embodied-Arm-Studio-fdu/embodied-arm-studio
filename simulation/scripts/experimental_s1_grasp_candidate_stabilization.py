#!/usr/bin/env python3
"""P2-R8 candidate-window position-hold evaluation (experimental only)."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

try:
    from isaacgym import gymapi as _gymapi_import_order  # noqa: F401
except ModuleNotFoundError:
    _gymapi_import_order = None

import torch

from simulation_core.evaluation import wilson_confidence_interval
from simulation_core.experimental.manipulation.grasp_env import (
    CURRICULUM_STAGE_C, GraspEnvConfig, GraspPPOEnvironment,
)
from simulation_core.experimental.manipulation.grasp_ppo import ExperimentalGraspPPOTrainer
from simulation_core.experimental.manipulation.grasp_schema import GRASP_POLICY_SCHEMA
from simulation_core.experimental.manipulation.scripted_runtime import ExperimentalMp4Recorder
from simulation_core.ppo.policy import flatten_observation
from simulation_core.ppo.trainer import PPOConfig
from simulation_core.task_catalog import task_catalog_document
from simulation_core.task_registry import PRODUCTION_TASK_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "experimental" / "p2_r8_s1_grasp"
CHECKPOINT = (ROOT / "artifacts" / "experimental" / "p2_r6_s1_grasp" /
              "random_xy" / "checkpoints" / "model_20000.pt")
EXPECTED_DIGEST = "sha256:c0f3feb0771f4ab636c61fa53ba0d49d5f2a489a224b6890ae38ed798423f791"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_env(with_camera=False, frame_callback=None):
    return GraspPPOEnvironment(
        GraspEnvConfig(curriculum_stage=CURRICULUM_STAGE_C, randomize_xy=.006),
        with_camera=with_camera, frame_callback=frame_callback,
    )


def load_trainer(env):
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    trainer = ExperimentalGraspPPOTrainer(
        env, config=PPOConfig(**payload["ppo_config"]), device="cuda:0")
    trainer.load_checkpoint(CHECKPOINT)
    trainer.policy.eval()
    return trainer


def policy_step(observation, trainer):
    raw = flatten_observation(observation, GRASP_POLICY_SCHEMA).to(trainer.device)
    normalized = trainer.normalizer.normalize(raw, update=False)
    with torch.no_grad():
        action, _, _ = trainer.policy.act(normalized, deterministic=True)
    return tuple(float(value) for value in action.detach().cpu())


def run_episode(env, trainer, episode):
    observation = env.reset()
    cube_xy = tuple(float(value) for value in env._initial_cube[:2])
    done = False
    total_reward = 0.0
    steps = 0
    info = {}
    stabilization_trace = []
    while not done:
        action = policy_step(observation, trainer)
        observation, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if (info.get("stabilizing_active") or info.get("stabilizing_entered")
                or info.get("stabilizing_completed")
                or info.get("stabilizing_evidence_lost") or done):
            stabilization_trace.append({
                "control_step": steps,
                "active": bool(info.get("stabilizing_active")),
                "entered": bool(info.get("stabilizing_entered")),
                "completed": bool(info.get("stabilizing_completed")),
                "evidence_lost": bool(info.get("stabilizing_evidence_lost")),
                "stabilization_steps": int(info.get("stabilization_steps", 0)),
                "candidate_window_progress": int(info.get("candidate_window_progress", 0)),
                "hold_target": info.get("stabilization_hold_target"),
                "z_target_margin_m": float(info.get("stabilization_z_target_margin", 0.0)),
                "jaw_command": float(info.get("jaw_command", 0.0)),
                "jaw_actual": tuple(float(v) for v in info.get("jaw_actual", ())),
                "blockage_gap_m": float(info.get("blockage_gap", 0.0)),
                "candidate_flags": dict(info.get("candidate_evidence_flags", {})),
                "failure_reason": str(info.get("failure_reason", "")),
                "success": bool(info.get("success", False)),
            })
        if steps > env.config.learning_horizon + 1:
            raise RuntimeError("evaluation episode exceeded environment horizon")
    return {
        "episode": episode,
        "randomized_cube_xy": cube_xy,
        "success": bool(info["success"]),
        "candidate": bool(info.get("candidate_once", False)),
        "verified": bool(info.get("verified_success", False)),
        "failure_reason": str(info.get("failure_reason", "")),
        "reward": total_reward,
        "learning_steps": int(info.get("learning_steps", steps)),
        "stabilizing_entered": bool(info.get("stabilizing_entered_once", False)),
        "stabilizing_completed": bool(info.get("stabilizing_completed_once", False)),
        "evidence_lost_count": int(info.get("stabilizing_evidence_lost_count", 0)),
        "stabilization_steps": int(info.get("stabilization_steps", 0)),
        "z_target_margin_m": float(info.get("stabilization_z_target_margin", 0.0)),
        "workspace_violation": info.get("failure_reason") == "WORKSPACE_VIOLATION",
        "unsafe": bool(info.get("unsafe", False)),
        "non_finite": bool(info.get("non_finite", False)),
        "clearance_reached": bool(info.get("clearance_reached", False)),
        "aligned": bool(info.get("aligned", False)),
        "post_clearance_descended": bool(info.get("post_clearance_descended", False)),
        "closure_started": bool(info.get("closure_started", False)),
        "jaw_below_030": bool(info.get("jaw_below_030", False)),
        "blockage_ge_0020": bool(info.get("blockage_ge_0020", False)),
        "jaw_command": float(info.get("jaw_command", 0.0)),
        "jaw_actual": tuple(float(v) for v in info.get("jaw_actual", ())),
        "maximum_blockage_gap_m": float(info.get("maximum_blockage_gap", 0.0)),
        "stabilization_trace": stabilization_trace,
    }


def summarize(records):
    episodes = len(records)
    successes = sum(item["success"] for item in records)
    lower, upper = wilson_confidence_interval(successes, episodes)
    entered = sum(item["stabilizing_entered"] for item in records)
    completed = sum(item["stabilizing_completed"] for item in records)
    reasons = {}
    for item in records:
        reason = item["failure_reason"] or "SUCCESS"
        reasons[reason] = reasons.get(reason, 0) + 1
    margins = [item["z_target_margin_m"] for item in records if item["stabilizing_entered"]]
    return {
        "deterministic": True,
        "episodes": episodes,
        "success_count": successes,
        "success_rate": successes / episodes,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "candidate_count": sum(item["candidate"] for item in records),
        "verified_count": sum(item["verified"] for item in records),
        "stabilizing_entered_count": entered,
        "stabilizing_completed_count": completed,
        "stabilizing_completion_rate": completed / entered if entered else 0.0,
        "evidence_lost_episode_count": sum(item["evidence_lost_count"] > 0 for item in records),
        "evidence_lost_event_count": sum(item["evidence_lost_count"] for item in records),
        "mean_stabilization_steps": mean(
            item["stabilization_steps"] for item in records if item["stabilizing_entered"]
        ) if entered else 0.0,
        "workspace_violation_count": reasons.get("WORKSPACE_VIOLATION", 0),
        "timeout_count": reasons.get("TIMEOUT", 0),
        "verification_fail_count": reasons.get("GRASP_VERIFICATION_FAILED", 0),
        "unsafe_count": sum(item["unsafe"] for item in records),
        "non_finite_count": sum(item["non_finite"] for item in records),
        "failure_reason_histogram": reasons,
        "mean_z_target_margin_m": mean(margins) if margins else 0.0,
        "minimum_z_target_margin_m": min(margins, default=0.0),
        "mean_maximum_blockage_gap_m": mean(
            item["maximum_blockage_gap_m"] for item in records),
        "behavior_funnel": {
            "episodes": episodes,
            "clearance_reached": sum(item["clearance_reached"] for item in records),
            "aligned": sum(item["aligned"] for item in records),
            "post_clearance_descended": sum(item["post_clearance_descended"] for item in records),
            "closure_started": sum(item["closure_started"] for item in records),
            "jaw_below_030": sum(item["jaw_below_030"] for item in records),
            "blockage_ge_0020": sum(item["blockage_ge_0020"] for item in records),
            "candidate": sum(item["candidate"] for item in records),
            "verified": sum(item["verified"] for item in records),
        },
    }


def evaluate(label, episodes):
    env = make_env()
    records = []
    try:
        env.initialize()
        trainer = load_trainer(env)
        records = [run_episode(env, trainer, episode) for episode in range(episodes)]
    finally:
        env.close()
    summary = summarize(records)
    summary.update({
        "label": label,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "environment": asdict(env.config),
        "training_transitions": 0,
        "resource_cleanup": True,
    })
    directory = OUTPUT / label
    write_json(directory / "episodes.json", records)
    write_json(directory / "summary.json", summary)
    return summary


def record_success(max_attempts=10):
    video_directory = OUTPUT / "video"
    env = make_env(with_camera=True)
    attempts = []
    successful_path = None
    try:
        env.initialize()
        trainer = load_trainer(env)
        for attempt in range(max_attempts):
            path = video_directory / ("attempt_%02d.mp4" % (attempt + 1))
            recorder = ExperimentalMp4Recorder(path).start()
            env.frame_callback = recorder.write
            record = run_episode(env, trainer, attempt)
            env.frame_callback = None
            encoded = recorder.close()
            attempts.append({"attempt": attempt + 1, "encoded": encoded,
                             "path": str(path), "episode": record})
            if record["success"] and encoded:
                successful_path = path
                break
    finally:
        env.close()
    result = {"success_video": str(successful_path) if successful_path else None,
              "attempts": attempts, "checkpoint_sha256": sha256(CHECKPOINT)}
    write_json(video_directory / "success_video.json", result)
    return result


def finalize():
    replay = json.loads((OUTPUT / "replay_20" / "summary.json").read_text())
    formal_path = OUTPUT / "randomized_100" / "summary.json"
    formal = json.loads(formal_path.read_text()) if formal_path.is_file() else None
    gate = bool(formal and formal["success_rate"] >= .80
                and formal["wilson_95_lower"] >= .70
                and formal["unsafe_count"] == 0 and formal["non_finite_count"] == 0)
    catalog = task_catalog_document()
    ids = [item.id for item in PRODUCTION_TASK_REGISTRY.definitions()]
    production = {
        "task_count": len(ids), "catalog_version": catalog["catalog_version"],
        "catalog_digest": catalog["catalog_digest"],
        "experimental_tasks_registered": sorted(
            set(ids).intersection({"s1_grasp", "s1_lift", "s1_place"})),
    }
    production["passed"] = (
        len(ids) == 7 and catalog["catalog_version"] == "task-catalog/v1"
        and catalog["catalog_digest"] == EXPECTED_DIGEST
        and not production["experimental_tasks_registered"])
    write_json(OUTPUT / "production_regression.json", production)
    result = {
        "experiment": "P2-R8", "training_transitions": 0,
        "P2_R8_CANDIDATE_STABILIZATION_GATE": "PASS" if gate else "FAIL",
        "P2_S1_GRASP_PPO_GATE": "PASS" if gate else "FAIL",
        "replay_20": replay, "randomized_100": formal,
        "production_regression": production,
    }
    write_json(OUTPUT / "result.json", result)
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(OUTPUT))] = {
                "bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(OUTPUT / "manifest.json", {**result, "files": files})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=("replay", "evaluate", "video", "finalize"))
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    os.environ.pop("DISPLAY", None)
    if args.mode == "replay":
        result = evaluate("replay_20", args.episodes or 20)
    elif args.mode == "evaluate":
        result = evaluate("randomized_100", args.episodes or 100)
    elif args.mode == "video":
        result = record_success()
    else:
        result = finalize()
    print("P2_R8_%s=%s" % (args.mode.upper(), json.dumps(result, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
