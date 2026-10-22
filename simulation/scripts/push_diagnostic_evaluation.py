#!/usr/bin/env python3
"""Evaluate headlessly, or replay one fixed successful evaluation episode."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from isaacgym import gymapi as _gymapi  # noqa: F401
import torch

from push_diagnostic_training import RawVideoWriter, seed_all
from simulation_core.evaluation import wilson_confidence_interval
from simulation_core.isaac_gym_camera import IsaacGymCameraConfig
from simulation_core.ppo.policy import MLPActorCritic, flatten_observation
from simulation_core.ppo.normalizer import ObservationNormalizer
from simulation_core.ppo.trainer import (
    validate_checkpoint_action_schema,
    validate_checkpoint_observation_schema,
)
from simulation_core.push_adapter import IsaacGymPushBackend, PushAdapter, PushConfig
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig


def average(records, field):
    return sum(float(row[field]) for row in records) / len(records)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(checkpoint):
    policy = MLPActorCritic().to("cuda:0")
    payload = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
    validate_checkpoint_action_schema(payload)
    validate_checkpoint_observation_schema(payload)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    normalizer = ObservationNormalizer(frozen=True)
    normalizer.load_state_dict(payload["normalizer_state"], frozen=True)
    return policy, normalizer, payload


def push_config(payload):
    values = dict(payload.get("environment_config", {}))
    allowed = set(PushConfig.__dataclass_fields__)
    return PushConfig(**{name: value for name, value in values.items() if name in allowed})


def evaluate_episode(env, backend, policy, normalizer, config, seed, writer=None):
    seed_all(seed)
    observation = env.reset()
    episode_return = 0.0
    length = 0
    minimum_distance = None
    action_norm = 0.0
    saturation = 0.0
    info = {}
    done = False
    with torch.no_grad():
        while not done and length < config.max_steps:
            raw = flatten_observation(observation).to("cuda:0")
            action, _, _ = policy.act(
                normalizer.normalize(raw, update=False), deterministic=True
            )
            observation, reward, done, info = env.step(
                tuple(float(value) for value in action.detach().cpu())
            )
            length += 1
            if writer is not None:
                writer.write(backend.capture_camera_frame())
            episode_return += float(reward)
            distance = float(info["distance_to_goal"])
            minimum_distance = (
                distance if minimum_distance is None else min(minimum_distance, distance)
            )
            action_norm += float(info.get("action_norm", 0.0))
            saturation += float(info.get("saturation_rate", 0.0))
    return {
        "seed": seed,
        "success": bool(info.get("success", False)),
        "timeout": bool(info.get("timeout", False)) or not done,
        "failure": bool(info.get("failure", False)),
        "episode_return": episode_return,
        "episode_length": length,
        "final_distance": float(info.get("distance_to_goal", float("inf"))),
        "minimum_distance": float(minimum_distance),
        "object_displacement": float(info.get("object_displacement", 0.0)),
        "action_norm": action_norm / max(1, length),
        "action_saturation_rate": saturation / max(1, length),
    }


def run_headless_evaluation(run_root, episodes):
    checkpoint = run_root / "checkpoints/model_latest.pt"
    policy, normalizer, payload = load_policy(checkpoint)
    config = push_config(payload)
    backend = IsaacGymPushBackend(config)
    env = PushAdapter(config, backend=backend)
    records = []
    try:
        env.initialize()
        for seed in range(episodes):
            records.append(evaluate_episode(
                env, backend, policy, normalizer, config, seed
            ))
            print("EVALUATION_EPISODE=%d/%d" % (seed + 1, episodes), flush=True)
    finally:
        env.close()
    output = run_root / "evaluation"
    output.mkdir(exist_ok=True)
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    successes = sum(row["success"] for row in records)
    successful_seeds = [row["seed"] for row in records if row["success"]]
    lower, upper = wilson_confidence_interval(successes, len(records))
    summary = {
        "episode_count": len(records), "success_count": successes,
        "success_rate": successes / len(records),
        "confidence_interval_95": {"method": "wilson", "lower": lower, "upper": upper},
        "timeout_rate": average(records, "timeout"),
        "failure_rate": average(records, "failure"),
        "mean_episode_return": average(records, "episode_return"),
        "mean_episode_length": average(records, "episode_length"),
        "mean_final_distance": average(records, "final_distance"),
        "mean_minimum_distance": average(records, "minimum_distance"),
        "mean_object_displacement": average(records, "object_displacement"),
        "mean_action_norm": average(records, "action_norm"),
        "action_saturation_rate": average(records, "action_saturation_rate"),
        "mean_action_saturation_rate": average(records, "action_saturation_rate"),
        "checkpoint_sha256": file_sha256(checkpoint),
        "selected_success_seed": min(successful_seeds) if successful_seeds else None,
        "evaluation_video": {
            "status": "unavailable",
            "failure_reason": "replay_not_attempted",
        },
    }
    if not all(
        math.isfinite(float(value))
        for row in records for value in row.values()
        if isinstance(value, (int, float))
    ):
        raise RuntimeError("non-finite evaluation metric")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("EVALUATION_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)


def run_success_replay(run_root, seed, output_path, fps):
    checkpoint = run_root / "checkpoints/model_latest.pt"
    policy, normalizer, payload = load_policy(checkpoint)
    config = push_config(payload)
    backend = IsaacGymPushBackend(
        config,
        s1_config=S1IsaacGymConfig(graphics_device_id=0, headless=True),
        camera_config=IsaacGymCameraConfig(width=640, height=480),
    )
    env = PushAdapter(config, backend=backend)
    partial = output_path.with_name(output_path.name + ".partial.mp4")
    writer = None
    try:
        env.initialize()
        writer = RawVideoWriter(partial, width=640, height=480, fps=fps)
        record = evaluate_episode(
            env, backend, policy, normalizer, config, seed, writer=writer
        )
        writer.close()
        writer = None
        if record["success"] is not True:
            raise RuntimeError("selected evaluation seed did not replay successfully")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(partial), str(output_path))
        print("EVALUATION_REPLAY=" + json.dumps(record, sort_keys=True), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--replay-seed", type=int)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=int, default=12)
    args = parser.parse_args()
    if args.replay_seed is None:
        if args.episodes != 100:
            raise ValueError("production evaluation requires 100 episodes")
        if args.video_output is not None:
            raise ValueError("headless evaluation does not accept --video-output")
        run_headless_evaluation(args.run_root, args.episodes)
        return
    if args.video_output is None:
        raise ValueError("successful replay requires --video-output")
    if not 10 <= args.video_fps <= 15:
        raise ValueError("evaluation replay fps must be between 10 and 15")
    run_success_replay(
        args.run_root, args.replay_seed, args.video_output, args.video_fps
    )


if __name__ == "__main__":
    main()
