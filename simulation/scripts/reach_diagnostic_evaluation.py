#!/usr/bin/env python3
"""Fresh-process deterministic evaluation and Camera replay for S1 Reach."""

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
from simulation_core.ppo.normalizer import ObservationNormalizer
from simulation_core.ppo.policy import MLPActorCritic, REACH_POLICY_SCHEMA, flatten_observation
from simulation_core.ppo.trainer import (
    PPOTaskContract, validate_checkpoint_action_schema,
    validate_checkpoint_observation_schema,
)
from simulation_core.reach_adapter import (
    IsaacGymReachBackend, ReachAdapter, ReachConfig,
    REACH_REWARD_SCHEMA_VERSION, REACH_REWARD_SEMANTICS,
    REACH_REWARD_TERM_NAMES, reward_schema_metadata,
)
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig


CONTRACT = PPOTaskContract(
    policy_schema=REACH_POLICY_SCHEMA,
    reward_schema_version=REACH_REWARD_SCHEMA_VERSION,
    reward_semantics=REACH_REWARD_SEMANTICS,
    reward_term_names=REACH_REWARD_TERM_NAMES,
    reward_metadata_factory=reward_schema_metadata,
)


def average(records, field):
    return sum(float(row[field]) for row in records) / len(records)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(checkpoint):
    payload = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
    validate_checkpoint_action_schema(payload, CONTRACT)
    validate_checkpoint_observation_schema(payload, CONTRACT)
    schema = CONTRACT.policy_schema
    policy = MLPActorCritic(
        schema.observation_dim, schema.action_dim,
        action_low=schema.action_low, action_high=schema.action_high,
    ).to("cuda:0")
    policy.load_state_dict(payload["policy"]); policy.eval()
    normalizer = ObservationNormalizer(dimension=18, frozen=True)
    normalizer.load_state_dict(payload["normalizer_state"], frozen=True)
    return policy, normalizer, payload


def reach_config(payload, seed):
    values = dict(payload.get("environment_config", {}))
    allowed = set(ReachConfig.__dataclass_fields__)
    values = {key: value for key, value in values.items() if key in allowed}
    values["seed"] = seed
    values["max_steps"] = 128
    values["physics_steps_per_action"] = 8
    return ReachConfig(**values)


def evaluate_episode(env, backend, policy, normalizer, config, seed, writer=None):
    seed_all(seed)
    observation = env.reset(); episode_return = 0.; length = 0
    minimum_distance = None; action_norm = 0.; saturation = 0.; info = {}; done = False
    with torch.no_grad():
        while not done and length < config.max_steps:
            raw = flatten_observation(observation, REACH_POLICY_SCHEMA).to("cuda:0")
            action, _, _ = policy.act(normalizer.normalize(raw, update=False), deterministic=True)
            observation, reward, done, info = env.step(tuple(float(v) for v in action.detach().cpu()))
            length += 1
            if writer is not None: writer.write(backend.capture_camera_frame())
            episode_return += float(reward)
            distance = float(info["distance_to_goal"])
            minimum_distance = distance if minimum_distance is None else min(minimum_distance, distance)
            action_norm += float(info.get("action_norm", 0.))
            saturation += float(info.get("saturation_rate", 0.))
    state = backend.read_robot_state(); ee = tuple(float(v) for v in state.ee_link_state)
    return {
        "seed": seed, "success": bool(info.get("success", False)),
        "timeout": bool(info.get("timeout", False)) or not done,
        "failure": bool(info.get("failure", False)),
        "failure_reason": info.get("failure_reason"),
        "episode_return": episode_return, "episode_length": length,
        "initial_distance": float(info.get("initial_distance_to_goal", float("inf"))),
        "final_distance": float(info.get("distance_to_goal", float("inf"))),
        "minimum_distance": float(minimum_distance),
        "object_displacement": 0.0,
        "action_norm": action_norm/max(1,length),
        "action_saturation_rate": saturation/max(1,length),
        "time_to_success": info.get("time_to_success"),
        "target_xy": list(env.target_xy), "final_ee_xy": [ee[0], ee[1]],
        "success_distance_m": config.success_distance_m,
    }


def make_env(payload, seed, camera=False):
    config = reach_config(payload, seed)
    backend = IsaacGymReachBackend(
        config,
        s1_config=S1IsaacGymConfig(graphics_device_id=0, headless=True) if camera else None,
        camera_config=IsaacGymCameraConfig(width=640, height=480) if camera else None,
    )
    return config, backend, ReachAdapter(config, backend=backend)


def run_headless(run_root, episodes):
    checkpoint = run_root / "checkpoints/model_latest.pt"
    policy, normalizer, payload = load_policy(checkpoint)
    records = []
    for index in range(episodes):
        seed = 10000 + index
        config, backend, env = make_env(payload, seed)
        try:
            env.initialize(); records.append(evaluate_episode(env, backend, policy, normalizer, config, seed))
        finally: env.close()
        print("EVALUATION_EPISODE=%d/%d" % (index+1, episodes), flush=True)
    output = run_root / "evaluation"; output.mkdir(exist_ok=True)
    with (output/"episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in records: stream.write(json.dumps(row, sort_keys=True)+"\n")
    successes = sum(row["success"] for row in records)
    lower, upper = wilson_confidence_interval(successes, len(records))
    successful = [row["seed"] for row in records if row["success"]]
    target_x = [row["target_xy"][0] for row in records]; target_y = [row["target_xy"][1] for row in records]
    summary = {
        "task_id": "s1_reach", "episode_count": len(records), "success_count": successes,
        "success_rate": successes/len(records),
        "confidence_interval_95": {"method":"wilson","lower":lower,"upper":upper},
        "timeout_rate": average(records,"timeout"), "failure_rate": average(records,"failure"),
        "mean_episode_return": average(records,"episode_return"),
        "mean_episode_length": average(records,"episode_length"),
        "mean_final_distance": average(records,"final_distance"),
        "mean_minimum_distance": average(records,"minimum_distance"),
        "mean_object_displacement": 0.0, "mean_action_norm": average(records,"action_norm"),
        "action_saturation_rate": average(records,"action_saturation_rate"),
        "target_sampling": {"base_seed":10000, "x_min":min(target_x),"x_max":max(target_x),
                            "y_min":min(target_y),"y_max":max(target_y)},
        "checkpoint_sha256": file_sha256(checkpoint),
        "selected_success_seed": min(successful) if successful else None,
        "evaluation_video":{"status":"unavailable","failure_reason":"replay_not_attempted"},
    }
    if not all(math.isfinite(float(v)) for row in records for v in row.values()
               if isinstance(v,(int,float)) and not isinstance(v,bool)):
        raise RuntimeError("non-finite evaluation metric")
    (output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def run_replay(run_root, seed, output_path, fps):
    checkpoint = run_root / "checkpoints/model_latest.pt"
    policy, normalizer, payload = load_policy(checkpoint)
    config, backend, env = make_env(payload, seed, camera=True)
    partial = output_path.with_name(output_path.name+".partial.mp4"); writer = None
    try:
        env.initialize(); writer = RawVideoWriter(partial,640,480,fps)
        record = evaluate_episode(env,backend,policy,normalizer,config,seed,writer)
        writer.close(); writer=None
        if not record["success"]: raise RuntimeError("selected seed did not replay successfully")
        output_path.parent.mkdir(parents=True,exist_ok=True); os.replace(str(partial),str(output_path))
        print("EVALUATION_REPLAY="+json.dumps(record,sort_keys=True),flush=True)
    finally:
        if writer is not None: writer.close()
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True); parser.add_argument("--episodes",type=int,default=100)
    parser.add_argument("--replay-seed",type=int); parser.add_argument("--video-output",type=Path)
    parser.add_argument("--video-fps",type=int,default=12); args=parser.parse_args()
    if args.replay_seed is None:
        if args.episodes != 100: raise ValueError("production evaluation requires 100 episodes")
        run_headless(args.run_root,args.episodes)
    else:
        if args.video_output is None: raise ValueError("replay requires --video-output")
        run_replay(args.run_root,args.replay_seed,args.video_output,args.video_fps)


if __name__ == "__main__": main()
