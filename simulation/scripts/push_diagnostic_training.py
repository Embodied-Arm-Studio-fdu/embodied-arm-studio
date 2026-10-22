#!/usr/bin/env python3
"""One-shot real-GPU S1 Push training, deterministic evaluation, and video."""

import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys

# Isaac Gym Preview 4 must be imported before torch.
from isaacgym import gymapi as _gymapi  # noqa: F401
import numpy as np
import torch

from simulation_core.evaluation import wilson_confidence_interval
from simulation_core.isaac_gym_camera import IsaacGymCameraConfig
from simulation_core.ppo import PPOConfig, PPOTrainer, PPOTrainingConfig
from simulation_core.ppo.policy import MLPActorCritic, flatten_observation
from simulation_core.push_adapter import IsaacGymPushBackend, PushAdapter, PushConfig
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-mode", choices=("training",), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--rollout-steps", type=int, required=True)
    result.add_argument("--num-updates", type=int, required=True)
    result.add_argument("--episode-steps", type=int, required=True)
    result.add_argument("--physics-steps-per-action", type=int, required=True)
    result.add_argument("--update-epochs", type=int, required=True)
    result.add_argument("--batch-size", type=int, required=True)
    result.add_argument("--learning-rate", type=float, required=True)
    result.add_argument("--checkpoint-every", type=int, required=True)
    result.add_argument("--evaluation-episodes", type=int, required=True)
    result.add_argument("--evaluation-video", action="store_true")
    result.add_argument("--project-id", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--artifacts-root", type=Path, required=True)
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment(config, video=False):
    camera = IsaacGymCameraConfig() if video else None
    s1 = S1IsaacGymConfig(graphics_device_id=0, headless=True) if video else None
    backend = IsaacGymPushBackend(config, s1_config=s1, camera_config=camera)
    return PushAdapter(config, backend=backend)


class RawVideoWriter:
    def __init__(self, path, width=640, height=480, fps=15):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "/home/user/opt/ffmpeg-nvenc/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgba", "-video_size",
            "%dx%d" % (width, height), "-framerate", str(fps), "-i", "-",
            "-an", "-c:v", "h264_nvenc", "-preset", "p4", "-y", str(self.path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame.data).tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        returncode = self.process.wait(timeout=30)
        if returncode != 0:
            raise RuntimeError("evaluation video ffmpeg failed: %d" % returncode)


def evaluate_episode(policy, normalizer, push_config, device, seed, video_path=None):
    seed_all(seed)
    env = environment(push_config, video=video_path is not None)
    writer = None
    try:
        env.initialize()
        observation = env.reset()
        if video_path is not None:
            writer = RawVideoWriter(video_path)
        episode_return = 0.0
        length = 0
        minimum_distance = None
        action_norm_sum = 0.0
        saturation_sum = 0.0
        final_info = {}
        done = False
        with torch.no_grad():
            while not done and length < push_config.max_steps:
                raw = flatten_observation(observation).to(device)
                tensor = normalizer.normalize(raw, update=False)
                action, _, _ = policy.act(tensor, deterministic=True)
                observation, reward, done, final_info = env.step(
                    tuple(float(value) for value in action.detach().cpu())
                )
                if writer is not None:
                    writer.write(env.backend.capture_camera_frame())
                length += 1
                episode_return += float(reward)
                distance = float(final_info["distance_to_goal"])
                minimum_distance = distance if minimum_distance is None else min(
                    minimum_distance, distance
                )
                action_norm_sum += float(final_info.get("action_norm", 0.0))
                saturation_sum += float(final_info.get("saturation_rate", 0.0))
        forced_timeout = not done
        return {
            "seed": seed,
            "success": bool(final_info.get("success", False)) and not forced_timeout,
            "timeout": bool(final_info.get("timeout", False)) or forced_timeout,
            "failure": bool(final_info.get("failure", False)),
            "episode_return": episode_return,
            "episode_length": length,
            "final_distance": float(final_info.get("distance_to_goal", float("inf"))),
            "minimum_distance": float(minimum_distance if minimum_distance is not None else float("inf")),
            "object_displacement": float(final_info.get("object_displacement", 0.0)),
            "action_norm": action_norm_sum / max(1, length),
            "action_saturation_rate": saturation_sum / max(1, length),
        }
    finally:
        if writer is not None:
            writer.close()
        env.close()


def mean(records, field):
    return sum(float(record[field]) for record in records) / len(records)


def main():
    args = parser().parse_args()
    if args.episode_steps != 256:
        raise ValueError("training mode requires --episode-steps=256")
    if args.physics_steps_per_action != 8:
        raise ValueError("training mode requires --physics-steps-per-action=8")
    if args.rollout_steps * args.num_updates != 20000:
        raise ValueError("diagnostic training requires exactly 20000 total steps")
    if args.evaluation_episodes != 100:
        raise ValueError("diagnostic evaluation requires exactly 100 episodes")
    run_root = args.artifacts_root / args.project_id / args.run_id
    if run_root.exists():
        raise FileExistsError("refusing to reuse artifact run root: %s" % run_root)
    run_root.mkdir(parents=True)
    config = vars(args).copy()
    config["artifacts_root"] = str(args.artifacts_root)
    config["device"] = "cuda:0"
    config["total_training_steps"] = args.rollout_steps * args.num_updates
    (run_root / "config").mkdir()
    (run_root / "config" / "diagnostic_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    seed_all(args.seed)
    push_config = PushConfig(
        max_steps=args.episode_steps,
        physics_steps_per_action=args.physics_steps_per_action,
    )
    env = environment(push_config)
    trainer = None
    try:
        env.initialize()
        trainer = PPOTrainer(
            env,
            config=PPOConfig(
                rollout_steps=args.rollout_steps,
                update_epochs=args.update_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            ),
            device="cuda:0",
        )
        records = trainer.run_training(PPOTrainingConfig(
            rollout_steps=args.rollout_steps,
            num_updates=args.num_updates,
            project_id=args.project_id,
            run_id=args.run_id,
            artifacts_root=args.artifacts_root,
            checkpoint_every=args.checkpoint_every,
        ))
    finally:
        env.close()

    finite = all(
        math.isfinite(float(value))
        for record in records
        for value in record.values()
        if isinstance(value, (int, float))
    ) and all(torch.isfinite(parameter).all().item() for parameter in trainer.policy.parameters())
    evaluation_dir = run_root / "evaluation"
    evaluation_dir.mkdir()
    training_summary = {
        "training_finite_no_nan": finite,
        "training_steps": trainer.training_steps,
        "update_steps": trainer.update_steps,
        "checkpoint": str(run_root / "checkpoints" / "model_latest.pt"),
    }
    (evaluation_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Isaac Gym Preview 4 can block when creating a second simulator after the
    # training simulator was destroyed. Evaluation therefore owns a fresh child
    # process and reuses one simulator across all deterministic resets.
    subprocess.run([
        sys.executable,
        str(Path(__file__).with_name("push_diagnostic_evaluation.py")),
        "--run-root", str(run_root),
        "--episodes", str(args.evaluation_episodes),
    ], check=True)
    print("DIAGNOSTIC_RESULT=" + json.dumps(training_summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
