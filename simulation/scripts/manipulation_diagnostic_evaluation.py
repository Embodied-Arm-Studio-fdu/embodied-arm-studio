#!/usr/bin/env python3
"""Fresh-process deterministic evaluation and successful replay for S1 manipulation."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from isaacgym import gymapi as _gymapi  # noqa: F401
import torch

from push_diagnostic_training import RawVideoWriter, seed_all
from simulation_core.evaluation import wilson_confidence_interval
from simulation_core.ppo import PPOConfig
from simulation_core.experimental.manipulation.grasp_env import (
    GraspEnvConfig, GraspPPOEnvironment,
)
from simulation_core.experimental.manipulation.grasp_ppo import (
    ExperimentalGraspPPOTrainer, deterministic_evaluate as evaluate_grasp,
)
from simulation_core.experimental.manipulation.lift_env import (
    LiftEnvConfig, LiftPPOEnvironment,
)
from simulation_core.experimental.manipulation.lift_ppo import (
    ExperimentalLiftPPOTrainer, deterministic_evaluate as evaluate_lift,
)
from simulation_core.experimental.manipulation.lower_release_schema import (
    LOWER_RELEASE_POLICY_SCHEMA,
)
from simulation_core.experimental.manipulation.staged_place_env import (
    StagedFullPlaceEnvironment, StagedPlaceEnvConfig,
)
from simulation_core.ppo.normalizer import ObservationNormalizer
from simulation_core.ppo.policy import MLPActorCritic, flatten_observation


TASK_IDS = ("s1_grasp", "s1_lift", "s1_lower_release")
SIMULATION_ROOT = Path(__file__).resolve().parents[1]
PLACE_TRANSPORT_CHECKPOINT = (
    SIMULATION_ROOT
    / "artifacts/experimental/p4_r7_s1_place/train_3k/checkpoints/model_latest.pt"
)
PLACE_TRANSPORT_SHA256 = (
    "d2f7f662efe7d7c978ac241050664dfd8d94e3b192bd68806ffbb69bef7ea587"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenPlacePolicy:
    """Validated deterministic policy runtime used by staged Place evaluation."""

    def __init__(self, checkpoint, expected_mask, *, expected_sha256=None):
        checkpoint = Path(checkpoint)
        digest = file_sha256(checkpoint)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("staged Place checkpoint SHA-256 mismatch")
        payload = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        expected = {
            "task_id": "s1_place",
            "observation_dim": 32,
            "action_dim": 4,
            "active_action_mask": tuple(expected_mask),
        }
        for name, value in expected.items():
            actual = payload.get(name)
            if name == "active_action_mask":
                actual = tuple(actual or ())
            if actual != value:
                raise ValueError("staged Place checkpoint %s mismatch" % name)
        self.policy = MLPActorCritic(
            32, 4, action_low=(-1.0,) * 4, action_high=(1.0,) * 4,
        ).to("cuda:0")
        self.policy.load_state_dict(payload["policy"])
        self.policy.eval()
        self.normalizer = ObservationNormalizer(
            dimension=32, previous_action_slice=slice(0, 32),
        )
        self.normalizer.load_state_dict(payload["normalizer_state"], frozen=True)
        self.sha256 = digest

    def act(self, observation):
        raw = flatten_observation(observation, LOWER_RELEASE_POLICY_SCHEMA).to("cuda:0")
        normalized = self.normalizer.normalize(raw, update=False)
        with torch.no_grad():
            action, _, _ = self.policy.act(normalized, deterministic=True)
        return tuple(float(value) for value in action.detach().cpu())


def evaluate_staged_place(environment, runtimes, episodes=100):
    records = []
    fields = (
        "transport_target_reached", "transport_success",
        "policy_switched_to_lower", "legal_release_height",
        "auto_release_ready", "release_started", "release_latched",
        "actual_jaw_opened", "detached", "table_contact",
        "inside_target_zone", "settled_0p5s", "settled_1p0s",
        "xy_error", "bottom_clearance", "minimum_bottom_clearance",
        "release_height", "detach_height", "cube_linear_speed",
        "cube_angular_speed", "cube_tilt", "grasp_valid",
        "workspace_violation", "drop_outside_target", "table_penetration",
        "object_unstable", "unsafe", "non_finite", "staged_phase",
        "auto_release_condition",
    )
    for episode in range(episodes):
        observation = environment.reset()
        done = False
        reward_sum = action_norm_sum = saturation_sum = 0.0
        steps = 0
        while not done:
            if environment.staged_phase == "TRANSPORT":
                action = runtimes.transport.act(observation)
            elif environment.staged_phase == "LOWER":
                action = runtimes.lower.act(observation)
            else:
                action = (0.0, 0.0, 0.0, 0.0)
            observation, reward, done, info = environment.step(action)
            reward_sum += float(reward)
            action_norm_sum += math.sqrt(sum(value * value for value in action))
            saturation_sum += sum(abs(value) >= 0.999 for value in action) / 4.0
            steps += 1
            if steps > environment.max_steps + 1:
                raise RuntimeError("staged Place evaluation exceeded horizon")
        records.append({
            "episode": episode,
            "success": bool(info["success"]),
            "failure_reason": str(info.get("failure_reason") or ""),
            "reward": reward_sum,
            "steps": steps,
            "action_norm": action_norm_sum / steps,
            "action_saturation_rate": saturation_sum / steps,
            **{name: info.get(name) for name in fields},
        })

    count = len(records)
    successes = sum(row["success"] for row in records)
    lower, upper = wilson_confidence_interval(successes, count)
    tally = lambda name: sum(bool(row.get(name)) for row in records)
    histogram = {}
    for row in records:
        reason = row["failure_reason"] or "SUCCESS"
        histogram[reason] = histogram.get(reason, 0) + 1
    summary = {
        "deterministic": True,
        "episodes": count,
        "success_count": successes,
        "success_rate": successes / count,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "transport_checkpoint_sha256": runtimes.transport.sha256,
        "lower_checkpoint_sha256": runtimes.lower.sha256,
        "failure_reason_histogram": histogram,
        "timeout_count": histogram.get("TIMEOUT", 0),
        "workspace_violation_count": tally("workspace_violation"),
        "drop_outside_target_count": tally("drop_outside_target"),
        "table_penetration_count": tally("table_penetration"),
        "unstable_placement_count": tally("object_unstable"),
        "unsafe_count": tally("unsafe"),
        "non_finite_count": tally("non_finite"),
        "behavior_funnel": {
            "valid_reset": count,
            "transport_target_reached": tally("transport_target_reached"),
            "transport_success": tally("transport_success"),
            "policy_switched_to_lower": tally("policy_switched_to_lower"),
            "legal_release_height": tally("legal_release_height"),
            "auto_release_ready": tally("auto_release_ready"),
            "release_latched": tally("release_latched"),
            "actual_jaw_opened": tally("actual_jaw_opened"),
            "detached": tally("detached"),
            "table_contact": tally("table_contact"),
            "settled_0p5s": tally("settled_0p5s"),
            "settled_1p0s": tally("settled_1p0s"),
            "success": successes,
        },
    }
    return summary, records


def load_run_config(run_root, task_id):
    values = json.loads((run_root / "config/training_config.json").read_text(
        encoding="utf-8"
    ))
    if values.get("environment_type") != task_id:
        raise ValueError("evaluation task does not match the frozen Run config")
    return values


def evaluation_checkpoint(run_root, task_id):
    return run_root / "checkpoints" / (
        "model_best.pt"
        if task_id in {"s1_grasp", "s1_lift"}
        else "model_latest.pt"
    )


def build_runtime(run_root, task_id, *, camera=False, frame_callback=None):
    values = load_run_config(run_root, task_id)
    checkpoint = evaluation_checkpoint(run_root, task_id)
    ppo = PPOConfig(
        rollout_steps=int(values["rollout_steps"]),
        update_epochs=int(values.get("update_epochs", 1)),
        batch_size=int(values.get("batch_size", 16)),
    )
    if task_id == "s1_grasp":
        config = GraspEnvConfig(
            physics_steps_per_action=int(values["physics_steps_per_action"]),
            learning_horizon=int(values["episode_steps"]),
            xyz_delta_scale=float(values["action_delta_m"]),
            randomize_xy=float(values["randomize_xy_m"]),
            random_seed=int(values["seed"]),
        )
        environment = GraspPPOEnvironment(
            config, with_camera=camera, frame_callback=frame_callback,
        )
        trainer_type, evaluator = ExperimentalGraspPPOTrainer, evaluate_grasp
    elif task_id == "s1_lift":
        hold_seconds = float(values["hold_seconds"])
        config = LiftEnvConfig(
            physics_steps_per_action=int(values["physics_steps_per_action"]),
            max_episode_seconds=9.0 if hold_seconds == 5.0 else 6.0,
            dz_scale=float(values["action_delta_m"]),
            hold_seconds=hold_seconds,
        )
        environment = LiftPPOEnvironment(
            config, with_camera=camera, frame_callback=frame_callback,
        )
        trainer_type, evaluator = ExperimentalLiftPPOTrainer, evaluate_lift
    else:
        config = StagedPlaceEnvConfig(
            physics_steps_per_action=int(values["physics_steps_per_action"]),
            dz_scale=float(values["action_delta_m"]),
            release_trigger_threshold=float(values["release_trigger_threshold"]),
        )
        environment = StagedFullPlaceEnvironment(
            config, with_camera=camera, frame_callback=frame_callback,
        )
        environment.initialize()
        runtimes = SimpleNamespace(
            transport=FrozenPlacePolicy(
                PLACE_TRANSPORT_CHECKPOINT, (1, 1, 0, 0),
                expected_sha256=PLACE_TRANSPORT_SHA256,
            ),
            lower=FrozenPlacePolicy(checkpoint, (0, 0, 1, 1)),
        )
        return environment, runtimes, evaluate_staged_place
    environment.initialize()
    trainer = trainer_type(environment, config=ppo, device="cuda:0")
    trainer.load_checkpoint(checkpoint)
    return environment, trainer, evaluator


def generic_record(task_id, index, record):
    success = bool(record.get("success", False))
    reason = str(record.get("failure_reason", ""))
    timeout = reason.startswith("TIMEOUT")
    final_distance = float(record.get(
        "final_xy_error", record.get("xy_error", record.get("final_distance", 0.0))
    ))
    minimum_distance = float(record.get(
        "minimum_xy_error", record.get("minimum_distance", final_distance)
    ))
    result = dict(record)
    result.update({
        "seed": int(index),
        "success": success,
        "timeout": timeout,
        "failure": not success and not timeout,
        "episode_return": float(record.get("reward", 0.0)),
        "episode_length": int(record.get(
            "learning_steps", record.get("steps", 0)
        )),
        "final_distance": final_distance,
        "minimum_distance": minimum_distance,
        "object_displacement": float(record.get(
            "cube_rise", record.get("maximum_cube_rise", 0.0)
        )),
        "action_norm": float(record.get("action_norm", 0.0)),
        "action_saturation_rate": float(record.get("action_saturation_rate", 0.0)),
        "task_id": task_id,
    })
    return result


def average(records, field):
    return sum(float(row[field]) for row in records) / len(records)


def run_headless(run_root, task_id, episodes):
    seed_all(load_run_config(run_root, task_id)["seed"])
    environment = trainer = None
    try:
        environment, trainer, evaluator = build_runtime(run_root, task_id)
        domain_summary, domain_records = evaluator(environment, trainer, episodes=episodes)
    finally:
        if environment is not None:
            environment.close()
    records = [generic_record(task_id, index, row)
               for index, row in enumerate(domain_records)]
    successes = sum(row["success"] for row in records)
    lower, upper = wilson_confidence_interval(successes, len(records))
    successful_seeds = [row["seed"] for row in records if row["success"]]
    summary = {
        "task_id": task_id,
        "episode_count": len(records),
        "success_count": successes,
        "success_rate": successes / len(records),
        "confidence_interval_95": {
            "method": "wilson", "lower": lower, "upper": upper,
        },
        "timeout_rate": average(records, "timeout"),
        "failure_rate": average(records, "failure"),
        "mean_episode_return": average(records, "episode_return"),
        "mean_episode_length": average(records, "episode_length"),
        "mean_final_distance": average(records, "final_distance"),
        "mean_minimum_distance": average(records, "minimum_distance"),
        "mean_object_displacement": average(records, "object_displacement"),
        "mean_action_norm": average(records, "action_norm"),
        "action_saturation_rate": average(records, "action_saturation_rate"),
        "checkpoint_sha256": file_sha256(
            evaluation_checkpoint(run_root, task_id)
        ),
        "selected_success_seed": min(successful_seeds) if successful_seeds else None,
        "evaluation_video": {
            "status": "unavailable", "failure_reason": "replay_not_attempted",
        },
        "task_metrics": domain_summary,
    }
    if not all(math.isfinite(float(value)) for row in records for value in row.values()
               if isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise RuntimeError("non-finite manipulation evaluation metric")
    output = run_root / "evaluation"
    output.mkdir(exist_ok=True)
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("EVALUATION_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)


def run_replay(run_root, task_id, seed, output_path, fps):
    writer = environment = None
    partial = output_path.with_name(output_path.name + ".partial.mp4")
    try:
        seed_all(load_run_config(run_root, task_id)["seed"])
        environment, trainer, evaluator = build_runtime(
            run_root, task_id, camera=True,
        )
        if seed:
            evaluator(environment, trainer, episodes=seed)
        writer = RawVideoWriter(partial, width=640, height=480, fps=fps)
        environment.frame_callback = writer.write
        _, records = evaluator(environment, trainer, episodes=1)
        writer.close()
        writer = None
        record = generic_record(task_id, seed, records[0])
        if record["success"] is not True:
            raise RuntimeError("selected evaluation seed did not replay successfully")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(partial), str(output_path))
        print("EVALUATION_REPLAY=" + json.dumps(record, sort_keys=True), flush=True)
    finally:
        if writer is not None:
            writer.close()
        if environment is not None:
            environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--replay-seed", type=int)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=int, default=12)
    args = parser.parse_args()
    if args.replay_seed is None:
        if args.episodes != 100:
            raise ValueError("production evaluation requires 100 episodes")
        run_headless(args.run_root, args.task_id, args.episodes)
    else:
        if args.video_output is None:
            raise ValueError("successful replay requires --video-output")
        run_replay(
            args.run_root, args.task_id, args.replay_seed,
            args.video_output, args.video_fps,
        )


if __name__ == "__main__":
    main()
