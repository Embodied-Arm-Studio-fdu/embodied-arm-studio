#!/usr/bin/env python3
"""Configurable short-cycle PPO training entry for the virtual S1 Push task."""

import argparse
from pathlib import Path

# Isaac Gym Preview 4 must load before PPO imports PyTorch.
from isaacgym import gymapi as _gymapi  # noqa: F401

from simulation_core.ppo import PPOConfig, PPOTrainer, PPOTrainingConfig
from simulation_core.push_adapter import IsaacGymPushBackend, PushAdapter, PushConfig
from simulation_core.domain import (
    SMOKE_EPISODE_STEPS,
    TRAINING_EPISODE_STEPS,
    TRAINING_PHYSICS_STEPS_PER_ACTION,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-steps", type=int, required=True)
    parser.add_argument("--num-updates", type=int, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--physics-steps-per-action", type=int)
    parser.add_argument("--run-mode", choices=("smoke", "training"), default="smoke")
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--resume-from", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_steps = args.episode_steps
    if episode_steps is None:
        episode_steps = (
            SMOKE_EPISODE_STEPS
            if args.run_mode == "smoke"
            else TRAINING_EPISODE_STEPS
        )
    if args.run_mode == "training" and episode_steps != TRAINING_EPISODE_STEPS:
        raise ValueError(
            "training run mode requires --episode-steps=%d"
            % TRAINING_EPISODE_STEPS
        )
    physics_steps_per_action = args.physics_steps_per_action
    if physics_steps_per_action is None:
        physics_steps_per_action = (
            2
            if args.run_mode == "smoke"
            else TRAINING_PHYSICS_STEPS_PER_ACTION
        )
    if (
        args.run_mode == "training"
        and physics_steps_per_action != TRAINING_PHYSICS_STEPS_PER_ACTION
    ):
        raise ValueError(
            "training run mode requires --physics-steps-per-action=%d"
            % TRAINING_PHYSICS_STEPS_PER_ACTION
        )
    training = PPOTrainingConfig(
        rollout_steps=args.rollout_steps,
        num_updates=args.num_updates,
        project_id=args.project_id,
        run_id=args.run_id,
        artifacts_root=args.artifacts_root,
        checkpoint_every=args.checkpoint_every,
    )
    ppo_config = PPOConfig(
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
    )
    push_config = PushConfig(
        physics_steps_per_action=physics_steps_per_action,
        max_steps=episode_steps,
    )
    environment = PushAdapter(push_config, backend=IsaacGymPushBackend(push_config))
    try:
        environment.initialize()
        trainer = PPOTrainer(environment, config=ppo_config, device="cuda:0")
        if args.resume_from is not None:
            trainer.load_checkpoint(args.resume_from)
            print("RESUMED_FROM=%s" % args.resume_from)
        records = trainer.run_training(training)
        checkpoint = (
            training.artifacts_root
            / training.project_id
            / training.run_id
            / "checkpoints"
            / "model_latest.pt"
        )
        for record in records:
            print(
                "update={update_index} steps={training_steps} avg_reward={average_reward} "
                "episode_length={episode_length} policy_loss={policy_loss} "
                "value_loss={value_loss} entropy={entropy}".format(**record)
            )
        print("PPO_SHORT_TRAIN_RESULT=success")
        print("checkpoint=%s" % checkpoint)
    finally:
        environment.close()
        print("CLEANUP=environment_closed")


if __name__ == "__main__":
    main()
