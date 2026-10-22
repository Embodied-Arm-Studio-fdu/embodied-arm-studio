#!/usr/bin/env python3
"""One-update real-GPU PPO smoke test over the virtual S1 PushAdapter."""

import argparse
from pathlib import Path

# Isaac Gym Preview 4 requires its binding to load before PyTorch. PPO modules
# import torch, so an executable GPU entry point must bootstrap Gym first.
from isaacgym import gymapi as _gymapi  # noqa: F401

from simulation_core.ppo import PPOConfig, PPOTrainer
from simulation_core.push_adapter import IsaacGymPushBackend, PushAdapter, PushConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--project-id", default="phase-3-5-1")
    parser.add_argument("--run-id", default="ppo-smoke")
    args = parser.parse_args()
    if args.rollout_steps <= 1:
        parser.error("--rollout-steps must be greater than one")

    push_config = PushConfig(physics_steps_per_action=2, max_steps=300)
    backend = IsaacGymPushBackend(push_config)
    environment = PushAdapter(push_config, backend=backend)
    try:
        environment.initialize()
        trainer = PPOTrainer(
            environment,
            config=PPOConfig(
                rollout_steps=args.rollout_steps,
                update_epochs=1,
                batch_size=min(8, args.rollout_steps),
            ),
            device="cuda:0",
        )
        metrics = trainer.train_one_update()
        checkpoint = trainer.save_checkpoint(
            args.artifacts_root, args.project_id, args.run_id
        )
        print("PPO_SMOKE_RESULT=success")
        print("training_steps=%d update_steps=%d" % (
            trainer.training_steps, trainer.update_steps
        ))
        print("rollout_reward=%s" % metrics["rollout_reward"])
        print("policy_loss=%s value_loss=%s entropy=%s" % (
            metrics["policy_loss"], metrics["value_loss"], metrics["entropy"]
        ))
        print("checkpoint=%s" % checkpoint)
    finally:
        environment.close()
        print("CLEANUP=environment_closed")


if __name__ == "__main__":
    main()
