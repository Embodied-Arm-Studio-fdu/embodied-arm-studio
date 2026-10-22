#!/usr/bin/env python3
"""One-update real-GPU PPO math smoke; not a policy performance run."""

import argparse
import json
import math
from pathlib import Path

from isaacgym import gymapi as _gymapi  # noqa: F401
import torch

from simulation_core.ppo import ACTION_DIM, OBSERVATION_DIM, PPOConfig, PPOTrainer
from simulation_core.push_adapter import IsaacGymPushBackend, PushAdapter, PushConfig


def changed(before, parameters):
    return any(not torch.equal(old, current.detach()) for old, current in zip(before, parameters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(14)
    config = PushConfig(max_steps=256, physics_steps_per_action=8)
    environment = PushAdapter(config, backend=IsaacGymPushBackend(config))
    ppo_config = PPOConfig(rollout_steps=260, update_epochs=2, batch_size=16)
    trainer = PPOTrainer(environment, config=ppo_config, device="cuda:0")
    try:
        environment.initialize()
        rollout = trainer.collect_rollout()
        actor_parameters = list(trainer.policy.actor.parameters()) + [trainer.policy.log_std]
        critic_parameters = list(trainer.policy.critic.parameters())
        actor_before = [parameter.detach().clone() for parameter in actor_parameters]
        critic_before = [parameter.detach().clone() for parameter in critic_parameters]
        diagnostics = trainer.update()
        actor_changed = changed(actor_before, actor_parameters)
        critic_changed = changed(critic_before, critic_parameters)
        parameters_finite = all(
            torch.isfinite(parameter).all().item() for parameter in trainer.policy.parameters()
        )
        buffer = trainer.buffer
        tensors_finite = all(torch.isfinite(value).all().item() for value in (
            torch.stack(buffer.observations), torch.stack(buffer.actions),
            torch.stack(buffer.values), torch.stack(buffer.next_values),
            torch.stack(buffer.log_probs), buffer.advantages, buffer.returns,
        ))
        timeout_boundaries = sum(
            done and bootstrap == 1.0 and continuation == 0.0
            for done, bootstrap, continuation in zip(
                buffer.dones, buffer.bootstrap_masks, buffer.continuation_masks
            )
        )
        true_terminal_boundaries = sum(
            done and bootstrap == 0.0 and continuation == 0.0
            for done, bootstrap, continuation in zip(
                buffer.dones, buffer.bootstrap_masks, buffer.continuation_masks
            )
        )
    finally:
        environment.close()
    expected_uses = ppo_config.rollout_steps * ppo_config.update_epochs
    passed = bool(
        len(buffer) == 260
        and tuple(torch.stack(buffer.observations).shape) == (260, OBSERVATION_DIM)
        and tuple(torch.stack(buffer.actions).shape) == (260, ACTION_DIM)
        and timeout_boundaries >= 1
        and diagnostics["samples_consumed_per_epoch"] == [260, 260]
        and diagnostics["total_sample_uses"] == expected_uses
        and actor_changed and critic_changed and parameters_finite and tensors_finite
        and all(
            math.isfinite(float(value))
            for value in diagnostics.values()
            if isinstance(value, (int, float))
        )
        and 0.0 <= diagnostics["clip_fraction"] <= 1.0
    )
    result = {
        "gate": "P1.4 PPO correctness one-update GPU smoke",
        "runtime": "Isaac Gym Preview 4, GPU PhysX, RTX 4080",
        "not_a_training_result": True,
        "environment": {
            "action_dim": ACTION_DIM,
            "observation_dim": OBSERVATION_DIM,
            "episode_horizon": config.max_steps,
            "physics_steps_per_action": config.physics_steps_per_action,
        },
        "rollout": {
            "samples": len(buffer),
            "observation_shape": list(torch.stack(buffer.observations).shape),
            "action_shape": list(torch.stack(buffer.actions).shape),
            "value_shape": list(torch.stack(buffer.values).shape),
            "next_value_shape": list(torch.stack(buffer.next_values).shape),
            "log_prob_shape": list(torch.stack(buffer.log_probs).shape),
            "advantage_shape": list(buffer.advantages.shape),
            "return_shape": list(buffer.returns.shape),
            "timeout_boundaries": timeout_boundaries,
            "true_terminal_boundaries": true_terminal_boundaries,
            "bootstrap_masks": buffer.bootstrap_masks,
            "continuation_masks": buffer.continuation_masks,
            "reward_finite": all(math.isfinite(value) for value in buffer.rewards),
            "all_tensors_finite": tensors_finite,
            "rollout_metrics": rollout,
        },
        "update": diagnostics,
        "actor_parameters_changed": actor_changed,
        "critic_parameters_changed": critic_changed,
        "parameters_finite_after_update": parameters_finite,
        "result": "PASS" if passed else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("P1_4_PPO_CORRECTNESS_GATE=%s" % result["result"])
    print(json.dumps({
        "rollout": {key: value for key, value in result["rollout"].items() if key not in ("bootstrap_masks", "continuation_masks")},
        "update": diagnostics,
        "actor_parameters_changed": actor_changed,
        "critic_parameters_changed": critic_changed,
        "parameters_finite_after_update": parameters_finite,
    }, indent=2))


if __name__ == "__main__":
    main()
