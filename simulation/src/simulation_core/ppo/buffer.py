"""Minimal on-policy rollout storage and GAE calculation."""

from typing import Dict, Iterator

import torch

from .policy import ACTION_DIM


class RolloutBuffer:
    def __init__(self, action_dim: int = ACTION_DIM) -> None:
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError("rollout action_dim must be a positive integer")
        self.action_dim = action_dim
        self.clear()

    def clear(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.next_values = []
        self.bootstrap_masks = []
        self.continuation_masks = []
        self.log_probs = []
        self.returns = None
        self.advantages = None

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self, observation, action, reward, done, value, log_prob,
        next_value, bootstrap_mask, continuation_mask,
    ) -> None:
        if tuple(action.shape) != (self.action_dim,):
            raise ValueError("rollout action must have shape (%d,)" % self.action_dim)
        self.observations.append(observation.detach().clone())
        self.actions.append(action.detach().clone())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(value.detach().clone().reshape(()))
        self.next_values.append(next_value.detach().clone().reshape(()))
        self.bootstrap_masks.append(float(bootstrap_mask))
        self.continuation_masks.append(float(continuation_mask))
        self.log_probs.append(log_prob.detach().clone().reshape(()))

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float
    ) -> None:
        if not self.rewards:
            raise ValueError("cannot compute returns for an empty rollout")
        values = torch.stack(self.values)
        next_values = torch.stack(self.next_values)
        rewards = values.new_tensor(self.rewards)
        bootstrap_masks = values.new_tensor(self.bootstrap_masks)
        continuation_masks = values.new_tensor(self.continuation_masks)
        advantages = torch.zeros_like(rewards)
        gae = values.new_zeros(())
        for index in reversed(range(len(self))):
            delta = (
                rewards[index]
                + gamma * bootstrap_masks[index] * next_values[index]
                - values[index]
            )
            gae = delta + gamma * gae_lambda * continuation_masks[index] * gae
            advantages[index] = gae
        self.advantages = advantages
        self.returns = advantages + values

    def batches(self, batch_size: int, advantages=None) -> Iterator[Dict[str, torch.Tensor]]:
        if self.returns is None or self.advantages is None:
            raise ValueError("returns and advantages must be computed before sampling")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        data = {
            "observations": torch.stack(self.observations),
            "actions": torch.stack(self.actions),
            "old_log_probs": torch.stack(self.log_probs),
            "returns": self.returns,
            "advantages": self.advantages if advantages is None else advantages,
        }
        permutation = torch.randperm(len(self), device=data["observations"].device)
        for start in range(0, len(self), batch_size):
            indices = permutation[start : start + batch_size]
            yield {name: values[indices] for name, values in data.items()}
