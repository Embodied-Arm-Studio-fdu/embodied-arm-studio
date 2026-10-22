"""Small bounded Gaussian MLP actor-critic for the minimal Push observation."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn


PUSH_OBSERVATION_SCHEMA_VERSION = 2
PUSH_OBSERVATION_SEMANTICS = (
    "arm_q_rad[6]",
    "arm_dq_rad_per_s[6]",
    "ee_to_object_xy_m[2]",
    "object_to_goal_xy_m[2]",
    "ee_velocity_xy_m_per_s[2]",
    "object_velocity_xy_m_per_s[2]",
    "target_tracking_error_xy_m[2]",
    "previous_action_xy_normalized[2]",
)
OBSERVATION_FIELDS = (
    "arm_q",
    "arm_dq",
    "ee_to_object_xy",
    "object_to_goal_xy",
    "ee_velocity_xy",
    "object_velocity_xy",
    "target_tracking_error_xy",
    "previous_action_xy",
)
OBSERVATION_SIZES = (6, 6, 2, 2, 2, 2, 2, 2)
OBSERVATION_SLICES = {
    name: slice(sum(OBSERVATION_SIZES[:index]), sum(OBSERVATION_SIZES[:index + 1]))
    for index, name in enumerate(OBSERVATION_FIELDS)
}
PUSH_OBSERVATION_DIM = sum(OBSERVATION_SIZES)
OBSERVATION_DIM = PUSH_OBSERVATION_DIM
ACTION_DIM = 2
ACTION_SEMANTICS = "xy_translation_fixed_z_fixed_orientation"

# Physical XY deltas consumed in full by PushActionMapper.
ACTION_LOW = (-0.005, -0.005)
ACTION_HIGH = (0.005, 0.005)


@dataclass(frozen=True)
class PolicySchema:
    task_id: str
    observation_schema_version: int
    observation_fields: tuple
    observation_sizes: tuple
    observation_semantics: tuple
    action_dim: int
    action_semantics: str
    action_low: tuple
    action_high: tuple

    @property
    def observation_dim(self):
        return sum(self.observation_sizes)


PUSH_POLICY_SCHEMA = PolicySchema(
    "s1_push", PUSH_OBSERVATION_SCHEMA_VERSION, OBSERVATION_FIELDS,
    OBSERVATION_SIZES, PUSH_OBSERVATION_SEMANTICS, ACTION_DIM,
    ACTION_SEMANTICS, ACTION_LOW, ACTION_HIGH,
)

REACH_POLICY_SCHEMA = PolicySchema(
    "s1_reach", 1,
    ("arm_q", "arm_dq", "ee_to_goal_xy", "ee_velocity_xy", "previous_action_xy"),
    (6, 6, 2, 2, 2),
    ("arm_q_rad[6]", "arm_dq_rad_per_s[6]", "ee_to_goal_xy_m[2]",
     "ee_velocity_xy_m_per_s[2]", "previous_action_xy_normalized[2]"),
    2, "normalized_xy_delta_fixed_z_fixed_orientation", (-1.0, -1.0), (1.0, 1.0),
)


def flatten_observation(
    observation: Mapping[str, Sequence[float]], schema: PolicySchema = PUSH_POLICY_SCHEMA
) -> torch.Tensor:
    """Convert an explicit task-aligned observation into one state tensor."""
    values = []
    for name, expected_size in zip(schema.observation_fields, schema.observation_sizes):
        if name not in observation:
            raise ValueError("observation is missing field: %s" % name)
        field = tuple(float(value) for value in observation[name])
        if len(field) != expected_size:
            raise ValueError(
                "observation field %s must contain %d values" % (name, expected_size)
            )
        if not all(math.isfinite(value) for value in field):
            raise ValueError("observation field %s contains non-finite values" % name)
        values.extend(field)
    return torch.tensor(values, dtype=torch.float32)


def _mlp(input_dim: int, output_dim: int, hidden_sizes) -> nn.Sequential:
    layers = []
    previous = input_dim
    for size in hidden_sizes:
        layers.extend((nn.Linear(previous, size), nn.Tanh()))
        previous = size
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class MLPActorCritic(nn.Module):
    """Independent actor/value MLPs with a tanh-squashed bounded action."""

    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes=(64, 64),
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("policy dimensions must be positive")
        if len(action_low) != action_dim or len(action_high) != action_dim:
            raise ValueError("action bounds must match action_dim")
        if any(float(low) >= float(high) for low, high in zip(action_low, action_high)):
            raise ValueError("each action low bound must be below its high bound")
        self.actor = _mlp(observation_dim, action_dim, hidden_sizes)
        self.critic = _mlp(observation_dim, 1, hidden_sizes)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))
        low = torch.tensor(action_low, dtype=torch.float32)
        high = torch.tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)
        self._active_action_mask = None

    def set_active_action_mask(self, mask):
        """Exclude environment-masked dimensions from PPO likelihood/entropy."""
        if mask is None:
            self._active_action_mask = None
            return
        values = tuple(bool(value) for value in mask)
        if len(values) != self.log_std.numel() or not any(values):
            raise ValueError("active action mask must match and enable an action dimension")
        self._active_action_mask = torch.tensor(values, dtype=torch.float32,
                                                device=self.log_std.device)

    def _distribution(self, observations):
        mean = self.actor(observations)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def _squash(self, raw_action):
        unit_action = torch.tanh(raw_action)
        return unit_action * self.action_scale + self.action_bias, unit_action

    def _log_prob(self, distribution, raw_action, unit_action):
        correction = torch.log(
            self.action_scale * (1.0 - unit_action.pow(2)) + 1.0e-6
        )
        terms = distribution.log_prob(raw_action) - correction
        if self._active_action_mask is not None:
            terms = terms * self._active_action_mask
        return terms.sum(dim=-1)

    def act(self, observations, deterministic: bool = False):
        distribution = self._distribution(observations)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        action, unit_action = self._squash(raw_action)
        log_prob = self._log_prob(distribution, raw_action, unit_action)
        value = self.critic(observations).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(self, observations, actions):
        distribution = self._distribution(observations)
        unit_action = ((actions - self.action_bias) / self.action_scale).clamp(
            -1.0 + 1.0e-6, 1.0 - 1.0e-6
        )
        raw_action = torch.atanh(unit_action)
        log_prob = self._log_prob(distribution, raw_action, unit_action)
        # Base Gaussian entropy is a stable proxy; exact squashed entropy has no
        # simple closed form and is unnecessary for this first smoke runner.
        entropy_terms = distribution.entropy()
        if self._active_action_mask is not None:
            entropy_terms = entropy_terms * self._active_action_mask
        entropy = entropy_terms.sum(dim=-1)
        value = self.critic(observations).squeeze(-1)
        return log_prob, entropy, value
