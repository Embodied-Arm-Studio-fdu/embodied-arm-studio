"""P4-2 reward and policy metadata for lower-and-release Place."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ...ppo.policy import PolicySchema
from .place_schema import (
    PLACE_ACTION_SCHEMA_ID, PLACE_OBSERVATION_DIM, PLACE_OBSERVATION_FIELDS,
    PLACE_OBSERVATION_SCHEMA_ID, PLACE_OBSERVATION_SEMANTICS,
    PLACE_OBSERVATION_SIZES,
)


LOWER_RELEASE_ACTIVE_ACTION_MASK = (0, 0, 1, 1)
LOWER_RELEASE_REWARD_VERSION = "s1-place-lower-release-reward/v2-trigger"
LOWER_RELEASE_REWARD_SCHEMA_VERSION = 2
LOWER_RELEASE_SUCCESS_CRITERIA_VERSION = "s1-place-lower-release-success/v2-trigger"
LOWER_RELEASE_ACTION_SEMANTICS = (
    "normalized_persistent_xyz_delta_plus_release_trigger;"
    "release_threshold_two_control_steps_then_latched_open"
)

LOWER_RELEASE_POLICY_SCHEMA = PolicySchema(
    "s1_place", 1, PLACE_OBSERVATION_FIELDS, PLACE_OBSERVATION_SIZES,
    PLACE_OBSERVATION_SEMANTICS, 4, LOWER_RELEASE_ACTION_SEMANTICS,
    (-1.0,) * 4, (1.0,) * 4,
)

LOWER_RELEASE_REWARD_TERM_NAMES = (
    "downward_progress", "release_readiness", "placement_accuracy",
    "settle_stability", "action_failure",
)
LOWER_RELEASE_REWARD_SEMANTICS = (
    "new_best_safe_cube_bottom_clearance_progress",
    "policy_release_trigger_latched_only_at_legal_release_height",
    "bounded_target_xy_accuracy_after_table_contact",
    "post_detach_low_linear_angular_velocity_continuous_settle",
    "small_active_action_early_open_and_terminal_failure_penalty",
)


@dataclass(frozen=True)
class LowerReleaseRewardConfig:
    downward_progress: float = 120.0
    release_readiness: float = 30.0
    placement_accuracy: float = 0.05
    settle_stability: float = 0.25
    success_bonus: float = 15.0
    action_magnitude: float = 0.001
    premature_opening: float = 5.0
    failure: float = 10.0


def reward_schema_metadata(config) -> Mapping[str, object]:
    reward = getattr(config, "reward", LowerReleaseRewardConfig())
    return {
        "reward_schema_version": LOWER_RELEASE_REWARD_SCHEMA_VERSION,
        "reward_version": LOWER_RELEASE_REWARD_VERSION,
        "reward_semantics": LOWER_RELEASE_REWARD_SEMANTICS,
        "reward_weights": {name: float(getattr(reward, name))
                           for name in reward.__dataclass_fields__},
        "success_criteria_version": LOWER_RELEASE_SUCCESS_CRITERIA_VERSION,
    }


def compute_lower_release_reward(*, config: LowerReleaseRewardConfig,
                                 downward_progress: float,
                                 legal_open_progress: float,
                                 premature_open_progress: float,
                                 in_target_zone: bool, table_contact: bool,
                                 settle_valid: bool, success_event: bool,
                                 action: Sequence[float], terminal_failure: bool):
    values = tuple(float(value) for value in action)
    scalars = (downward_progress, legal_open_progress,
               premature_open_progress) + values
    if len(values) != 4 or not all(math.isfinite(value) for value in scalars):
        raise ValueError("lower/release reward requires finite state and four actions")
    terms = {
        "downward_progress": config.downward_progress * max(0.0, downward_progress),
        "release_readiness": (
            config.release_readiness * max(0.0, legal_open_progress)
            - config.premature_opening * max(0.0, premature_open_progress)),
        "placement_accuracy": (
            config.placement_accuracy if table_contact and in_target_zone else 0.0),
        "settle_stability": (
            (config.settle_stability if settle_valid else 0.0)
            + (config.success_bonus if success_event else 0.0)),
        "action_failure": -(
            config.action_magnitude * (values[2] ** 2 + values[3] ** 2)
            + (config.failure if terminal_failure else 0.0)),
    }
    if not all(math.isfinite(value) for value in terms.values()):
        raise ValueError("lower/release reward produced non-finite term")
    return sum(terms.values()), terms


assert LOWER_RELEASE_POLICY_SCHEMA.observation_dim == PLACE_OBSERVATION_DIM
assert PLACE_OBSERVATION_SCHEMA_ID == "s1-place-observation/v1"
assert PLACE_ACTION_SCHEMA_ID == "s1-place-action/v1"
