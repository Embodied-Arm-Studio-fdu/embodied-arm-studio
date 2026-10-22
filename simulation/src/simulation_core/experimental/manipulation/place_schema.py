"""Versioned policy/reward contract for experimental S1 Place."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ...ppo.policy import PolicySchema


PLACE_OBSERVATION_SCHEMA_ID = "s1-place-observation/v1"
PLACE_ACTION_SCHEMA_ID = "s1-place-action/v1"
PLACE_REWARD_VERSION = "s1-place-reward/v1"
PLACE_REWARD_SCHEMA_VERSION = 1
PLACE_SUCCESS_CRITERIA_VERSION = "s1-place-transport-success/v1"
PLACE_ACTIVE_ACTION_MASK = (1, 1, 0, 0)

PLACE_OBSERVATION_FIELDS = (
    "ee_position", "ee_velocity", "cube_position", "cube_linear_velocity",
    "cube_angular_velocity", "target_position", "cube_target_relative",
    "ee_cube_relative", "relative_rotation_error", "jaw_actual",
    "blockage_gap", "grasp_stability", "previous_action",
)
PLACE_OBSERVATION_SIZES = (3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 4)
PLACE_OBSERVATION_SEMANTICS = (
    "ee_xyz_workspace_center_half_range_normalized[3]",
    "ee_linear_velocity_div_1mps[3]",
    "cube_xyz_relative_nominal_div_0p15m[3]",
    "cube_linear_velocity_div_1mps[3]",
    "cube_angular_velocity_div_5radps[3]",
    "target_xyz_relative_nominal_div_0p15m[3]",
    "cube_minus_target_div_0p15m[3]",
    "cube_minus_ee_div_0p15m[3]",
    "cube_rotation_from_reset_div_0p25rad[1]",
    "mean_jaw_q_div_0p05m[1]",
    "actual_minus_commanded_jaw_div_0p002m[1]",
    "binary_stable_grasp[1]",
    "previous_normalized_dx_dy_dz_grip[4]",
)
PLACE_OBSERVATION_DIM = sum(PLACE_OBSERVATION_SIZES)
PLACE_ACTION_DIM = 4
PLACE_ACTION_SEMANTICS = (
    "normalized_persistent_xyz_delta_plus_absolute_grip;"
    "p4_1_active_mask_dx1_dy1_dz0_grip0"
)

PLACE_POLICY_SCHEMA = PolicySchema(
    "s1_place", 1, PLACE_OBSERVATION_FIELDS, PLACE_OBSERVATION_SIZES,
    PLACE_OBSERVATION_SEMANTICS, PLACE_ACTION_DIM, PLACE_ACTION_SEMANTICS,
    (-1.0,) * 4, (1.0,) * 4,
)

PLACE_REWARD_TERM_NAMES = (
    "xy_progress", "target_zone", "height_stability", "grasp_stability",
    "action_failure",
)
PLACE_REWARD_SEMANTICS = (
    "new_best_cube_target_xy_distance_progress",
    "per_step_reward_inside_15mm_target_zone",
    "transport_height_deviation_penalty",
    "ee_cube_drift_and_cube_angular_velocity_penalty",
    "small_action_cost_and_terminal_failure_penalty",
)


@dataclass(frozen=True)
class PlaceRewardConfig:
    xy_progress: float = 100.0
    target_zone: float = 0.25
    height_stability: float = 8.0
    translation_drift: float = 2.0
    angular_velocity: float = 0.02
    action_magnitude: float = 0.001
    failure: float = 10.0


def reward_schema_metadata(config) -> Mapping[str, object]:
    reward = getattr(config, "reward", PlaceRewardConfig())
    return {"reward_schema_version": PLACE_REWARD_SCHEMA_VERSION,
            "reward_version": PLACE_REWARD_VERSION,
            "reward_semantics": PLACE_REWARD_SEMANTICS,
            "reward_weights": {name: float(getattr(reward, name))
                               for name in reward.__dataclass_fields__},
            "success_criteria_version": PLACE_SUCCESS_CRITERIA_VERSION}


def compute_place_reward(*, config: PlaceRewardConfig, new_best_progress: float,
                         in_target_zone: bool, height_error: float,
                         translation_drift: float, angular_speed: float,
                         action: Sequence[float], terminal_failure: bool):
    values = tuple(float(v) for v in action)
    scalars = (new_best_progress, height_error, translation_drift, angular_speed) + values
    if len(values) != 4 or not all(math.isfinite(v) for v in scalars):
        raise ValueError("place reward requires four finite actions and finite state")
    terms = {
        "xy_progress": config.xy_progress * max(0.0, float(new_best_progress)),
        "target_zone": config.target_zone if in_target_zone else 0.0,
        "height_stability": -config.height_stability * abs(float(height_error)),
        "grasp_stability": -(
            config.translation_drift * max(0.0, float(translation_drift) - .003)
            + config.angular_velocity * float(angular_speed)),
        "action_failure": -(
            config.action_magnitude * sum(v * v for v in values)
            + (config.failure if terminal_failure else 0.0)),
    }
    if not all(math.isfinite(v) for v in terms.values()):
        raise ValueError("place reward produced non-finite term")
    return sum(terms.values()), terms
