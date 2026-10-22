"""Versioned policy and reward contract for experimental S1 Lift PPO."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ...ppo.policy import PolicySchema


LIFT_OBSERVATION_SCHEMA_ID = "s1-lift-observation/v1"
LIFT_ACTION_SCHEMA_ID = "s1-lift-action/v1"
LIFT_REWARD_VERSION = "s1-lift-reward/v1"
LIFT_REWARD_SCHEMA_VERSION = 1
LIFT_SUCCESS_CRITERIA_VERSION = "s1-lift-success/v1"
LIFT_LONG_HOLD_SUCCESS_CRITERIA_VERSION = "s1-lift-success/v2-5s-hold"

LIFT_OBSERVATION_FIELDS = (
    "ee_z", "ee_linear_z_velocity", "cube_z", "cube_linear_velocity",
    "cube_angular_velocity", "cube_rise", "ee_cube_relative_position",
    "relative_rotation_error", "jaw_actual", "blockage_gap",
    "target_height_error", "previous_dz_action",
)
LIFT_OBSERVATION_SIZES = (1, 1, 1, 3, 3, 1, 3, 1, 1, 1, 1, 1)
LIFT_OBSERVATION_SEMANTICS = (
    "ee_z_workspace_center_half_range_normalized[1]",
    "ee_linear_z_velocity_div_1mps[1]",
    "cube_z_relative_nominal_div_0p10m[1]",
    "cube_linear_velocity_div_1mps[3]",
    "cube_angular_velocity_div_5radps[3]",
    "cube_rise_div_target_0p060m[1]",
    "cube_minus_ee_translation_div_0p15m[3]",
    "cube_rotation_from_lift_reset_div_0p25rad[1]",
    "mean_jaw_q_div_0p05m[1]",
    "actual_minus_commanded_jaw_div_0p002m[1]",
    "target_rise_error_div_0p060m[1]",
    "previous_normalized_dz[1]",
)
LIFT_OBSERVATION_DIM = sum(LIFT_OBSERVATION_SIZES)
LIFT_ACTION_DIM = 1
LIFT_ACTION_SEMANTICS = "normalized_persistent_ee_z_delta_fixed_xy_orientation_and_jaw"

LIFT_POLICY_SCHEMA = PolicySchema(
    "s1_lift", 1, LIFT_OBSERVATION_FIELDS, LIFT_OBSERVATION_SIZES,
    LIFT_OBSERVATION_SEMANTICS, LIFT_ACTION_DIM, LIFT_ACTION_SEMANTICS,
    (-1.0,), (1.0,),
)

LIFT_REWARD_TERM_NAMES = (
    "upward_progress", "target_band", "hold_progress", "stability", "action_failure",
)
LIFT_REWARD_SEMANTICS = (
    "new_max_cube_rise_progress_clipped_at_target",
    "per_step_reward_inside_55_to_65mm_band",
    "incremental_continuous_valid_hold_reward",
    "translation_rotation_and_in_band_vertical_velocity_penalty",
    "small_dz_cost_and_terminal_drop_or_unsafe_penalty",
)


@dataclass(frozen=True)
class LiftRewardConfig:
    upward_progress: float = 100.0
    target_band: float = 0.25
    hold_progress: float = 0.50
    translation_drift: float = 2.0
    angular_velocity: float = 0.02
    band_vertical_velocity: float = 0.10
    action_magnitude: float = 0.001
    failure: float = 10.0


def reward_schema_metadata(config) -> Mapping[str, object]:
    reward = getattr(config, "reward", LiftRewardConfig())
    return {
        "reward_schema_version": LIFT_REWARD_SCHEMA_VERSION,
        "reward_version": LIFT_REWARD_VERSION,
        "reward_semantics": LIFT_REWARD_SEMANTICS,
        "reward_weights": {
            name: float(getattr(reward, name)) for name in reward.__dataclass_fields__
        },
        "success_criteria_version": LIFT_SUCCESS_CRITERIA_VERSION,
    }


def compute_lift_reward(*, config: LiftRewardConfig, new_max_progress: float,
                        in_target_band: bool, hold_valid: bool,
                        translation_drift: float, cube_angular_speed: float,
                        cube_vertical_speed: float, action: Sequence[float],
                        terminal_failure: bool) -> tuple:
    values = tuple(float(v) for v in action)
    scalars = (new_max_progress, translation_drift, cube_angular_speed,
               cube_vertical_speed) + values
    if len(values) != 1 or not all(math.isfinite(float(v)) for v in scalars):
        raise ValueError("lift reward requires one finite action and finite state")
    terms = {
        "upward_progress": config.upward_progress * max(0.0, float(new_max_progress)),
        "target_band": config.target_band if in_target_band else 0.0,
        "hold_progress": config.hold_progress if hold_valid else 0.0,
        "stability": -(
            config.translation_drift * max(0.0, float(translation_drift) - 0.003)
            + config.angular_velocity * float(cube_angular_speed)
            + (config.band_vertical_velocity * abs(float(cube_vertical_speed))
               if in_target_band else 0.0)
        ),
        "action_failure": -(
            config.action_magnitude * values[0] * values[0]
            + (config.failure if terminal_failure else 0.0)
        ),
    }
    if not all(math.isfinite(value) for value in terms.values()):
        raise ValueError("lift reward produced non-finite term")
    return sum(terms.values()), terms
