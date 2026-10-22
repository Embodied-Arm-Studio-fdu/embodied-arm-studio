"""Versioned state/action/reward contracts for experimental S1 Grasp PPO."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ...ppo.policy import PolicySchema


GRASP_OBSERVATION_SCHEMA_ID = "s1-grasp-observation/v1"
GRASP_ACTION_SCHEMA_ID = "s1-grasp-action/v1"
GRASP_SUCCESS_CRITERIA_VERSION = "s1-grasp-success/v1"
GRASP_REWARD_SCHEMA_VERSION = 3
GRASP_REWARD_VERSION = "s1-grasp-reward/v3"

GRASP_OBSERVATION_FIELDS = (
    "arm_q", "arm_dq", "gripper_q", "gripper_dq", "ee_position",
    "object_position", "object_velocity", "ee_object_relative_position",
    "gripper_width", "previous_action", "grasp_corridor_error",
    "object_table_clearance",
)
GRASP_OBSERVATION_SIZES = (6, 6, 2, 2, 3, 3, 6, 3, 1, 4, 3, 1)
GRASP_OBSERVATION_SEMANTICS = (
    "arm_q_fixed_joint_range_normalized[6]",
    "arm_dq_div_10_normalized[6]",
    "gripper_q_0_to_0p05_normalized[2]",
    "gripper_dq_div_0p2_normalized[2]",
    "ee_position_workspace_center_half_range_normalized[3]",
    "object_position_relative_nominal_div_0p10_normalized[3]",
    "object_linear_div_1_angular_div_5_normalized[6]",
    "ee_to_object_div_0p15_normalized[3]",
    "estimated_aperture_div_0p0901_normalized[1]",
    "previous_action_normalized[4]",
    "object_minus_expected_corridor_div_0p10_normalized[3]",
    "object_table_clearance_div_0p05_normalized[1]",
)
GRASP_OBSERVATION_DIM = sum(GRASP_OBSERVATION_SIZES)
GRASP_ACTION_DIM = 4
GRASP_ACTION_SEMANTICS = (
    "normalized_persistent_xyz_delta_plus_absolute_symmetric_grip_fixed_orientation"
)

GRASP_POLICY_SCHEMA = PolicySchema(
    "s1_grasp", 1, GRASP_OBSERVATION_FIELDS, GRASP_OBSERVATION_SIZES,
    GRASP_OBSERVATION_SEMANTICS, GRASP_ACTION_DIM, GRASP_ACTION_SEMANTICS,
    (-1.0, -1.0, -1.0, -1.0), (1.0, 1.0, 1.0, 1.0),
)

GRASP_REWARD_TERM_NAMES = (
    "approach_progress", "alignment_progress", "controlled_descend",
    "closure_progress", "blockage_progress", "candidate_verified_bonus", "penalty",
)
GRASP_REWARD_SEMANTICS = (
    "signed_staged_grasp_frame_progress",
    "signed_horizontal_corridor_alignment_progress",
    "grasp_height_progress_only_when_xy_aligned",
    "gated_normalized_commanded_close_potential_difference",
    "contact_gated_normalized_blockage_potential_difference",
    "candidate_once_plus_scripted_verification_terminal_bonus",
    "action_saturation_clamp_premature_push_and_failure_penalty",
)


@dataclass(frozen=True)
class GraspRewardConfig:
    # Cartesian progress remains the locked v2 baseline.  v3 replaces only
    # the closure term with normalized close/blockage potential differences.
    approach_progress: float = 100.0
    alignment_progress: float = 50.0
    controlled_descend: float = 50.0
    closure_progress: float = 2.0
    blockage_progress: float = 3.0
    candidate_bonus: float = 5.0
    verified_bonus: float = 20.0
    action_magnitude: float = 0.002
    saturation: float = 0.01
    clamp: float = 0.25
    failure: float = 5.0


def reward_schema_metadata(config) -> Mapping[str, object]:
    reward = getattr(config, "reward", GraspRewardConfig())
    return {
        "reward_schema_version": GRASP_REWARD_SCHEMA_VERSION,
        "reward_version": GRASP_REWARD_VERSION,
        "reward_semantics": GRASP_REWARD_SEMANTICS,
        "reward_weights": {
            name: float(getattr(reward, name)) for name in reward.__dataclass_fields__
        },
        "success_criteria_version": GRASP_SUCCESS_CRITERIA_VERSION,
    }


def closure_reward_potentials(*, open_safe: float, close_safe: float,
                              commanded_jaw: float, actual_jaw: float,
                              finite: bool, in_corridor: bool,
                              xy_aligned: bool, at_grasp_height: bool,
                              no_premature_push: bool, safe_contact: bool,
                              bilateral_contact: bool,
                              cube_contact: bool) -> tuple:
    """Return gated v3 close/blockage potentials and audit flags."""
    denominator = float(open_safe) - float(close_safe)
    if denominator <= 0:
        raise ValueError("open_safe must exceed close_safe")
    closure_gate = all((finite, in_corridor, xy_aligned, at_grasp_height,
                        no_premature_push, safe_contact))
    contact_gate = closure_gate and bilateral_contact and cube_contact
    close_fraction = max(0.0, min(
        1.0, (float(open_safe) - float(commanded_jaw)) / denominator))
    blockage_gap = max(0.0, float(actual_jaw) - float(commanded_jaw))
    normalized_blockage = max(0.0, min(1.0, blockage_gap / 0.002))
    return (
        close_fraction if closure_gate else 0.0,
        normalized_blockage if contact_gate else 0.0,
        {
            "closure_gate": closure_gate, "contact_gate": contact_gate,
            "close_fraction": close_fraction, "blockage_gap": blockage_gap,
            "normalized_blockage": normalized_blockage,
        },
    )


def compute_grasp_reward_v2(*, approach_progress: float,
                            alignment_progress: float, descend_progress: float,
                            xy_aligned: bool, closure_credit: float,
                            closure_gated: bool, candidate_event: bool,
                            verified_event: bool, action: Sequence[float],
                            saturated: float, clamped: bool,
                            failure: bool) -> tuple:
    """Frozen v2 equation retained only for the P2-R2 numerical audit."""
    action_values = tuple(float(v) for v in action)
    terms = {
        "approach_progress": 100.0 * float(approach_progress),
        "alignment_progress": 50.0 * float(alignment_progress),
        "controlled_descend": 50.0 * float(descend_progress) if xy_aligned else 0.0,
        "valid_closure": 40.0 * max(0.0, float(closure_credit)) if closure_gated else 0.0,
        "candidate_verified_bonus": (
            (5.0 if candidate_event else 0.0) + (20.0 if verified_event else 0.0)),
        "penalty": -(0.002 * sum(v * v for v in action_values)
                     + 0.01 * float(saturated) + (0.25 if clamped else 0.0)
                     + (5.0 if failure else 0.0)),
    }
    return sum(terms.values()), terms


def compute_grasp_reward(*, config: GraspRewardConfig, approach_progress: float,
                         alignment_progress: float, descend_progress: float,
                         xy_aligned: bool, closure_potential_progress: float,
                         blockage_potential_progress: float,
                         candidate_event: bool,
                         verified_event: bool, action: Sequence[float],
                         saturated: float, clamped: bool,
                         failure: bool) -> tuple:
    action_values = tuple(float(v) for v in action)
    if len(action_values) != 4 or not all(math.isfinite(v) for v in action_values):
        raise ValueError("grasp reward action must contain four finite values")
    terms = {
        "approach_progress": config.approach_progress * float(approach_progress),
        "alignment_progress": config.alignment_progress * float(alignment_progress),
        "controlled_descend": (
            config.controlled_descend * float(descend_progress) if xy_aligned else 0.0
        ),
        "closure_progress": (
            config.closure_progress * float(closure_potential_progress)
        ),
        "blockage_progress": (
            config.blockage_progress * float(blockage_potential_progress)
        ),
        "candidate_verified_bonus": (
            (config.candidate_bonus if candidate_event else 0.0)
            + (config.verified_bonus if verified_event else 0.0)
        ),
        "penalty": -(
            config.action_magnitude * sum(v * v for v in action_values)
            + config.saturation * float(saturated)
            + (config.clamp if clamped else 0.0)
            + (config.failure if failure else 0.0)
        ),
    }
    if not all(math.isfinite(value) for value in terms.values()):
        raise ValueError("grasp reward produced non-finite term")
    return sum(terms.values()), terms
