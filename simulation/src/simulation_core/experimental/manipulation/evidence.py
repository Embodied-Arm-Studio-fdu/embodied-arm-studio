"""Pure, testable multi-signal grasp evidence for the P1 scripted gate."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence, Tuple


def _finite(values):
    return all(math.isfinite(float(v)) for v in values)


def quaternion_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a = tuple(float(v) for v in left); b = tuple(float(v) for v in right)
    if len(a) != 4 or len(b) != 4 or not _finite(a + b):
        return float("inf")
    na = math.sqrt(sum(v*v for v in a)); nb = math.sqrt(sum(v*v for v in b))
    if na <= 1e-12 or nb <= 1e-12:
        return float("inf")
    dot = abs(sum(x*y for x, y in zip(a, b)) / (na*nb))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


@dataclass(frozen=True)
class GraspEvidenceConfig:
    open_safe: float = 0.05
    self_contact_floor: float = 0.0095
    minimum_closure: float = 0.004
    corridor_tolerance: float = 0.032
    minimum_table_clearance: float = 0.008
    minimum_object_rise: float = 0.012
    minimum_lift_ratio: float = 0.55
    maximum_translation_drift: float = 0.012
    maximum_rotation_drift: float = 0.25
    force_evidence_threshold: float = 0.2


@dataclass(frozen=True)
class GraspEvidence:
    grasp_candidate: bool
    grasp_verified: bool
    confidence: float
    flags: Mapping[str, bool]
    failure_reason: str


def evaluate_grasp(*, config: GraspEvidenceConfig, actual_jaw: float,
                   object_in_corridor: bool, object_table_clearance: float,
                   object_rise: float, ee_rise: float,
                   relative_translation_drift: float,
                   relative_rotation_drift: float, state_values: Sequence[float],
                   left_force: float = 0.0, right_force: float = 0.0,
                   object_force: float = 0.0) -> GraspEvidence:
    finite = _finite(tuple(state_values) + (
        actual_jaw, object_table_clearance, object_rise, ee_rise,
        relative_translation_drift, relative_rotation_drift,
        left_force, right_force, object_force,
    ))
    flags = {
        "finite": finite,
        "jaw_closed": actual_jaw <= config.open_safe - config.minimum_closure,
        "jaw_safe": actual_jaw >= config.self_contact_floor,
        "object_in_corridor": bool(object_in_corridor),
        "object_off_table": object_table_clearance >= config.minimum_table_clearance,
        "object_rose": object_rise >= config.minimum_object_rise,
        "co_motion": ee_rise > 0 and object_rise >= config.minimum_lift_ratio * ee_rise,
        "translation_stable": relative_translation_drift <= config.maximum_translation_drift,
        "rotation_stable": relative_rotation_drift <= config.maximum_rotation_drift,
        "force_evidence": (
            object_force >= config.force_evidence_threshold
            and (left_force >= config.force_evidence_threshold
                 or right_force >= config.force_evidence_threshold)
        ),
    }
    candidate_keys = ("finite", "jaw_closed", "jaw_safe", "object_in_corridor")
    verified_keys = candidate_keys + (
        "object_off_table", "object_rose", "co_motion",
        "translation_stable", "rotation_stable",
    )
    candidate = all(flags[key] for key in candidate_keys)
    verified = all(flags[key] for key in verified_keys)
    confidence_keys = verified_keys + ("force_evidence",)
    confidence = sum(bool(flags[key]) for key in confidence_keys) / len(confidence_keys)
    failed = next((key for key in verified_keys if not flags[key]), "")
    return GraspEvidence(candidate, verified, confidence, flags,
                         "" if verified else failed)

