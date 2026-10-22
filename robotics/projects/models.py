"""Canonical S1 Push teaching Project domain and frozen configuration."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
from typing import Any, Dict


STARTER_CODE_MAX_BYTES = 512 * 1024
SEED_MIN = 0
SEED_MAX = 2147483647
TASK_TYPE = "training"
ENVIRONMENT_TYPE = "s1_push"
ALGORITHM = "PPO"
SHORT_CONFIG_VERSION = 1
STANDARD_CONFIG_VERSION = 4
SUPPORTED_ENVIRONMENT_TYPES = frozenset({
    "s1_push", "s1_reach", "s1_grasp", "s1_lift", "s1_lower_release", "s1_place",
})


class TrainingProfile(str, Enum):
    SHORT = "short"
    STANDARD = "standard"


@dataclass(frozen=True)
class ProjectPermissions:
    can_view: bool
    can_use: bool
    can_modify: bool


@dataclass(frozen=True)
class TeachingProject:
    project_id: str
    course_id: str
    title: str
    description: str
    instructions: str
    starter_code: str
    starter_code_sha256: str
    training_profile: TrainingProfile
    video_enabled: bool
    parameter_schema: Dict[str, Any]
    evaluation_rubric: Dict[str, Any]
    config_version: int
    enabled: bool
    created_by_user_id: str
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        for name in ("project_id", "course_id", "title", "created_by_user_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("%s must be a non-empty string" % name)
        for name in ("description", "instructions", "starter_code"):
            if not isinstance(getattr(self, name), str):
                raise ValueError("%s must be a string" % name)
        if len(self.starter_code.encode("utf-8")) > STARTER_CODE_MAX_BYTES:
            raise ValueError("starter_code exceeds 512 KiB")
        if self.starter_code_sha256 != starter_code_digest(self.starter_code):
            raise ValueError("starter_code_sha256 does not match starter_code")
        object.__setattr__(self, "training_profile", TrainingProfile(self.training_profile))
        if not isinstance(self.video_enabled, bool) or not isinstance(self.enabled, bool):
            raise ValueError("Project flags must be boolean")
        if isinstance(self.config_version, bool) or not isinstance(self.config_version, int):
            raise ValueError("config_version must be an integer")
        if self.config_version < 1:
            raise ValueError("config_version must be positive")
        if not isinstance(self.parameter_schema, dict) or not isinstance(
            self.evaluation_rubric, dict
        ):
            raise ValueError("Project JSON fields must be objects")
        object.__setattr__(self, "parameter_schema", deepcopy(self.parameter_schema))
        object.__setattr__(self, "evaluation_rubric", deepcopy(self.evaluation_rubric))
        _require_utc(self.created_at, "created_at")
        _require_utc(self.updated_at, "updated_at")

    @property
    def expected_training_steps(self) -> int:
        return int(
            profile_config(self.training_profile, self.config_version)[
                "expected_training_steps"
            ]
        )

    @property
    def environment_type(self) -> str:
        value = self.parameter_schema.get("x-environment-type", ENVIRONMENT_TYPE)
        if value not in SUPPORTED_ENVIRONMENT_TYPES:
            raise ValueError("Project environment_type is unsupported")
        return value


@dataclass(frozen=True)
class TeachingProjectAccess:
    project: TeachingProject
    permissions: ProjectPermissions


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def starter_code_digest(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("starter_code must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("starter_code must be valid UTF-8") from exc
    if len(encoded) > STARTER_CODE_MAX_BYTES:
        raise ValueError("starter_code exceeds 512 KiB")
    return hashlib.sha256(encoded).hexdigest()


def parameter_schema(environment_type: str = ENVIRONMENT_TYPE) -> Dict[str, Any]:
    if environment_type not in SUPPORTED_ENVIRONMENT_TYPES:
        raise ValueError("environment_type is unsupported")
    return {
        "type": "object",
        "x-environment-type": environment_type,
        "additionalProperties": False,
        "required": ["seed"],
        "properties": {
            "seed": {
                "type": "integer",
                "minimum": SEED_MIN,
                "maximum": SEED_MAX,
                "default": 42,
            }
        },
    }


def success_condition(environment_type: str = ENVIRONMENT_TYPE) -> Dict[str, Any]:
    if environment_type == "s1_reach":
        return {
            "metric": "end_effector_to_target_distance",
            "operator": "lte",
            "value": 0.02,
            "unit": "m",
            "hold_steps": 3,
        }
    if environment_type == "s1_grasp":
        return {"metric": "verified_success_rate", "operator": "gte", "value": 0.8}
    if environment_type == "s1_lift":
        return {
            "metric": "cube_rise", "operator": "between",
            "minimum": 0.055, "maximum": 0.065, "unit": "m",
        }
    if environment_type == "s1_lower_release":
        return {"metric": "settled_1s_rate", "operator": "gte", "value": 0.8}
    if environment_type == "s1_place":
        return {"metric": "hold_1s_rate", "operator": "gte", "value": 0.8}
    if environment_type != ENVIRONMENT_TYPE:
        raise ValueError("environment_type is unsupported")
    return {
        "metric": "object_to_goal_distance",
        "operator": "lte",
        "value": 0.05,
        "unit": "m",
    }


def current_config_version(profile: TrainingProfile) -> int:
    profile = TrainingProfile(profile)
    if profile is TrainingProfile.SHORT:
        return SHORT_CONFIG_VERSION
    return STANDARD_CONFIG_VERSION


def profile_config(
    profile: TrainingProfile, config_version: int = SHORT_CONFIG_VERSION
) -> Dict[str, int]:
    profile = TrainingProfile(profile)
    if isinstance(config_version, bool) or not isinstance(config_version, int):
        raise ValueError("config_version must be an integer")
    if config_version < 1:
        raise ValueError("config_version must be positive")
    if profile is TrainingProfile.SHORT:
        return {
            "rollout_steps": 16,
            "num_updates": 2,
            "episode_steps": 32,
            "expected_training_steps": 32,
        }
    if config_version == 1:
        return {
            "rollout_steps": 16,
            "num_updates": 20,
            "episode_steps": 64,
            "expected_training_steps": 320,
        }
    return {
        "rollout_steps": 200,
        "num_updates": 100,
        "episode_steps": 256,
        "expected_training_steps": 20000,
    }


def evaluation_rubric(
    profile: TrainingProfile, config_version: int = SHORT_CONFIG_VERSION,
    environment_type: str = ENVIRONMENT_TYPE,
) -> Dict[str, Any]:
    if environment_type not in SUPPORTED_ENVIRONMENT_TYPES:
        raise ValueError("environment_type is unsupported")
    values = profile_config(profile, config_version)
    return {
        "version": "%s-training-evidence-v1" % environment_type.replace("_", "-"),
        "criteria": [
            {
                "criterion_id": "terminal_completed",
                "evidence_path": "run.status",
                "operator": "eq",
                "target": "completed",
                "weight": 40,
            },
            {
                "criterion_id": "training_steps",
                "evidence_path": "run.training_steps",
                "operator": "gte",
                "target": values["expected_training_steps"],
                "weight": 40,
            },
            {
                "criterion_id": "update_steps",
                "evidence_path": "run.update_steps",
                "operator": "gte",
                "target": values["num_updates"],
                "weight": 20,
            },
        ],
        "grade_bands": [
            {"minimum_score": 100, "grade": "complete"},
            {"minimum_score": 0, "grade": "incomplete"},
        ],
    }


def worker_config(
    profile: TrainingProfile,
    video_enabled: bool,
    seed: int,
    *,
    config_version: int = SHORT_CONFIG_VERSION,
    environment_type: str = ENVIRONMENT_TYPE,
) -> Dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or not SEED_MIN <= seed <= SEED_MAX:
        raise ValueError("seed must be an integer between 0 and 2147483647")
    if not isinstance(video_enabled, bool):
        raise ValueError("video_enabled must be boolean")
    profile = TrainingProfile(profile)
    if environment_type not in SUPPORTED_ENVIRONMENT_TYPES:
        raise ValueError("environment_type is unsupported")
    values = profile_config(profile, config_version)
    config = {
        "task_type": TASK_TYPE,
        "environment_type": environment_type,
        "seed": seed,
        "rollout_steps": values["rollout_steps"],
        "num_updates": values["num_updates"],
        "checkpoint_every": 10 if (
            profile is TrainingProfile.STANDARD and config_version >= 2
        ) else 1,
        "episode_steps": (
            {"s1_grasp": 120, "s1_lift": 90, "s1_lower_release": 120, "s1_place": 160}.get(
                environment_type,
                128 if environment_type == "s1_reach" else values["episode_steps"],
            )
            if profile is TrainingProfile.STANDARD and config_version >= 2
            else values["episode_steps"]
        ),
        "physics_steps_per_action": (
            4 if environment_type in {"s1_grasp", "s1_lift", "s1_lower_release", "s1_place"}
            else 8
        ) if profile is TrainingProfile.STANDARD and config_version >= 2 else 2,
        "update_epochs": 1,
        "batch_size": 16,
        "device": "cuda:0",
        "video_enabled": video_enabled,
        "camera_id": "overview",
        "video_fps": 15.0,
        "video_width": 640,
        "video_height": 480,
    }
    if profile is TrainingProfile.STANDARD and config_version >= 2:
        config["run_mode"] = (
            "smoke" if environment_type in {
                "s1_grasp", "s1_lift", "s1_lower_release", "s1_place",
            } else "training"
        )
    return config


def _require_utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("%s must be timezone-aware" % name)
    if value.utcoffset() != timedelta(0):
        raise ValueError("%s must use UTC" % name)
