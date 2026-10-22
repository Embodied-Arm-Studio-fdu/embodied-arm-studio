"""Static registry for executable simulation tasks and planning metadata."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple
import hashlib
import json

from .artifact_finalizer import DEMO_ARTIFACT_POLICY, TRAINING_ARTIFACT_POLICY
from .domain import (
    ManipulationTrainingTaskConfig, ReachTrainingTaskConfig, TrainingTaskConfig,
)
from .demos import (
    create_circle_tracking_runtime, create_joint_position_runtime,
    create_line_tracking_runtime, create_target_pose_runtime, create_joint_sweep_runtime,
    create_scripted_pick_place_runtime,
    parse_circle_tracking, parse_joint_position, parse_line_tracking,
    parse_target_pose, parse_joint_sweep,
    parse_scripted_pick_place,
)


TRAINING_RUNNER = "training"
DEMO_RUNNER = "demo"

FUTURE_TASK_IDS = (
    "s1_reach",
    "demo_s1_target_pose",
    "demo_s1_line_tracking",
    "demo_s1_circle_tracking",
    "demo_s1_joint_position",
    "demo_s1_joint_sweep",
    "demo_s1_scripted_pick_place",
)

PARAMETER_SCHEMA_VERSION = "task-parameters/v1"


@dataclass(frozen=True)
class LocalizedText:
    zh_cn: str
    en: str

    def to_dict(self) -> Dict[str, str]:
        return {"zh-CN": self.zh_cn, "en": self.en}


@dataclass(frozen=True)
class GoalSuccess:
    availability: str
    source: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "availability": self.availability,
            "source": self.source,
            "description": self.description,
        }


@dataclass(frozen=True)
class ExpectedArtifacts:
    required: Tuple[str, ...]
    conditional: Tuple[str, ...]
    optional: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": list(self.required),
            "conditional": list(self.conditional),
            "optional": list(self.optional),
        }


@dataclass(frozen=True)
class ResourceProfile:
    gpu_required: bool
    duration_class: str
    video_supported: bool
    duration_hint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_required": self.gpu_required,
            "duration_class": self.duration_class,
            "video_supported": self.video_supported,
            "duration_hint": self.duration_hint,
        }


@dataclass(frozen=True)
class Composition:
    workflow_enabled: bool = False
    preserves_world_state: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "workflow_enabled": self.workflow_enabled,
            "preserves_world_state": self.preserves_world_state,
        }


@dataclass(frozen=True)
class Safety:
    parameter_whitelist_only: bool = True
    arbitrary_joint_targets: bool = False
    real_robot_allowed: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "parameter_whitelist_only": self.parameter_whitelist_only,
            "arbitrary_joint_targets": self.arbitrary_joint_targets,
            "real_robot_allowed": self.real_robot_allowed,
        }


@dataclass(frozen=True)
class TaskSemantics:
    """Immutable Brain V1 semantics owned by one production definition."""

    task_id: str
    display_name: LocalizedText
    summary: LocalizedText
    run_kind: str
    environment_type: str
    planning_role: str
    intent_tags: Tuple[str, ...]
    preconditions: Tuple[str, ...]
    expected_metrics: Tuple[str, ...]
    expected_artifacts: ExpectedArtifacts
    goal_success: GoalSuccess
    resource_profile: ResourceProfile
    composition: Composition = Composition()
    safety: Safety = Safety()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "display_names": self.display_name.to_dict(),
            "summary": self.summary.to_dict(),
            "run_kind": self.run_kind,
            "environment_type": self.environment_type,
            "planning_role": self.planning_role,
            "intent_tags": list(self.intent_tags),
            "preconditions": list(self.preconditions),
            "expected_metrics": list(self.expected_metrics),
            "expected_artifacts": self.expected_artifacts.to_dict(),
            "goal_success": self.goal_success.to_dict(),
            "resource_profile": self.resource_profile.to_dict(),
            "composition": self.composition.to_dict(),
            "safety": self.safety.to_dict(),
        }


class UnknownTaskDefinition(ValueError):
    """Raised before enqueue when a task is not executable in this release."""


@dataclass(frozen=True)
class TaskDefinition:
    id: str
    display_name: str
    task_type: str
    robot: str
    supports_training: bool
    supports_video: bool
    runner_kind: str
    config_parser: Callable[[Mapping[str, Any]], Any]
    runtime_factory: Optional[Callable[..., Any]] = None
    runner_factory: Optional[Callable[..., Any]] = None
    artifact_policy: Any = TRAINING_ARTIFACT_POLICY
    catalog_visible: bool = True
    parameters: Tuple[Mapping[str, Any], ...] = ()
    parameter_constraints: Tuple[Mapping[str, Any], ...] = ()
    semantics: Optional[TaskSemantics] = None

    def __post_init__(self) -> None:
        if not self.id or not self.display_name or not self.robot:
            raise ValueError("task definition identity fields are required")
        if self.runner_kind not in (TRAINING_RUNNER, DEMO_RUNNER):
            raise ValueError("runner_kind must be 'training' or 'demo'")
        if self.task_type != self.runner_kind:
            raise ValueError("task_type and runner_kind must match")
        if self.supports_training != (self.runner_kind == TRAINING_RUNNER):
            raise ValueError("supports_training must match runner_kind")
        if not callable(self.config_parser):
            raise TypeError("config_parser must be callable")
        if self.semantics is not None:
            if self.semantics.task_id != self.id:
                raise ValueError("semantics task_id must match task definition")
            if self.semantics.environment_type != self.id:
                raise ValueError("semantics environment_type must match task definition")

    def parse_config(self, values: Mapping[str, Any]):
        if not isinstance(values, Mapping):
            raise ValueError("task config must be an object")
        config = self.config_parser(dict(values))
        if config.environment_type != self.id:
            raise ValueError("environment_type does not match task definition")
        if config.task_type != self.task_type:
            raise ValueError("task_type does not match task definition")
        return config

    def catalog_entry(self) -> Dict[str, Any]:
        schema = self.parameter_schema()
        schema_digest = _sha256_json(schema)
        entry = {
            "id": self.id,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "robot": self.robot,
            "supports_training": self.supports_training,
            "supports_video": self.supports_video,
            "parameters": [dict(parameter) for parameter in self.parameters],
            "parameter_schema": schema,
            "parameter_schema_ref": (
                "registry://task/%s/parameters?digest=%s" % (self.id, schema_digest)
            ),
            "parameter_schema_digest": schema_digest,
        }
        if self.semantics is not None:
            entry.update(self.semantics.to_dict())
        return entry

    def parameter_schema(self) -> Dict[str, Any]:
        return {
            "schema_version": PARAMETER_SCHEMA_VERSION,
            "task_id": self.id,
            "type": "object",
            "additionalProperties": False,
            "parameters": [dict(parameter) for parameter in self.parameters],
            "constraints": [dict(constraint) for constraint in self.parameter_constraints],
            "validation_authority": "TaskDefinition.config_parser",
        }


class TaskRegistry:
    """Small explicit mapping; never stores simulator or GPU runtime state."""

    def __init__(self, definitions: Iterable[TaskDefinition]) -> None:
        self._definitions: Dict[str, TaskDefinition] = {}
        for definition in definitions:
            if definition.id in self._definitions:
                raise ValueError("duplicate task definition: %s" % definition.id)
            self._definitions[definition.id] = definition

    def get(self, task_id: str) -> TaskDefinition:
        try:
            return self._definitions[task_id]
        except KeyError as exc:
            raise UnknownTaskDefinition(
                "unknown or unavailable environment_type: %s" % task_id
            ) from exc

    def parse_config(self, values: Mapping[str, Any]):
        if not isinstance(values, Mapping):
            raise ValueError("task config must be an object")
        environment_type = values.get("environment_type")
        if not isinstance(environment_type, str) or not environment_type:
            raise ValueError("environment_type is required")
        return self.get(environment_type).parse_config(values)

    def definitions(self):
        return tuple(self._definitions[key] for key in sorted(self._definitions))


def _parse_s1_push(values):
    try:
        return TrainingTaskConfig(**dict(values))
    except TypeError as exc:
        raise ValueError("invalid s1_push configuration: %s" % exc) from exc


def _parse_s1_reach(values):
    try:
        return ReachTrainingTaskConfig(**dict(values))
    except TypeError as exc:
        raise ValueError("invalid s1_reach configuration: %s" % exc) from exc


def _parse_s1_manipulation(values):
    try:
        return ManipulationTrainingTaskConfig(**dict(values))
    except TypeError as exc:
        raise ValueError("invalid manipulation configuration: %s" % exc) from exc


def _sha256_json(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _parameter(
    name, kind, default, *, required=False, fixed=False,
    planning_access="user_editable", **constraints
):
    value = {
        "name": name,
        "type": kind,
        "required": required,
        "fixed": fixed,
        "default": default,
        "planning_access": planning_access,
    }
    value.update(constraints)
    return value


COMMON_VIDEO_PARAMETERS = (
    _parameter("video_enabled", "boolean", True, planning_access="model_selectable"),
    _parameter("camera_id", "enum", "overview", fixed=True,
               planning_access="fixed", enum=["overview"]),
)


def _training_parameters(task_id, episode_steps, physics_steps_per_action=8):
    return (
    _parameter("task_type", "string", "training", required=True, fixed=True,
               planning_access="fixed", enum=["training"]),
    _parameter("environment_type", "string", task_id, required=True, fixed=True,
               planning_access="fixed", enum=[task_id]),
    _parameter("seed", "integer", 7, required=True,
               planning_access="model_selectable", minimum=0),
    _parameter("run_mode", "enum", "training", planning_access="backend_resolved",
               enum=["smoke", "training"]),
    _parameter("rollout_steps", "integer", 200, required=True,
               planning_access="backend_resolved", minimum=1),
    _parameter("num_updates", "integer", 100, required=True,
               planning_access="backend_resolved", minimum=1),
    _parameter("checkpoint_every", "integer", 10, required=True,
               planning_access="backend_resolved", minimum=1),
    _parameter("episode_steps", "integer", episode_steps, required=True,
               planning_access="backend_resolved", minimum=1),
    _parameter("physics_steps_per_action", "integer", physics_steps_per_action, required=True,
               planning_access="backend_resolved", minimum=1),
    _parameter("update_epochs", "integer", 1, planning_access="backend_resolved", minimum=1),
    _parameter("batch_size", "integer", 16, planning_access="backend_resolved", minimum=1),
    _parameter("device", "string", "cuda:0", fixed=True,
               planning_access="fixed", enum=["cuda:0"]),
) + COMMON_VIDEO_PARAMETERS + (
    _parameter("video_fps", "number", 15, exclusive_minimum=0),
    _parameter("video_width", "integer", 640, minimum=1),
    _parameter("video_height", "integer", 480, minimum=1),
    )


def _fixed_demo_parameters(task_id):
    return (
        _parameter("task_type", "string", "demo", required=True, fixed=True,
                   planning_access="fixed", enum=["demo"]),
        _parameter("environment_type", "string", task_id, required=True, fixed=True,
                   planning_access="fixed", enum=[task_id]),
    )


DEMO_PARAMETERS = (
    _parameter("duration_seconds", "number", 30, minimum=5, maximum=300),
    _parameter("seed", "integer", 7, planning_access="model_selectable", minimum=0),
    _parameter("physics_steps_per_command", "integer", 1,
               planning_access="backend_resolved", minimum=1, maximum=16),
) + COMMON_VIDEO_PARAMETERS + (
    _parameter("fps", "number", 15, minimum=1, maximum=60),
    _parameter("width", "integer", 640, minimum=160, maximum=1920),
    _parameter("height", "integer", 480, minimum=120, maximum=1080),
)

TRAINING_ARTIFACTS = ExpectedArtifacts(
    required=("checkpoint_latest", "checkpoint_best", "training_config",
              "training_metrics", "stdout_log", "stderr_log", "task_log", "manifest"),
    conditional=("evaluation_summary", "evaluation_episodes"),
    optional=("training_video", "evaluation_video"),
)
DEMO_ARTIFACTS = ExpectedArtifacts(
    required=("demo_config", "demo_metrics", "stdout_log", "stderr_log",
              "task_log", "manifest"),
    conditional=(), optional=("demo_video",),
)
TRAINING_METRICS = (
    "training.update_index", "training.training_steps", "training.average_reward",
    "training.episode_length", "training.success_rate", "training.timeout_rate",
    "training.failure_rate", "training.final_distance", "training.minimum_distance",
    "evaluation.policy_passed", "evaluation.success_rate",
    "evaluation.wilson_95_lower_bound", "evaluation.wilson_95_upper_bound",
)
TRACKING_METRICS = (
    "target_xyz_m", "ee_xyz_m", "tracking_error_m",
    "rms_tracking_error_m", "max_tracking_error_m",
)


def _semantics(
    task_id, zh_name, en_name, zh_summary, en_summary, planning_role,
    intent_tags, preconditions, metrics, artifacts, goal_success, duration_class,
):
    return TaskSemantics(
        task_id=task_id,
        display_name=LocalizedText(zh_name, en_name),
        summary=LocalizedText(zh_summary, en_summary),
        run_kind="train" if planning_role == "training" else "play",
        environment_type=task_id,
        planning_role=planning_role,
        intent_tags=tuple(intent_tags),
        preconditions=tuple(preconditions),
        expected_metrics=tuple(metrics),
        expected_artifacts=artifacts,
        goal_success=goal_success,
        resource_profile=ResourceProfile(
            gpu_required=True, duration_class=duration_class,
            video_supported=True,
            duration_hint="Long-running PPO training" if duration_class == "long"
            else "Short bounded demonstration",
        ),
    )


PUSH_CONSTRAINTS = ({
    "kind": "conditional",
    "if": {"run_mode": "training"},
    "then": {"episode_steps": {"const": 256},
             "physics_steps_per_action": {"const": 8}},
},)
REACH_CONSTRAINTS = ({
    "kind": "conditional",
    "if": {"run_mode": "training"},
    "then": {"episode_steps": {"const": 128},
             "physics_steps_per_action": {"const": 8}},
}, {
    "kind": "cross_field",
    "left": "success_hold_steps", "operator": "less_than", "right": "episode_steps",
})


S1_PUSH_DEFINITION = TaskDefinition(
    id="s1_push",
    display_name="S1 推动任务",
    task_type="training",
    robot="s1",
    supports_training=True,
    supports_video=True,
    runner_kind=TRAINING_RUNNER,
    config_parser=_parse_s1_push,
    artifact_policy=TRAINING_ARTIFACT_POLICY,
    parameters=_training_parameters("s1_push", 256) + (
        _parameter("object_size", "number_triplet", None,
                   planning_access="backend_resolved",
                   exclusive_minimum=0, nullable=True),
        _parameter("object_start", "number_triplet", None,
                   planning_access="backend_resolved", nullable=True),
        _parameter("goal_position", "number_triplet", None,
                   planning_access="backend_resolved", nullable=True),
    ),
    parameter_constraints=PUSH_CONSTRAINTS,
    semantics=_semantics(
        "s1_push", "S1 推动策略训练", "S1 Push Policy Training",
        "训练 S1 将物体推向目标的策略。", "Train an S1 policy to push an object to a goal.",
        "training", ("push_object", "policy_training", "compare_training"),
        ("project_supports_s1_push", "isaac_gym_gpu_available"),
        TRAINING_METRICS + ("training.object_displacement", "evaluation.mean_object_displacement"),
        TRAINING_ARTIFACTS,
        GoalSuccess("formal", "evaluation",
                    "Run completed does not mean policy passed; success comes from deterministic Evaluation."),
        "long",
    ),
)

S1_REACH_DEFINITION = TaskDefinition(
    id="s1_reach",
    display_name="S1 平面到达任务",
    task_type="training",
    robot="s1",
    supports_training=True,
    supports_video=True,
    runner_kind=TRAINING_RUNNER,
    config_parser=_parse_s1_reach,
    artifact_policy=TRAINING_ARTIFACT_POLICY,
    # M2 CPU, workspace, scripted, PPO, production, evaluation and video gates
    # passed on 2026-09-20; this task is now production-visible.
    catalog_visible=True,
    parameters=_training_parameters("s1_reach", 128) + (
        _parameter("action_delta_m", "number", 0.005, minimum=0.001, maximum=0.01),
        _parameter("target_center_xy_m", "number_pair", [0.36, 0.0]),
        _parameter("target_half_extent_xy_m", "number_pair", [0.03, 0.08], exclusive_minimum=0, maximum=0.15),
        _parameter("success_distance_m", "number", 0.02, minimum=0.01, maximum=0.05),
        _parameter("success_hold_steps", "integer", 3, minimum=1),
    ),
    parameter_constraints=REACH_CONSTRAINTS,
    semantics=_semantics(
        "s1_reach", "S1 目标到达训练", "S1 Reach Policy Training",
        "训练 S1 到达平面目标的策略。", "Train an S1 policy to reach planar targets.",
        "training", ("reach_target", "policy_training", "compare_distance"),
        ("project_supports_s1_reach", "isaac_gym_gpu_available"),
        TRAINING_METRICS, TRAINING_ARTIFACTS,
        GoalSuccess("formal", "evaluation",
                    "Run completed does not mean policy passed; success comes from deterministic Evaluation."),
        "long",
    ),
)

MANIPULATION_METRICS = TRAINING_METRICS + (
    "training.candidate_rate", "training.verified_success_rate",
    "training.mean_maximum_cube_rise", "training.mean_hold_duration",
    "training.descended_rate", "training.detached_rate",
    "training.settled_1s_rate", "training.mean_final_xy_error",
)


def _manipulation_parameters(task_id, episode_steps, extras):
    base = []
    for parameter in _training_parameters(task_id, episode_steps, 4):
        if parameter["name"] == "video_width":
            parameter = _parameter(
                "video_width", "enum", 640, fixed=True,
                planning_access="fixed", enum=[640],
            )
        elif parameter["name"] == "video_height":
            parameter = _parameter(
                "video_height", "enum", 480, fixed=True,
                planning_access="fixed", enum=[480],
            )
        base.append(parameter)
    return tuple(base) + tuple(extras)


S1_MANIPULATION_DEFINITIONS = (
    TaskDefinition(
        "s1_grasp", "S1 抓取", "training", "s1", True, True,
        TRAINING_RUNNER, _parse_s1_manipulation,
        artifact_policy=TRAINING_ARTIFACT_POLICY,
        parameters=_manipulation_parameters("s1_grasp", 120, (
            _parameter("randomize_xy_m", "number", 0.0, minimum=0.0, maximum=0.006),
            _parameter("action_delta_m", "number", 0.005, minimum=0.0025, maximum=0.005),
            _parameter("hold_seconds", "enum", 1.0, fixed=True, planning_access="fixed", enum=[1.0]),
            _parameter("release_trigger_threshold", "enum", 0.0, fixed=True, planning_access="fixed", enum=[0.0]),
        )),
        semantics=_semantics(
            "s1_grasp", "S1 抓取训练", "S1 Grasp Policy Training",
            "训练 S1 对齐、下降、闭合夹爪并验证稳定抓取。",
            "Train S1 to align, descend, close the gripper and verify a stable grasp.",
            "training", ("grasp_object", "policy_training", "stable_grasp"),
            ("project_supports_s1_grasp", "isaac_gym_gpu_available"),
            MANIPULATION_METRICS, TRAINING_ARTIFACTS,
            GoalSuccess("formal", "task_metrics", "Success requires the grasp verifier."),
            "long",
        ),
    ),
    TaskDefinition(
        "s1_lift", "S1 抓取抬起", "training", "s1", True, True,
        TRAINING_RUNNER, _parse_s1_manipulation,
        artifact_policy=TRAINING_ARTIFACT_POLICY,
        parameters=_manipulation_parameters("s1_lift", 90, (
            _parameter("randomize_xy_m", "enum", 0.0, fixed=True, planning_access="fixed", enum=[0.0]),
            _parameter("action_delta_m", "number", 0.003, minimum=0.0025, maximum=0.005),
            _parameter("hold_seconds", "enum", 1.0, enum=[1.0, 5.0]),
            _parameter("release_trigger_threshold", "enum", 0.0, fixed=True, planning_access="fixed", enum=[0.0]),
        )),
        semantics=_semantics(
            "s1_lift", "S1 抓取抬起训练", "S1 Grasp-and-Lift Policy Training",
            "从稳定抓取状态抬升方块到 60 mm 目标带并保持。",
            "Lift a stably grasped cube into the 60 mm target band and hold it.",
            "training", ("lift_object", "policy_training", "stable_hold"),
            ("project_supports_s1_lift", "isaac_gym_gpu_available"),
            MANIPULATION_METRICS, TRAINING_ARTIFACTS,
            GoalSuccess("formal", "task_metrics", "Success requires target-band hold and grasp stability."),
            "long",
        ),
    ),
    TaskDefinition(
        "s1_lower_release", "S1 抓取放下", "training", "s1", True, True,
        TRAINING_RUNNER, _parse_s1_manipulation,
        artifact_policy=TRAINING_ARTIFACT_POLICY,
        parameters=_manipulation_parameters("s1_lower_release", 120, (
            _parameter("randomize_xy_m", "enum", 0.0, fixed=True, planning_access="fixed", enum=[0.0]),
            _parameter("action_delta_m", "enum", 0.003, fixed=True, planning_access="fixed", enum=[0.003]),
            _parameter("hold_seconds", "enum", 1.0, fixed=True, planning_access="fixed", enum=[1.0]),
            _parameter("release_trigger_threshold", "number", 0.0, minimum=-0.8, maximum=0.8),
        )),
        semantics=_semantics(
            "s1_lower_release", "S1 抓取放下训练", "S1 Lower-and-Release Policy Training",
            "将已抓取方块下降到目标区域，触发松爪并等待稳定放置。",
            "Lower a grasped cube into the target, release it and verify stable placement.",
            "training", ("lower_object", "release_object", "stable_placement"),
            ("project_supports_s1_lower_release", "isaac_gym_gpu_available"),
            MANIPULATION_METRICS, TRAINING_ARTIFACTS,
            GoalSuccess("formal", "task_metrics", "Success requires legal release, detachment and stable placement."),
            "long",
        ),
    ),
    TaskDefinition(
        "s1_place", "S1 搬运放置", "training", "s1", True, True,
        TRAINING_RUNNER, _parse_s1_manipulation,
        artifact_policy=TRAINING_ARTIFACT_POLICY,
        parameters=_manipulation_parameters("s1_place", 160, (
            _parameter("randomize_xy_m", "enum", 0.0, fixed=True, planning_access="fixed", enum=[0.0]),
            _parameter("action_delta_m", "enum", 0.0045, fixed=True, planning_access="fixed", enum=[0.0045]),
            _parameter("hold_seconds", "enum", 1.0, fixed=True, planning_access="fixed", enum=[1.0]),
            _parameter("release_trigger_threshold", "enum", 0.0, fixed=True, planning_access="fixed", enum=[0.0]),
        )),
        semantics=_semantics(
            "s1_place", "S1 搬运放置训练", "S1 Transport and Place Policy Training",
            "从验证过的抓取抬起状态水平搬运方块到目标区域，并保持稳定。",
            "Transport a verified grasped-and-lifted cube to a validated target and hold it stable.",
            "training", ("transport_object", "place_object", "stable_placement"),
            ("project_supports_s1_place", "validated_target_candidate", "isaac_gym_gpu_available"),
            MANIPULATION_METRICS + ("training.target_reached_rate", "training.hold_1s_rate", "training.failure_reason_histogram"),
            TRAINING_ARTIFACTS,
            GoalSuccess("formal", "task_metrics", "Success requires a validated target, 15 mm XY tolerance and one-second stable hold."),
            "long",
        ),
    ),
)

S1_DEMO_DEFINITIONS = (
    TaskDefinition("demo_s1_target_pose", "S1 目标位姿控制", "demo", "s1", False, True, DEMO_RUNNER, parse_target_pose, create_target_pose_runtime, artifact_policy=DEMO_ARTIFACT_POLICY, parameters=_fixed_demo_parameters("demo_s1_target_pose") + DEMO_PARAMETERS + (_parameter("target_offset_xy_m", "number_pair", [0.08, 0.05], planning_access="model_selectable", minimum=-0.15, maximum=0.15),), semantics=_semantics("demo_s1_target_pose", "S1 目标位姿控制", "S1 Target Pose Control", "移动到有界 XY 目标并报告跟踪误差。", "Move to a bounded XY target and report tracking error.", "atomic_demo", ("target_pose", "bounded_cartesian_motion", "control_demo"), ("project_supports_s1", "isaac_gym_gpu_available"), TRACKING_METRICS + ("final_ee_target_distance_m", "minimum_distance_m"), DEMO_ARTIFACTS, GoalSuccess("metrics_only", "normalized_metrics", "C0 publishes metrics without a formal success verdict."), "short")),
    TaskDefinition("demo_s1_line_tracking", "S1 直线轨迹跟踪", "demo", "s1", False, True, DEMO_RUNNER, parse_line_tracking, create_line_tracking_runtime, artifact_policy=DEMO_ARTIFACT_POLICY, parameters=_fixed_demo_parameters("demo_s1_line_tracking") + DEMO_PARAMETERS + (_parameter("line_length_m", "number", 0.15, planning_access="model_selectable", minimum=0.02, maximum=0.18), _parameter("period_seconds", "number", 6, planning_access="model_selectable", minimum=2, maximum=30), _parameter("direction", "enum", "x", planning_access="model_selectable", enum=["x", "y"])), semantics=_semantics("demo_s1_line_tracking", "S1 直线轨迹跟踪", "S1 Line Tracking", "沿有界直线轨迹运动并报告跟踪误差。", "Track a bounded line and report tracking error.", "atomic_demo", ("line_tracking", "trajectory_demo"), ("project_supports_s1", "isaac_gym_gpu_available"), TRACKING_METRICS, DEMO_ARTIFACTS, GoalSuccess("metrics_only", "normalized_metrics", "C0 publishes metrics without a formal success verdict."), "short")),
    TaskDefinition("demo_s1_circle_tracking", "S1 圆轨迹跟踪", "demo", "s1", False, True, DEMO_RUNNER, parse_circle_tracking, create_circle_tracking_runtime, artifact_policy=DEMO_ARTIFACT_POLICY, parameters=_fixed_demo_parameters("demo_s1_circle_tracking") + DEMO_PARAMETERS + (_parameter("radius_m", "number", 0.05, planning_access="model_selectable", minimum=0.01, maximum=0.08), _parameter("period_seconds", "number", 8, planning_access="model_selectable", minimum=2, maximum=30)), semantics=_semantics("demo_s1_circle_tracking", "S1 圆轨迹跟踪", "S1 Circle Tracking", "沿有界圆轨迹运动并报告跟踪误差。", "Track a bounded circle and report tracking error.", "atomic_demo", ("circle_tracking", "trajectory_demo"), ("project_supports_s1", "isaac_gym_gpu_available"), TRACKING_METRICS, DEMO_ARTIFACTS, GoalSuccess("metrics_only", "normalized_metrics", "C0 publishes metrics without a formal success verdict."), "short")),
    TaskDefinition("demo_s1_joint_position", "S1 关节位置控制", "demo", "s1", False, True, DEMO_RUNNER, parse_joint_position, create_joint_position_runtime, artifact_policy=DEMO_ARTIFACT_POLICY, parameters=_fixed_demo_parameters("demo_s1_joint_position") + DEMO_PARAMETERS + (_parameter("pose", "enum", "ready", planning_access="model_selectable", enum=["home", "ready", "side"]),), semantics=_semantics("demo_s1_joint_position", "S1 安全预设位姿", "S1 Safe Preset Pose", "移动到白名单中的安全预设关节位姿。", "Move to an allowlisted safe preset joint pose.", "atomic_demo", ("preset_pose", "home", "ready", "side"), ("project_supports_s1", "isaac_gym_gpu_available"), ("pose", "ramp_fraction", "final_joint_error_rad", "max_joint_error_rad"), DEMO_ARTIFACTS, GoalSuccess("metrics_only", "normalized_metrics", "C0 publishes metrics without a formal success verdict."), "short")),
    # M4 CPU, six-joint GPU safety, production video and stop gates passed.
    TaskDefinition("demo_s1_joint_sweep", "S1 关节范围扫描", "demo", "s1", False, True, DEMO_RUNNER, parse_joint_sweep, create_joint_sweep_runtime, artifact_policy=DEMO_ARTIFACT_POLICY, catalog_visible=True, parameters=_fixed_demo_parameters("demo_s1_joint_sweep") + DEMO_PARAMETERS + (_parameter("joint_index", "enum", "all_arm", planning_access="model_selectable", enum=["all_arm", 0, 1, 2, 3, 4, 5]), _parameter("speed_scale", "number", 0.2, planning_access="model_selectable", minimum=0.05, maximum=0.5)), semantics=_semantics("demo_s1_joint_sweep", "S1 受限关节扫动", "S1 Restricted Joint Sweep", "在注册表限定的安全范围内执行关节诊断扫动。", "Run a diagnostic joint sweep within registry-defined safe ranges.", "diagnostic", ("joint_diagnostic", "restricted_sweep"), ("project_supports_s1", "isaac_gym_gpu_available"), ("scanned_joint_indices", "active_joint_index", "sweep_phase", "segment_index", "segment_count", "segment_fraction", "per_joint_results", "max_tracking_error_rad", "final_home_error_rad", "sweep_completed", "non_finite_count", "limit_clamp_count", "physical_lower_limits", "physical_upper_limits"), DEMO_ARTIFACTS, GoalSuccess("metrics_only", "normalized_metrics", "C0 publishes metrics without a formal success verdict."), "short")),
    TaskDefinition("demo_s1_scripted_pick_place", "S1 规则抓取放下", "demo", "s1", False, True, DEMO_RUNNER, parse_scripted_pick_place, create_scripted_pick_place_runtime, artifact_policy=DEMO_ARTIFACT_POLICY, catalog_visible=True, parameters=_fixed_demo_parameters("demo_s1_scripted_pick_place") + tuple(_parameter("physics_steps_per_command", "enum", 4, required=True, fixed=True, planning_access="fixed", enum=[4]) if item["name"] == "physics_steps_per_command" else _parameter("width", "enum", 640, fixed=True, planning_access="fixed", enum=[640]) if item["name"] == "width" else _parameter("height", "enum", 480, fixed=True, planning_access="fixed", enum=[480]) if item["name"] == "height" else item for item in DEMO_PARAMETERS), semantics=_semantics("demo_s1_scripted_pick_place", "S1 规则抓取放下", "S1 Scripted Pick and Place", "使用固定规则依次执行对齐、抓取、抬起、放下、松爪和稳定性检查。", "Use a fixed controller to align, grasp, lift, place, release and verify settling.", "atomic_demo", ("scripted_pick_place", "grasp_object", "release_object", "control_demo"), ("project_supports_s1", "isaac_gym_gpu_available"), ("success", "stage_reached", "stage_latency", "grasp_verified_once", "release_height", "settle_linear_velocity", "settle_angular_velocity", "failure_reason", "resource_cleanup"), DEMO_ARTIFACTS, GoalSuccess("metrics_only", "normalized_metrics", "The fixed controller reports its own physical success evidence; it does not train a policy."), "short")),
)

PRODUCTION_TASK_REGISTRY = TaskRegistry(
    (S1_PUSH_DEFINITION, S1_REACH_DEFINITION)
    + S1_MANIPULATION_DEFINITIONS
    + S1_DEMO_DEFINITIONS
)
