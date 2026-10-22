"""Dispatch persisted simulation Runs through the external HTTP boundary."""

from copy import deepcopy
import math
from typing import Any, Dict, Mapping

from robotics.runs import ExecutionMode, Run, RunKind, RunService, RunTaskMismatch
from robotics.projects.models import (
    ALGORITHM,
    ENVIRONMENT_TYPE,
    TASK_TYPE,
    TrainingProfile,
    evaluation_rubric,
    parameter_schema,
    profile_config,
    success_condition,
    worker_config,
)

from .client import (SimulationClient, SimulationRejectedError, SimulationTemporaryError,
                     SimulationResponseError)
from robotics.runs import RunStatus
from .contracts import SimulationTaskCatalog, resolve_catalog_task_parameters
from robotics.scenes import push_geometry_valid


class SimulationDispatchError(RuntimeError):
    pass


class SimulationSnapshotMismatch(SimulationDispatchError):
    pass


_LEGACY_WORKER_CONFIG_KEYS = frozenset(
    {
        "task_type", "environment_type", "seed", "rollout_steps",
        "num_updates", "checkpoint_every", "episode_steps",
        "physics_steps_per_action", "update_epochs", "batch_size", "device",
        "video_enabled", "camera_id", "video_fps", "video_width", "video_height",
    }
)
_WORKER_CONFIG_KEYS = _LEGACY_WORKER_CONFIG_KEYS | {"run_mode"}

_TEACHING_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version", "project_id", "course_id", "class_id", "title",
        "description", "instructions", "task_type", "environment_type",
        "algorithm", "success_condition", "training_profile",
        "expected_training_steps", "video_enabled", "parameter_schema",
        "resolved_parameters", "evaluation_rubric", "config_version",
        "project_enabled_at_creation", "starter_code_sha256",
        "training_config",
    }
)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _scene_push_geometry(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    scene = snapshot.get("scene")
    if scene is None:
        return {}
    try:
        document = scene["document"]
        robot = document["robot"]
        objects = document["objects"]
        if (
            document["schema_version"] != "1.0"
            or document["capability"] != "simulation"
            or robot["id"] != "yhrg-s1-v2"
            or robot["dof"] != 6
            or not isinstance(objects, list)
        ):
            raise ValueError("unsupported scene identity")
        push_objects = [
            item for item in objects
            if item.get("type") == "box" and item.get("static") is False
        ]
        targets = [item for item in objects if item.get("type") == "target"]
        if len(push_objects) != 1 or len(targets) != 1:
            raise ValueError("scene requires one push object and one target")
        push_object = push_objects[0]
        target = targets[0]
        if (not push_geometry_valid(push_object, target)
                or any(abs(value) > 1e-6 for value in push_object.get("rotation", [0, 0, 0]))
                or any(item.get("type") in {"wall", "cylinder", "sphere"} for item in objects)):
            raise ValueError("scene does not pass Push geometry preflight")
        object_position = tuple(float(value) for value in push_object["position"])
        object_scale = tuple(float(value) for value in push_object["scale"])
        target_position = tuple(float(value) for value in target["position"])
        if (
            len(object_position) != 3
            or len(object_scale) != 3
            or len(target_position) != 3
            or not all(math.isfinite(value) for value in object_position + object_scale + target_position)
            or not all(value > 0 for value in object_scale)
        ):
            raise ValueError("invalid scene geometry")
    except (KeyError, TypeError, ValueError) as exc:
        raise SimulationSnapshotMismatch(
            "Run frozen Scene cannot configure the S1 Push Worker"
        ) from exc

    table_top = 0.375
    object_size = [object_scale[0], object_scale[2], object_scale[1]]
    object_start = [
        object_position[0], object_position[2], table_top + object_position[1]
    ]
    goal_position = [target_position[0], target_position[2], object_start[2]]
    return {
        "object_size": object_size,
        "object_start": object_start,
        "goal_position": goal_position,
    }


def build_simulation_config(run: Run) -> Dict[str, Any]:
    """Map immutable Run facts to the exact config keys consumed by the Worker."""

    if not isinstance(run, Run):
        raise TypeError("run must be a canonical Run")
    if run.execution_mode is not ExecutionMode.SIMULATION:
        raise SimulationSnapshotMismatch("Run is not a simulation teaching Run")
    if run.class_id is None:
        raise SimulationSnapshotMismatch("Run teaching identity is incomplete")
    snapshot = deepcopy(run.project_config_snapshot)
    parameters = deepcopy(run.parameters)
    if "catalog_task" in snapshot or "task_config" in snapshot:
        return _build_catalog_config(run, snapshot, parameters)
    if run.run_kind is not RunKind.TRAIN:
        raise SimulationSnapshotMismatch("Legacy teaching Run must be training")
    if set(parameters) != {"seed"}:
        raise SimulationSnapshotMismatch("Run parameters are not canonical")
    seed = parameters.get("seed")
    try:
        if not _TEACHING_SNAPSHOT_KEYS.issubset(snapshot):
            raise ValueError("missing teaching snapshot facts")
        profile = TrainingProfile(snapshot["training_profile"])
        config_version = snapshot["config_version"]
        expected = worker_config(
            profile,
            snapshot["video_enabled"],
            seed,
            config_version=config_version,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SimulationSnapshotMismatch(
            "Run frozen training configuration is invalid"
        ) from exc
    config = snapshot.get("training_config")
    expected_keys = (
        _WORKER_CONFIG_KEYS if "run_mode" in expected else _LEGACY_WORKER_CONFIG_KEYS
    )
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise SimulationSnapshotMismatch(
            "Run frozen training configuration keys are invalid"
        )
    identity = {
        "project_id": run.project_id,
        "class_id": run.class_id,
        "resolved_parameters": parameters,
        "task_type": TASK_TYPE,
        "environment_type": ENVIRONMENT_TYPE,
        "algorithm": ALGORITHM,
        "schema_version": "1.0",
        "success_condition": success_condition(),
        "parameter_schema": parameter_schema(),
        "evaluation_rubric": evaluation_rubric(profile, config_version),
        "expected_training_steps": profile_config(profile, config_version)[
            "expected_training_steps"
        ],
        "project_enabled_at_creation": True,
    }
    if any(snapshot.get(name) != value for name, value in identity.items()):
        raise SimulationSnapshotMismatch("Run frozen snapshot identity is invalid")
    if (
        not isinstance(snapshot.get("course_id"), str)
        or not snapshot["course_id"].strip()
        or not isinstance(snapshot.get("title"), str)
        or not snapshot["title"].strip()
        or not isinstance(snapshot.get("description"), str)
        or not isinstance(snapshot.get("instructions"), str)
        or isinstance(snapshot.get("config_version"), bool)
        or not isinstance(snapshot.get("config_version"), int)
        or snapshot["config_version"] < 1
        or not _valid_digest(snapshot.get("starter_code_sha256"))
        or ("enabled" in snapshot and snapshot["enabled"] is not True)
    ):
        raise SimulationSnapshotMismatch("Run frozen snapshot facts are invalid")
    if config != expected:
        raise SimulationSnapshotMismatch(
            "Run frozen training configuration was altered"
        )
    dispatched = deepcopy(config)
    dispatched.update(_scene_push_geometry(snapshot))
    return dispatched


def _build_catalog_config(
    run: Run, snapshot: Mapping[str, Any], parameters: Mapping[str, Any]
) -> Dict[str, Any]:
    expected_type = "training" if run.run_kind is RunKind.TRAIN else "demo"
    common_identity = {
        "schema_version": "1.0",
        "project_id": run.project_id,
        "class_id": run.class_id,
        "task_type": expected_type,
        "resolved_parameters": parameters,
        "project_enabled_at_creation": True,
    }
    if any(snapshot.get(name) != value for name, value in common_identity.items()):
        raise SimulationSnapshotMismatch("Run catalog snapshot identity is invalid")
    if (
        not isinstance(snapshot.get("course_id"), str)
        or not snapshot["course_id"].strip()
        or not isinstance(snapshot.get("title"), str)
        or not snapshot["title"].strip()
        or not isinstance(snapshot.get("description"), str)
        or not isinstance(snapshot.get("instructions"), str)
        or isinstance(snapshot.get("config_version"), bool)
        or not isinstance(snapshot.get("config_version"), int)
        or snapshot["config_version"] < 1
        or not _valid_digest(snapshot.get("starter_code_sha256"))
    ):
        raise SimulationSnapshotMismatch("Run catalog snapshot facts are invalid")
    try:
        catalog = SimulationTaskCatalog.from_dict(
            {"tasks": [snapshot["catalog_task"]]}
        )
        task, resolved = resolve_catalog_task_parameters(
            catalog,
            snapshot["environment_type"],
            expected_type,
            parameters,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SimulationSnapshotMismatch(
            "Run frozen catalog task or parameters are invalid"
        ) from exc
    if resolved != parameters:
        raise SimulationSnapshotMismatch("Run resolved parameters were altered")
    expected_config = {
        "task_type": expected_type,
        "environment_type": task.id,
        **deepcopy(resolved),
    }
    if run.run_kind is RunKind.TRAIN:
        expected_config["device"] = "cuda:0"
        if (
            snapshot.get("algorithm") != "PPO"
            or snapshot.get("expected_training_steps")
            != resolved.get("rollout_steps", 0) * resolved.get("num_updates", 0)
            or not isinstance(snapshot.get("evaluation_rubric"), dict)
        ):
            raise SimulationSnapshotMismatch(
                "Run frozen training contract is invalid"
            )
    else:
        forbidden = {
            "algorithm", "training_profile", "expected_training_steps",
            "evaluation_rubric", "training_config", "scene",
        }
        if forbidden & set(snapshot):
            raise SimulationSnapshotMismatch(
                "Demo snapshot contains training-only configuration"
            )
    if snapshot.get("task_config") != expected_config:
        raise SimulationSnapshotMismatch("Run frozen task config was altered")
    if resolved.get("video_enabled") is True and not task.supports_video:
        raise SimulationSnapshotMismatch("Catalog task does not support video")
    dispatched = deepcopy(expected_config)
    if run.run_kind is RunKind.TRAIN and task.id == "s1_push":
        dispatched.update(_scene_push_geometry(snapshot))
    if snapshot.get("reward_extension") is not None:
        from simulation.src.simulation_core.reward_extension import validate_extension
        try:
            validate_extension(task.id, snapshot["reward_extension"])
        except ValueError as exc:
            raise SimulationSnapshotMismatch(str(exc)) from exc
        if run.run_kind is not RunKind.TRAIN:
            raise SimulationSnapshotMismatch("Demo Runs cannot use reward extensions")
        dispatched["reward_extension"] = deepcopy(snapshot["reward_extension"])
    return dispatched


class SimulationDispatchService:
    def __init__(
        self, run_service: RunService, client: SimulationClient,
        dispatch_authorizer: Any = None,
    ) -> None:
        self._run_service = run_service
        self._client = client
        self._dispatch_authorizer = dispatch_authorizer

    def inspect_stale_dispatch(self, run_id: str) -> Run:
        """Review old unbound records without starting a new GPU experiment."""
        run = self._run_service.get_run(run_id)
        if run.status != RunStatus.QUEUED or run.simulation_task_id is not None:
            return run
        try:
            build_simulation_config(run)
        except (SimulationSnapshotMismatch, ValueError):
            return self._run_service.record_dispatch_error(
                run_id, "DISPATCH_INVALID_CONFIG",
                "实验配置未通过派发校验，未启动仿真。请检查配置后重新创建实验。",
                definitive=True,
            )
        return self._run_service.record_dispatch_error(
            run_id, "DISPATCH_UNCONFIRMED",
            "实验长时间未绑定执行任务，历史派发结果无法确认。可取消排队后删除或重新创建。",
            definitive=False,
        )

    def dispatch(self, run_id: str) -> Run:
        run = self._run_service.get_run(run_id)
        if run.execution_mode != ExecutionMode.SIMULATION:
            raise RunTaskMismatch("Only simulation Runs can be dispatched")
        if run.simulation_task_id is not None or run.status != RunStatus.QUEUED:
            return run
        authorize = getattr(self._dispatch_authorizer, "authorize_dispatch", None)
        if callable(authorize):
            trusted_identity = authorize(run)
            if trusted_identity is not None:
                if not isinstance(trusted_identity, Mapping) or any(
                    run.project_config_snapshot.get(name) != value
                    for name, value in trusted_identity.items()
                ):
                    raise SimulationSnapshotMismatch(
                        "Run frozen snapshot teaching identity is invalid"
                    )
        # Validate before the request: these failures prove no task was created.
        try:
            config = build_simulation_config(run)
        except (SimulationSnapshotMismatch, ValueError) as exc:
            self._run_service.record_dispatch_error(
                run_id, "DISPATCH_INVALID_CONFIG",
                "实验配置未通过派发校验，未启动仿真。请检查配置后重新创建实验。",
                definitive=True,
            )
            if isinstance(exc, SimulationSnapshotMismatch):
                raise
            raise SimulationSnapshotMismatch("Run dispatch configuration is invalid") from exc
        try:
            task = self._client.create_task(run.run_id, run.project_id, config)
        except SimulationRejectedError as exc:
            # These documented 422 codes are returned before task persistence.
            definitive = exc.status_code == 422 and exc.code in {
                "invalid_request", "invalid_task_config",
            }
            self._run_service.record_dispatch_error(
                run_id,
                "DISPATCH_REJECTED" if definitive else "DISPATCH_UNCONFIRMED",
                "仿真服务拒绝了实验配置，任务未启动。请检查配置后重新创建实验。"
                if definitive else
                "派发请求未被确认，请稍后重试或取消排队。不能据此判定训练已失败。",
                definitive=definitive,
            )
            raise
        except (SimulationTemporaryError, SimulationResponseError):
            self._run_service.record_dispatch_error(
                run_id, "DISPATCH_UNCONFIRMED",
                "派发结果暂未确认，任务可能已被接收。请稍后重试或取消排队。",
                definitive=False,
            )
            raise
        if task.run_id != run.run_id or task.project_id != run.project_id:
            raise SimulationSnapshotMismatch(
                "Simulation task snapshot does not match the persisted Run"
            )
        return self._run_service.bind_simulation_task(run.run_id, task.task_id)

