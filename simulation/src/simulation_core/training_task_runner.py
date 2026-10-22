"""Process-local execution boundary connecting a task snapshot to Push PPO."""

from dataclasses import dataclass, field, replace
import inspect
from pathlib import Path
import random
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from .artifact_finalizer import ArtifactPaths
from .domain import SimulationTask, TaskStatus, TrainingTaskConfig, TRAINING_RUN_MODE


P4_1_PLACE_TRANSFER_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/experimental/p4_1_s1_place/train_20k/checkpoints/model_latest.pt"
)
P3_1_LIFT_TRANSFER_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/experimental/p3_1_s1_lift/train_20k/checkpoints/model_best.pt"
)
P3_1_LIFT_TRANSFER_SHA256 = (
    "3a564c8effb7b06cad2f9e624e57732ba209198c70267569d26a40def180727b"
)
P2_R8_GRASP_TRANSFER_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "artifacts/experimental/p2_r6_s1_grasp/random_xy/checkpoints/model_20000.pt"
)
P2_R8_GRASP_TRANSFER_SHA256 = (
    "3f7ce0e141c16bde00646ced6482f23cbae1562e6c074001b18239c2b8d01309"
)


@dataclass(frozen=True)
class TrainingTaskSpec:
    """Serializable input used to rebuild training inside a child process."""

    task_id: str
    run_id: str
    project_id: str
    config: Any
    created_at: str = ""

    @classmethod
    def from_task(cls, task: SimulationTask) -> "TrainingTaskSpec":
        from .task_registry import PRODUCTION_TASK_REGISTRY
        return cls(
            task_id=task.task_id,
            run_id=task.run_id,
            project_id=task.project_id,
            config=PRODUCTION_TASK_REGISTRY.parse_config(task.config.values),
            created_at=task.created_at.isoformat(),
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainingTaskSpec":
        from .task_registry import PRODUCTION_TASK_REGISTRY
        return cls(
            task_id=str(values["task_id"]),
            run_id=str(values["run_id"]),
            project_id=str(values["project_id"]),
            config=PRODUCTION_TASK_REGISTRY.parse_config(values["config"]),
            created_at=str(values.get("created_at", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "project_id": self.project_id,
            "config": self.config.to_dict(),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TrainingTaskResult:
    """Worker-safe training result containing no simulator or tensor objects."""

    task_id: str
    run_id: str
    status: TaskStatus
    training_steps: int
    update_steps: int
    metrics_path: str
    checkpoint_path: str
    final_metrics_summary: Mapping[str, Any]
    evaluation: Optional[Mapping[str, Any]] = None
    artifact_root: str = ""
    manifest_path: str = ""
    best_checkpoint_path: str = ""
    log_paths: Mapping[str, str] = field(default_factory=dict)
    error: Optional[Mapping[str, Any]] = None
    artifacts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    started_at: str = ""
    ended_at: str = ""

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainingTaskResult":
        return cls(
            task_id=str(values["task_id"]),
            run_id=str(values["run_id"]),
            status=TaskStatus(values["status"]),
            training_steps=int(values["training_steps"]),
            update_steps=int(values["update_steps"]),
            metrics_path=str(values["metrics_path"]),
            checkpoint_path=str(values["checkpoint_path"]),
            final_metrics_summary=dict(values["final_metrics_summary"]),
            evaluation=(
                dict(values["evaluation"])
                if values.get("evaluation") is not None
                else None
            ),
            artifact_root=str(values.get("artifact_root", "")),
            manifest_path=str(values.get("manifest_path", "")),
            best_checkpoint_path=str(values.get("best_checkpoint_path", "")),
            log_paths=dict(values.get("log_paths", {})),
            error=(
                dict(values["error"])
                if values.get("error") is not None
                else None
            ),
            artifacts=tuple(dict(item) for item in values.get("artifacts", ())),
            started_at=str(values.get("started_at", "")),
            ended_at=str(values.get("ended_at", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "training_steps": self.training_steps,
            "update_steps": self.update_steps,
            "metrics_path": self.metrics_path,
            "checkpoint_path": self.checkpoint_path,
            "final_metrics_summary": dict(self.final_metrics_summary),
            "evaluation": (
                dict(self.evaluation) if self.evaluation is not None else None
            ),
            "artifact_root": self.artifact_root,
            "manifest_path": self.manifest_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "log_paths": dict(self.log_paths),
            "error": dict(self.error) if self.error is not None else None,
            "artifacts": [dict(item) for item in self.artifacts],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    def with_artifacts(self, summary, error=None) -> "TrainingTaskResult":
        """Attach a pure-data ArtifactSummary without runtime objects."""
        return replace(
            self,
            status=summary.final_status,
            artifact_root=summary.artifact_root,
            manifest_path=summary.manifest_path,
            metrics_path=summary.metrics_path,
            checkpoint_path=summary.latest_checkpoint_path,
            best_checkpoint_path=summary.best_checkpoint_path,
            log_paths=dict(summary.log_paths),
            error=dict(error) if error is not None else None,
            artifacts=tuple(dict(item) for item in summary.artifacts),
            started_at=summary.started_at,
            ended_at=summary.ended_at,
        )


class TrainingTaskRunner:
    """Execute training and own process-local environment cleanup.

    Task status is deliberately not persisted here. A parent-process Worker owns
    the authoritative SimulationService and receives progress through the callback.
    """

    def __init__(
        self,
        artifacts_root: Path,
        environment_factory: Optional[Callable[[TrainingTaskConfig], Any]] = None,
        trainer_factory: Optional[Callable[[Any, TrainingTaskConfig], Any]] = None,
        progress_callback: Optional[Callable[[TaskStatus], None]] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
        metric_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
        stream_gateway=None,
        stream_lifecycle_factory=None,
        recorder_factory=None,
        failure_injector=None,
        evaluation_launcher=None,
    ) -> None:
        self._artifacts_root = Path(artifacts_root)
        self._environment_factory = environment_factory or self._create_environment
        self._trainer_factory = trainer_factory or self._create_trainer
        self._progress_callback = progress_callback or (lambda status: None)
        self._cancel_requested = cancel_requested or (lambda: False)
        self._metric_callback = metric_callback or (lambda record: None)
        self._stream_gateway = stream_gateway
        self._stream_lifecycle_factory = stream_lifecycle_factory
        self._recorder_factory = recorder_factory
        self._failure_injector = failure_injector
        self._evaluation_launcher = evaluation_launcher
        self._trainer = None

    @property
    def training_steps(self) -> int:
        return int(self._trainer.training_steps) if self._trainer is not None else 0

    @property
    def update_steps(self) -> int:
        return int(self._trainer.update_steps) if self._trainer is not None else 0

    def run(self, task: TrainingTaskSpec) -> TrainingTaskResult:
        config = task.config
        environment = None
        trainer = None
        stream_lifecycle = None
        stream_sessions = ()
        terminal_status = TaskStatus.FAILED
        evaluation = None
        self._trainer = None
        try:
            environment_parameters = inspect.signature(self._environment_factory).parameters
            environment = (
                self._environment_factory(config, task)
                if len(environment_parameters) >= 2
                else self._environment_factory(config)
            )
            self._seed(config.seed)
            environment.initialize()
            trainer = self._trainer_factory(environment, config)
            self._trainer = trainer

            self._progress_callback(TaskStatus.RUNNING)
            if config.video_enabled:
                try:
                    stream_lifecycle = self._create_stream_lifecycle(environment, task)
                    stream_sessions = stream_lifecycle.on_status(
                        task.run_id, task.task_id, TaskStatus.RUNNING
                    )
                    self._publish_streams(stream_sessions)
                except Exception:
                    # Configuration/encoder/publisher failures are video-only.
                    stream_lifecycle = None
                    stream_sessions = ()
            if self._failure_injector is not None:
                self._failure_injector()
            try:
                training_config = self._ppo_training_config(task, config)
                parameters = inspect.signature(trainer.run_training).parameters
                if "metric_callback" in parameters:
                    records = trainer.run_training(
                        training_config,
                        cancel_requested=self._cancel_requested,
                        metric_callback=self._metric_callback,
                    )
                else:
                    records = trainer.run_training(
                        training_config,
                        cancel_requested=self._cancel_requested,
                    )
                    for record in records:
                        self._metric_callback(dict(record))
            except self._training_cancelled_type() as cancelled:
                terminal_status = TaskStatus.STOPPED
                return self._build_result(
                    task,
                    trainer,
                    TaskStatus.STOPPED,
                    cancelled.records,
                    require_artifacts=False,
                )
            if self._cancel_requested():
                terminal_status = TaskStatus.STOPPED
                return self._build_result(
                    task, trainer, TaskStatus.STOPPED, records,
                    require_artifacts=False,
                )
            self._progress_callback(TaskStatus.FINALIZING)
            terminal_status = TaskStatus.COMPLETED
            stream_lifecycle, environment = self._close_training_resources(
                stream_lifecycle, environment, task, terminal_status
            )
            if (
                config.run_mode == TRAINING_RUN_MODE
                or config.environment_type in {
                    "s1_grasp", "s1_lift", "s1_lower_release",
                }
            ):
                try:
                    summary = self._run_final_evaluation(task)
                    from .evaluation_subprocess import evaluation_contract
                    evaluation = evaluation_contract(summary)
                except self._training_cancelled_type() as cancelled:
                    terminal_status = TaskStatus.STOPPED
                    return self._build_result(
                        task,
                        trainer,
                        TaskStatus.STOPPED,
                        records or cancelled.records,
                        require_artifacts=False,
                    )
                except Exception as exc:
                    # Evaluation is evidence about policy quality, not training
                    # completion. Infrastructure failure must not rewrite the
                    # already completed training outcome.
                    evaluation = {
                        "status": "unavailable",
                        "deterministic": True,
                        "policy_passed": False,
                        "failure_reason": type(exc).__name__,
                    }
            return self._build_result(
                task,
                trainer,
                TaskStatus.COMPLETED,
                records,
                require_artifacts=True,
                evaluation=evaluation,
            )
        finally:
            self._close_training_resources(
                stream_lifecycle, environment, task, terminal_status
            )

    def _close_training_resources(self, stream_lifecycle, environment, task, status):
        if stream_lifecycle is not None:
            try:
                stream_sessions = stream_lifecycle.on_status(
                    task.run_id, task.task_id, status
                )
                self._publish_streams(stream_sessions)
            except Exception:
                # Video cleanup/reporting must never mask training outcome.
                pass
            stream_lifecycle = None
        if environment is not None:
            environment.close()
            environment = None
        return stream_lifecycle, environment

    def _run_final_evaluation(self, task) -> Mapping[str, Any]:
        if self._evaluation_launcher is None:
            from .evaluation_subprocess import ProductionEvaluationLauncher

            launcher = ProductionEvaluationLauncher(
                self._artifacts_root, task_id=task.config.environment_type
            )
        else:
            launcher = self._evaluation_launcher
        return launcher.run(
            task.project_id,
            task.run_id,
            cancel_requested=self._cancel_requested,
        )

    def _publish_streams(self, sessions):
        if self._stream_gateway is None: return
        for session in sessions: self._stream_gateway.publish(session)

    def _create_stream_lifecycle(self, environment, task):
        if self._stream_lifecycle_factory is not None:
            return self._stream_lifecycle_factory(environment, task)
        from .video_runtime import TaskVideoRuntimeFactory

        return TaskVideoRuntimeFactory(
            self._artifacts_root,
            stream_gateway=self._stream_gateway,
            recorder_factory=self._recorder_factory,
        ).create(environment, task)

    def _build_result(
        self, task, trainer, status, records, require_artifacts, evaluation=None
    ) -> TrainingTaskResult:
        run_root = ArtifactPaths(self._artifacts_root).run_root(
            task.project_id, task.run_id
        )
        metrics_path = run_root / "metrics" / "training.jsonl"
        checkpoint_path = run_root / "checkpoints" / "model_latest.pt"
        if require_artifacts and (
            not metrics_path.is_file() or not checkpoint_path.is_file()
        ):
            raise RuntimeError("training completed without required artifacts")
        summary: Dict[str, Any] = dict(records[-1]) if records else {}
        return TrainingTaskResult(
            task_id=task.task_id,
            run_id=task.run_id,
            status=status,
            training_steps=int(trainer.training_steps),
            update_steps=int(trainer.update_steps),
            metrics_path=str(metrics_path) if metrics_path.is_file() else "",
            checkpoint_path=str(checkpoint_path) if checkpoint_path.is_file() else "",
            final_metrics_summary=summary,
            evaluation=(dict(evaluation) if evaluation is not None else None),
        )

    @staticmethod
    def _training_cancelled_type():
        from .ppo import TrainingCancelled

        return TrainingCancelled

    def _ppo_training_config(self, task, config):
        from .ppo import PPOTrainingConfig

        return PPOTrainingConfig(
            rollout_steps=config.rollout_steps,
            num_updates=config.num_updates,
            project_id=task.project_id,
            run_id=task.run_id,
            artifacts_root=self._artifacts_root,
            checkpoint_every=config.checkpoint_every,
        )

    @staticmethod
    def _create_environment(config, task=None):
        # Isaac Gym Preview 4 must be imported before torch, and this method is
        # called only from the isolated training process in production.
        from isaacgym import gymapi as _gymapi  # noqa: F401
        from .isaac_gym_camera import IsaacGymCameraConfig
        from .s1_isaac_gym_adapter import S1IsaacGymConfig
        from .video_frame_pipeline import DropPolicy, FramePipeline, PixelFormat

        if config.environment_type in {"s1_grasp", "s1_lift", "s1_lower_release", "s1_place"}:
            pipeline = None
            frame_callback = None
            if config.video_enabled:
                pipeline = FramePipeline(
                    task.run_id, task.task_id, PixelFormat.RGB24,
                    max_frames=3, drop_policy=DropPolicy.DROP_OLDEST,
                    target_fps=config.video_fps,
                )
                frame_callback = pipeline.push_camera_frame
            if config.environment_type == "s1_place":
                from .experimental.manipulation.place_env import PlaceEnvConfig, PlaceTransportEnvironment
                environment = PlaceTransportEnvironment(
                    PlaceEnvConfig(
                        physics_steps_per_action=config.physics_steps_per_action,
                        max_episode_seconds=8.0,
                        hold_seconds=config.hold_seconds,
                        xy_delta_scale=0.0045,
                        reward_extension=config.reward_extension,
                    ),
                    with_camera=config.video_enabled,
                    frame_callback=frame_callback,
                )
            elif config.environment_type == "s1_grasp":
                from .experimental.manipulation.grasp_env import (
                    GraspEnvConfig, GraspPPOEnvironment,
                )
                environment = GraspPPOEnvironment(
                    GraspEnvConfig(
                        physics_steps_per_action=config.physics_steps_per_action,
                        learning_horizon=config.episode_steps,
                        xyz_delta_scale=config.action_delta_m,
                        randomize_xy=config.randomize_xy_m,
                        random_seed=config.seed,
                    ),
                    with_camera=config.video_enabled,
                    frame_callback=frame_callback,
                )
            elif config.environment_type == "s1_lift":
                from .experimental.manipulation.lift_env import (
                    LiftEnvConfig, LiftPPOEnvironment,
                )
                horizon = 9.0 if config.hold_seconds == 5.0 else 6.0
                environment = LiftPPOEnvironment(
                    LiftEnvConfig(
                        physics_steps_per_action=config.physics_steps_per_action,
                        max_episode_seconds=horizon,
                        dz_scale=config.action_delta_m,
                        hold_seconds=config.hold_seconds,
                    ),
                    with_camera=config.video_enabled,
                    frame_callback=frame_callback,
                )
            else:
                from .experimental.manipulation.lower_release_env import (
                    LowerReleaseEnvConfig, LowerReleaseEnvironment,
                )
                environment = LowerReleaseEnvironment(
                    LowerReleaseEnvConfig(
                        physics_steps_per_action=config.physics_steps_per_action,
                        dz_scale=config.action_delta_m,
                        release_trigger_threshold=config.release_trigger_threshold,
                    ),
                    with_camera=config.video_enabled,
                    frame_callback=frame_callback,
                )
            environment.frame_pipeline = pipeline
            return environment

        if config.environment_type == "s1_reach":
            from .reach_adapter import IsaacGymReachBackend, ReachAdapter, ReachConfig
            reach_config = ReachConfig(
                max_steps=config.episode_steps,
                physics_steps_per_action=config.physics_steps_per_action,
                seed=config.seed,
                action_delta_m=config.action_delta_m,
                target_center_xy_m=config.target_center_xy_m,
                target_half_extent_xy_m=config.target_half_extent_xy_m,
                success_distance_m=config.success_distance_m,
                success_hold_steps=config.success_hold_steps,
                reward_extension=config.reward_extension,
            )
            camera = pipeline = None
            s1_config = None
            if config.video_enabled:
                pipeline = FramePipeline(
                    task.run_id, task.task_id, PixelFormat.RGB24,
                    max_frames=3, drop_policy=DropPolicy.DROP_OLDEST,
                    target_fps=config.video_fps,
                )
                camera = IsaacGymCameraConfig(
                    camera_id=config.camera_id, width=config.video_width,
                    height=config.video_height,
                )
                s1_config = S1IsaacGymConfig(graphics_device_id=0, headless=True)
            backend = IsaacGymReachBackend(
                reach_config, s1_config=s1_config, camera_config=camera,
                frame_pipeline=pipeline,
            )
            return ReachAdapter(reach_config, backend=backend)

        from .push_adapter import IsaacGymPushBackend, PushAdapter, PushConfig

        scene_geometry = {
            name: getattr(config, name)
            for name in ("object_size", "object_start", "goal_position")
            if getattr(config, name) is not None
        }
        push_config = PushConfig(
            max_steps=config.episode_steps,
            physics_steps_per_action=config.physics_steps_per_action,
            reward_extension=config.reward_extension,
            **scene_geometry,
        )
        camera = pipeline = None
        s1_config = None
        if config.video_enabled:
            pipeline = FramePipeline(
                task.run_id, task.task_id, PixelFormat.RGB24,
                max_frames=3, drop_policy=DropPolicy.DROP_OLDEST,
                target_fps=config.video_fps,
            )
            camera = IsaacGymCameraConfig(
                camera_id=config.camera_id, width=config.video_width,
                height=config.video_height,
            )
            s1_config = S1IsaacGymConfig(graphics_device_id=0, headless=True)
        backend = IsaacGymPushBackend(
            push_config, s1_config=s1_config, camera_config=camera,
            frame_pipeline=pipeline,
        )
        return PushAdapter(push_config, backend=backend)

    @staticmethod
    def _create_trainer(environment, config):
        from .ppo import PPOConfig, PPOTrainer

        ppo_config = PPOConfig(
            rollout_steps=config.rollout_steps,
            update_epochs=config.update_epochs,
            batch_size=config.batch_size,
        )
        if config.environment_type == "s1_place":
            from .experimental.manipulation.place_ppo import ExperimentalPlacePPOTrainer
            return ExperimentalPlacePPOTrainer(
                environment, config=ppo_config, device=config.device,
            )
        if config.environment_type == "s1_grasp":
            from .experimental.manipulation.grasp_ppo import ExperimentalGraspPPOTrainer
            trainer = ExperimentalGraspPPOTrainer(
                environment, config=ppo_config, device=config.device,
            )
            if not P2_R8_GRASP_TRANSFER_CHECKPOINT.is_file():
                raise RuntimeError("P2-R8 Grasp transfer checkpoint is missing")
            trainer.initialize_from_p2_r8(
                P2_R8_GRASP_TRANSFER_CHECKPOINT,
                P2_R8_GRASP_TRANSFER_SHA256,
            )
            return trainer
        if config.environment_type == "s1_lift":
            from .experimental.manipulation.lift_ppo import ExperimentalLiftPPOTrainer
            trainer = ExperimentalLiftPPOTrainer(
                environment, config=ppo_config, device=config.device,
            )
            if not P3_1_LIFT_TRANSFER_CHECKPOINT.is_file():
                raise RuntimeError("P3-1 Lift transfer checkpoint is missing")
            trainer.initialize_from_p3_1(
                P3_1_LIFT_TRANSFER_CHECKPOINT,
                P3_1_LIFT_TRANSFER_SHA256,
            )
            return trainer
        if config.environment_type == "s1_lower_release":
            from .experimental.manipulation.lower_release_ppo import (
                ExperimentalLowerReleasePPOTrainer,
            )
            trainer = ExperimentalLowerReleasePPOTrainer(
                environment, config=ppo_config, device=config.device,
            )
            if not P4_1_PLACE_TRANSFER_CHECKPOINT.is_file():
                raise RuntimeError("P4-1 Place transfer checkpoint is missing")
            trainer.initialize_from_p4_1(P4_1_PLACE_TRANSFER_CHECKPOINT)
            return trainer
        if config.environment_type == "s1_reach":
            from .ppo.policy import REACH_POLICY_SCHEMA
            from .ppo.trainer import PPOTaskContract
            from .reach_adapter import (
                REACH_REWARD_SCHEMA_VERSION, REACH_REWARD_SEMANTICS,
                REACH_REWARD_TERM_NAMES, reward_schema_metadata,
            )
            contract = PPOTaskContract(
                policy_schema=REACH_POLICY_SCHEMA,
                reward_schema_version=REACH_REWARD_SCHEMA_VERSION,
                reward_semantics=REACH_REWARD_SEMANTICS,
                reward_term_names=REACH_REWARD_TERM_NAMES,
                reward_metadata_factory=reward_schema_metadata,
            )
            return PPOTrainer(
                environment, config=ppo_config, device=config.device,
                task_contract=contract,
            )
        return PPOTrainer(environment, config=ppo_config, device=config.device)

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
