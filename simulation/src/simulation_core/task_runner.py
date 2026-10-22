"""Task-neutral spec/result and a thin training/demo dispatcher."""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence

from .domain import SimulationTask, TaskStatus
from .task_registry import (
    DEMO_RUNNER, PRODUCTION_TASK_REGISTRY, TRAINING_RUNNER, TaskRegistry,
)


@dataclass(frozen=True)
class ExecutionTaskSpec:
    task_id: str
    run_id: str
    project_id: str
    config: Any
    definition_id: str
    created_at: str = ""

    @classmethod
    def from_task(cls, task: SimulationTask, registry=PRODUCTION_TASK_REGISTRY):
        config = registry.parse_config(task.config.values)
        return cls(
            task_id=task.task_id,
            run_id=task.run_id,
            project_id=task.project_id,
            config=config,
            definition_id=config.environment_type,
            created_at=task.created_at.isoformat(),
        )

    @classmethod
    def from_dict(cls, values, registry=PRODUCTION_TASK_REGISTRY):
        definition_id = str(values.get(
            "definition_id", values["config"].get("environment_type", "")
        ))
        definition = registry.get(definition_id)
        config = definition.parse_config(values["config"])
        return cls(
            task_id=str(values["task_id"]),
            run_id=str(values["run_id"]),
            project_id=str(values["project_id"]),
            config=config,
            definition_id=definition_id,
            created_at=str(values.get("created_at", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "project_id": self.project_id,
            "definition_id": self.definition_id,
            "config": self.config.to_dict(),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TaskExecutionResult:
    task_id: str
    run_id: str
    status: TaskStatus
    training_steps: int = 0
    update_steps: int = 0
    metrics_path: str = ""
    checkpoint_path: str = ""
    final_metrics_summary: Mapping[str, Any] = field(default_factory=dict)
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
    def from_dict(cls, values):
        return cls(
            task_id=str(values["task_id"]), run_id=str(values["run_id"]),
            status=TaskStatus(values["status"]),
            training_steps=int(values.get("training_steps", 0)),
            update_steps=int(values.get("update_steps", 0)),
            metrics_path=str(values.get("metrics_path", "")),
            checkpoint_path=str(values.get("checkpoint_path", "")),
            final_metrics_summary=dict(values.get("final_metrics_summary", {})),
            evaluation=(dict(values["evaluation"]) if values.get("evaluation") is not None else None),
            artifact_root=str(values.get("artifact_root", "")),
            manifest_path=str(values.get("manifest_path", "")),
            best_checkpoint_path=str(values.get("best_checkpoint_path", "")),
            log_paths=dict(values.get("log_paths", {})),
            error=(dict(values["error"]) if values.get("error") is not None else None),
            artifacts=tuple(dict(item) for item in values.get("artifacts", ())),
            started_at=str(values.get("started_at", "")),
            ended_at=str(values.get("ended_at", "")),
        )

    def to_dict(self):
        return {
            "task_id": self.task_id, "run_id": self.run_id,
            "status": self.status.value, "training_steps": self.training_steps,
            "update_steps": self.update_steps, "metrics_path": self.metrics_path,
            "checkpoint_path": self.checkpoint_path,
            "final_metrics_summary": dict(self.final_metrics_summary),
            "evaluation": dict(self.evaluation) if self.evaluation is not None else None,
            "artifact_root": self.artifact_root, "manifest_path": self.manifest_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "log_paths": dict(self.log_paths),
            "error": dict(self.error) if self.error is not None else None,
            "artifacts": [dict(item) for item in self.artifacts],
            "started_at": self.started_at, "ended_at": self.ended_at,
        }

    def with_artifacts(self, summary, error=None):
        return replace(
            self, status=summary.final_status,
            artifact_root=summary.artifact_root, manifest_path=summary.manifest_path,
            metrics_path=summary.metrics_path,
            checkpoint_path=summary.latest_checkpoint_path,
            best_checkpoint_path=summary.best_checkpoint_path,
            log_paths=dict(summary.log_paths),
            error=dict(error) if error is not None else None,
            artifacts=tuple(dict(item) for item in summary.artifacts),
            training_steps=summary.training_steps,
            update_steps=summary.update_steps,
            final_metrics_summary=dict(summary.final_metrics_summary),
            started_at=summary.started_at, ended_at=summary.ended_at,
        )


class TaskRunner:
    """Select a runner from static metadata; contains no task business logic."""

    def __init__(
        self, artifacts_root, registry: TaskRegistry = PRODUCTION_TASK_REGISTRY,
        progress_callback=None, cancel_requested=None, metric_callback=None,
        stream_gateway=None, failure_injector=None,
        training_runner_factory=None, demo_runner_factory=None,
    ):
        self.artifacts_root = artifacts_root
        self.registry = registry
        self.progress_callback = progress_callback
        self.cancel_requested = cancel_requested
        self.metric_callback = metric_callback
        self.stream_gateway = stream_gateway
        self.failure_injector = failure_injector
        self.training_runner_factory = training_runner_factory
        self.demo_runner_factory = demo_runner_factory
        self.active_runner = None

    @property
    def training_steps(self):
        return int(getattr(self.active_runner, "training_steps", 0))

    @property
    def update_steps(self):
        return int(getattr(self.active_runner, "update_steps", 0))

    def run(self, task: ExecutionTaskSpec) -> TaskExecutionResult:
        definition = self.registry.get(task.definition_id)
        if definition.runner_kind == TRAINING_RUNNER:
            from .training_task_runner import TrainingTaskRunner, TrainingTaskSpec
            factory = self.training_runner_factory or definition.runner_factory
            runner = factory() if factory is not None else TrainingTaskRunner(
                self.artifacts_root,
                progress_callback=self.progress_callback,
                cancel_requested=self.cancel_requested,
                metric_callback=self.metric_callback,
                stream_gateway=self.stream_gateway,
                failure_injector=self.failure_injector,
            )
            self.active_runner = runner
            result = runner.run(TrainingTaskSpec(
                task.task_id, task.run_id, task.project_id, task.config,
                task.created_at,
            ))
            return TaskExecutionResult.from_dict(result.to_dict())
        if definition.runner_kind == DEMO_RUNNER:
            from .demo_runner import DemoTaskRunner
            factory = self.demo_runner_factory or definition.runner_factory
            runner = factory() if factory is not None else DemoTaskRunner(
                self.artifacts_root,
                runtime_factory=definition.runtime_factory,
                progress_callback=self.progress_callback,
                cancel_requested=self.cancel_requested,
                metric_callback=self.metric_callback,
                stream_gateway=self.stream_gateway,
            )
            self.active_runner = runner
            return runner.run(task)
        raise ValueError("unsupported runner kind: %s" % definition.runner_kind)
