"""PPO-free lifecycle runner for bounded control demos."""

import json
import inspect
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional

from .domain import TaskStatus


class DemoTaskRunner:
    """Initialize, tick, stop, and clean up one non-training runtime."""

    def __init__(
        self,
        artifacts_root: Path,
        runtime_factory: Callable[[Any], Any],
        progress_callback: Optional[Callable[[TaskStatus], None]] = None,
        cancel_requested: Optional[Callable[[], bool]] = None,
        metric_callback: Optional[Callable[[Mapping[str, Any]], None]] = None,
        stream_gateway=None,
        stream_lifecycle_factory=None,
        recorder_factory=None,
        monotonic_clock: Callable[[], float] = None,
        sleeper: Callable[[float], None] = None,
    ) -> None:
        if not callable(runtime_factory):
            raise TypeError("runtime_factory must be callable")
        self._artifacts_root = Path(artifacts_root)
        self._runtime_factory = runtime_factory
        self._progress_callback = progress_callback or (lambda status: None)
        self._cancel_requested = cancel_requested or (lambda: False)
        self._metric_callback = metric_callback or (lambda record: None)
        self._stream_gateway = stream_gateway
        self._stream_lifecycle_factory = stream_lifecycle_factory
        self._recorder_factory = recorder_factory
        self._clock = monotonic_clock or time.monotonic
        self._sleep = sleeper or time.sleep
        self._ticks = 0

    @property
    def training_steps(self):
        return 0

    @property
    def update_steps(self):
        return 0

    def run(self, task):
        from .task_runner import TaskExecutionResult

        runtime = None
        stream_lifecycle = None
        records = []
        status = TaskStatus.FAILED
        self._ticks = 0
        metrics_path = (
            self._artifacts_root / task.project_id / task.run_id
            / "metrics" / "demo.jsonl"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            parameters = inspect.signature(self._runtime_factory).parameters
            if "task" in parameters:
                runtime = self._runtime_factory(task.config, task=task)
            else:
                runtime = self._runtime_factory(task.config)
            runtime.initialize()
            self._progress_callback(TaskStatus.RUNNING)
            if task.config.video_enabled:
                try:
                    stream_lifecycle = self._create_stream_lifecycle(runtime, task)
                    self._publish_streams(stream_lifecycle.on_status(
                        task.run_id, task.task_id, TaskStatus.RUNNING
                    ))
                except Exception:
                    stream_lifecycle = None

            started = self._clock()
            while self._clock() - started < task.config.duration_seconds:
                if self._cancel_requested():
                    status = TaskStatus.STOPPED
                    break
                values = runtime.step()
                self._ticks += 1
                record = {"tick": self._ticks, "elapsed_seconds": self._clock() - started}
                if values is not None:
                    record.update(dict(values))
                records.append(record)
                self._metric_callback(record)
                self._append_metric(metrics_path, record)
                remaining = task.config.duration_seconds - (self._clock() - started)
                if remaining > 0:
                    self._sleep(min(task.config.tick_interval_seconds, remaining))
            else:
                status = TaskStatus.COMPLETED

            if status == TaskStatus.COMPLETED:
                self._progress_callback(TaskStatus.FINALIZING)
            summary = dict(records[-1]) if records else {
                "tick": self._ticks,
                "elapsed_seconds": max(0.0, self._clock() - started),
            }
            if not records:
                self._append_metric(metrics_path, summary)
            return TaskExecutionResult(
                task_id=task.task_id,
                run_id=task.run_id,
                status=status,
                metrics_path=str(metrics_path),
                final_metrics_summary=summary,
            )
        finally:
            if stream_lifecycle is not None:
                try:
                    self._publish_streams(stream_lifecycle.on_status(
                        task.run_id, task.task_id, status
                    ))
                except Exception:
                    pass
            if runtime is not None:
                runtime.close()

    @staticmethod
    def _append_metric(path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(record), sort_keys=True) + "\n")

    def _create_stream_lifecycle(self, runtime, task):
        if self._stream_lifecycle_factory is not None:
            return self._stream_lifecycle_factory(runtime, task)
        from .video_runtime import TaskVideoRuntimeFactory
        return TaskVideoRuntimeFactory(
            self._artifacts_root,
            stream_gateway=self._stream_gateway,
            recorder_factory=self._recorder_factory,
        ).create(runtime, task)

    def _publish_streams(self, sessions):
        if self._stream_gateway is None:
            return
        for session in sessions:
            self._stream_gateway.publish(session)
