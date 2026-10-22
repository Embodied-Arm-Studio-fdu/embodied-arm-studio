"""Run-local artifact preparation, validation, metadata, and manifest writing."""

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Dict, Mapping, Optional, Tuple

from .artifacts import ArtifactPaths
from .domain import TaskStatus
from .video_recorder import probe_mp4


class ArtifactValidationError(RuntimeError):
    """Raised when a completed run is missing a required usable artifact."""


@dataclass(frozen=True)
class ArtifactPolicy:
    """Task-kind-specific files without weakening the training contract."""

    name: str
    config_type: str
    config_path: str
    metrics_type: str
    metrics_path: str
    video_type: str
    completed_required: frozenset
    include_checkpoints: bool
    include_evaluation: bool


TRAINING_ARTIFACT_POLICY = ArtifactPolicy(
    name="training",
    config_type="training_config",
    config_path="config/training_config.json",
    metrics_type="training_metrics",
    metrics_path="metrics/training.jsonl",
    video_type="training_video",
    completed_required=frozenset({
        "checkpoints/model_best.pt",
        "checkpoints/model_latest.pt",
        "config/training_config.json",
        "logs/stderr.log",
        "logs/stdout.log",
        "logs/task.log",
        "metrics/training.jsonl",
    }),
    include_checkpoints=True,
    include_evaluation=True,
)

DEMO_ARTIFACT_POLICY = ArtifactPolicy(
    name="demo",
    config_type="demo_config",
    config_path="config/demo_config.json",
    metrics_type="demo_metrics",
    metrics_path="metrics/demo.jsonl",
    video_type="demo_video",
    completed_required=frozenset({
        "config/demo_config.json",
        "logs/stderr.log",
        "logs/stdout.log",
        "logs/task.log",
        "metrics/demo.jsonl",
    }),
    include_checkpoints=False,
    include_evaluation=False,
)


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(".%s.tmp" % path.name)
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactSummary:
    """Pure-data artifact result safe to send across a process boundary."""

    final_status: TaskStatus
    artifact_root: str
    manifest_path: str
    metrics_path: str = ""
    latest_checkpoint_path: str = ""
    best_checkpoint_path: str = ""
    log_paths: Mapping[str, str] = field(default_factory=dict)
    artifacts: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    training_steps: int = 0
    update_steps: int = 0
    final_metrics_summary: Mapping[str, Any] = field(default_factory=dict)
    started_at: str = ""
    ended_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_status": self.final_status.value,
            "artifact_root": self.artifact_root,
            "manifest_path": self.manifest_path,
            "metrics_path": self.metrics_path,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "best_checkpoint_path": self.best_checkpoint_path,
            "log_paths": dict(self.log_paths),
            "artifacts": [dict(item) for item in self.artifacts],
            "training_steps": self.training_steps,
            "update_steps": self.update_steps,
            "final_metrics_summary": dict(self.final_metrics_summary),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


class ArtifactFinalizer:
    """Own one Run's local artifact layout and terminal manifest."""

    _KNOWN_ARTIFACTS = (
        ("checkpoint_best", "checkpoints/model_best.pt"),
        ("checkpoint_latest", "checkpoints/model_latest.pt"),
        ("training_config", "config/training_config.json"),
        ("stderr_log", "logs/stderr.log"),
        ("stdout_log", "logs/stdout.log"),
        ("task_log", "logs/task.log"),
        ("training_metrics", "metrics/training.jsonl"),
        ("training_video", "video/overview.mp4"),
    )
    _EVALUATION_ARTIFACTS = (
        ("evaluation_summary", "evaluation/summary.json"),
        ("evaluation_episodes", "evaluation/episodes.jsonl"),
        ("evaluation_video", "evaluation_video/successful_episode.mp4"),
    )
    _COMPLETED_REQUIRED = frozenset(
        {
            "checkpoints/model_best.pt",
            "checkpoints/model_latest.pt",
            "config/training_config.json",
            "logs/stderr.log",
            "logs/stdout.log",
            "logs/task.log",
            "metrics/training.jsonl",
        }
    )

    def __init__(
        self,
        artifacts_root: Path,
        policy: ArtifactPolicy = TRAINING_ARTIFACT_POLICY,
    ) -> None:
        self._paths = ArtifactPaths(Path(artifacts_root))
        self._policy = policy
        self._run_root: Optional[Path] = None
        self._started_at: Optional[str] = None
        self._task = None

    @property
    def run_root(self) -> Path:
        if self._run_root is None:
            raise RuntimeError("ArtifactFinalizer.prepare() must be called first")
        return self._run_root

    def prepare(
        self,
        task,
        child_pid: Optional[int] = None,
        initial_status: TaskStatus = TaskStatus.INITIALIZING,
    ) -> Path:
        """Create the stable Run layout and write the exact config snapshot."""
        self._task = task
        self._run_root = self._paths.run_root(task.project_id, task.run_id)
        self._started_at = _utc_now_text()
        directories = ["metrics", "logs", "config"]
        if self._policy.include_checkpoints:
            directories.insert(0, "checkpoints")
        for directory in directories:
            (self.run_root / directory).mkdir(parents=True, exist_ok=True)
        for log_name in ("stdout.log", "stderr.log", "task.log"):
            (self.run_root / "logs" / log_name).touch(exist_ok=True)
        _atomic_json_write(
            self.run_root / self._policy.config_path,
            task.config.to_dict(),
        )
        self.log_event(
            "task_started",
            status=initial_status.value,
            child_pid=child_pid,
        )
        return self.run_root

    @contextmanager
    def capture_process_output(self):
        """Capture Python and native child-process fd 1/2 output into Run logs."""
        stdout_path = self.run_root / "logs" / "stdout.log"
        stderr_path = self.run_root / "logs" / "stderr.log"
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        stdout_stream = stdout_path.open("a", encoding="utf-8", buffering=1)
        stderr_stream = stderr_path.open("a", encoding="utf-8", buffering=1)
        try:
            os.dup2(stdout_stream.fileno(), 1)
            os.dup2(stderr_stream.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            # Isaac Gym and CUDA extensions may write through native stdio.
            # Flush those buffers before restoring fds so manifest hashes match
            # the final log contents observed after process exit.
            try:
                ctypes.CDLL(None).fflush(None)
            except Exception:
                pass
            stdout_stream.flush()
            stderr_stream.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            stdout_stream.close()
            stderr_stream.close()

    def log_event(self, event: str, **fields: Any) -> None:
        record = {"timestamp": _utc_now_text(), "event": str(event)}
        record.update(
            {name: value for name, value in fields.items() if value is not None}
        )
        with (self.run_root / "logs" / "task.log").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def finalize(
        self,
        final_status: TaskStatus,
        training_steps: Optional[int] = None,
        update_steps: Optional[int] = None,
        final_metrics_summary: Optional[Mapping[str, Any]] = None,
        evaluation: Optional[Mapping[str, Any]] = None,
        error: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactSummary:
        if final_status not in (
            TaskStatus.COMPLETED,
            TaskStatus.STOPPED,
            TaskStatus.FAILED,
        ):
            raise ValueError("manifest final_status must be terminal")

        metrics_summary = self._last_metrics_record()
        summary = dict(final_metrics_summary or metrics_summary)
        resolved_training_steps = (
            int(training_steps)
            if training_steps is not None
            else int(metrics_summary.get("training_steps", 0))
        )
        resolved_update_steps = (
            int(update_steps)
            if update_steps is not None
            else int(metrics_summary.get("update_index", 0))
        )
        video_probe = self._training_video_probe_best_effort()
        self.log_event("cleanup_complete", final_status=final_status.value)
        artifact_definitions = self._artifact_definitions(evaluation)
        for artifact_type, relative_path in artifact_definitions:
            if artifact_type == self._policy.video_type and video_probe is None:
                continue
            if (self.run_root / relative_path).is_file():
                self.log_event(
                    "artifact_ready", type=artifact_type, path=relative_path
                )
        self.log_event(
            "final_result",
            final_status=final_status.value,
            training_steps=resolved_training_steps,
            update_steps=resolved_update_steps,
        )

        artifacts = self._artifact_metadata(
            artifact_definitions,
            include_training_video=video_probe is not None,
        )
        available_paths = {item["path"] for item in artifacts}
        if final_status == TaskStatus.COMPLETED:
            missing = self._policy.completed_required.difference(available_paths)
            if missing:
                raise ArtifactValidationError(
                    "completed run is missing required artifacts: %s"
                    % ", ".join(sorted(missing))
                )
            if not summary:
                raise ArtifactValidationError(
                    "completed run has no final metrics summary"
                )

        ended_at = _utc_now_text()
        manifest_artifacts = []
        for item in artifacts:
            manifest_item = dict(item)
            if item["type"] == self._policy.video_type and video_probe is not None:
                manifest_item["metadata"] = {
                    "actual_fps": video_probe.actual_fps,
                    "duration_seconds": video_probe.duration,
                }
            manifest_artifacts.append(manifest_item)
        manifest = {
            "schema_version": 1,
            "project_id": self._task.project_id,
            "run_id": self._task.run_id,
            "task_id": self._task.task_id,
            "final_status": final_status.value,
            "created_at": self._task.created_at,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "training_steps": resolved_training_steps,
            "update_steps": resolved_update_steps,
            "seed": self._task.config.seed,
            "environment_type": self._task.config.environment_type,
            "artifacts": manifest_artifacts,
            "final_metrics_summary": summary,
            "evaluation": dict(evaluation) if evaluation is not None else None,
            "error": dict(error) if error is not None else None,
        }
        manifest_path = self.run_root / "manifest.json"
        _atomic_json_write(manifest_path, manifest)
        # The manifest intentionally does not describe itself.  Its callback
        # metadata is calculated only after the atomic writer has flushed,
        # fsynced, closed, and installed the final file.
        manifest_metadata = {
            "type": "manifest",
            "path": manifest_path.relative_to(self.run_root).as_posix(),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": _file_sha256(manifest_path),
        }
        callback_artifacts = tuple(artifacts) + (manifest_metadata,)
        paths_by_type = {item["type"]: item["path"] for item in artifacts}

        def absolute_path(artifact_type: str) -> str:
            relative = paths_by_type.get(artifact_type)
            return str(self.run_root / relative) if relative else ""

        return ArtifactSummary(
            final_status=final_status,
            artifact_root=str(self.run_root),
            manifest_path=str(manifest_path),
            metrics_path=(
                absolute_path("training_episode_metrics")
                or absolute_path(self._policy.metrics_type)
            ),
            latest_checkpoint_path=absolute_path("checkpoint_latest"),
            best_checkpoint_path=absolute_path("checkpoint_best"),
            log_paths={
                "stdout": absolute_path("stdout_log"),
                "stderr": absolute_path("stderr_log"),
                "task": absolute_path("task_log"),
            },
            artifacts=callback_artifacts,
            training_steps=resolved_training_steps,
            update_steps=resolved_update_steps,
            final_metrics_summary=summary,
            started_at=self._started_at or self._task.created_at,
            ended_at=ended_at,
        )

    def _artifact_metadata(
        self, artifact_definitions, include_training_video: bool = False
    ) -> Tuple[Mapping[str, Any], ...]:
        records = []
        for artifact_type, relative_path in artifact_definitions:
            if artifact_type == self._policy.video_type and not include_training_video:
                continue
            validated = self._safe_registered_file(relative_path)
            if validated is None:
                continue
            path, metadata = validated
            records.append(
                {
                    "type": artifact_type,
                    "path": path.relative_to(self.run_root).as_posix(),
                    "size_bytes": metadata.st_size,
                    "sha256": _file_sha256(path),
                }
            )
        return tuple(sorted(records, key=lambda item: item["path"]))

    def _artifact_definitions(self, evaluation):
        definitions = [
            (self._policy.config_type, self._policy.config_path),
            ("stderr_log", "logs/stderr.log"),
            ("stdout_log", "logs/stdout.log"),
            ("task_log", "logs/task.log"),
            (self._policy.metrics_type, self._policy.metrics_path),
            (self._policy.video_type, "video/overview.mp4"),
        ]
        if self._policy.include_checkpoints:
            definitions[0:0] = [
                ("checkpoint_best", "checkpoints/model_best.pt"),
                ("checkpoint_latest", "checkpoints/model_latest.pt"),
            ]
        if self._policy.include_evaluation and evaluation is not None:
            definitions = [
                (
                    "training_episode_metrics"
                    if artifact_type == "training_metrics"
                    else artifact_type,
                    relative_path,
                )
                for artifact_type, relative_path in definitions
            ]
            if evaluation.get("status") in {"passed", "failed"}:
                definitions.extend(self._EVALUATION_ARTIFACTS)
        return tuple(definitions)

    def _training_video_probe_best_effort(self):
        if self._task is None or self._task.config.video_enabled is not True:
            return None
        path = self.run_root / "video" / "overview.mp4"
        if not path.exists():
            return None
        ffmpeg_path = str(os.environ.get("SIMULATION_FFMPEG_PATH", "")).strip()
        if not ffmpeg_path:
            self.log_event(
                "video_recording_unavailable",
                failure_reason="recorder_probe_configuration_missing",
            )
            return None
        try:
            return probe_mp4(
                Path(ffmpeg_path).with_name("ffprobe"),
                path,
                self._task.config.video_width,
                self._task.config.video_height,
                self._task.config.video_fps,
            )
        except Exception:
            # A late recording validation failure is video-only. The final file
            # is deliberately retained but omitted from manifest/callback.
            self.log_event(
                "video_recording_unavailable",
                failure_reason="recorder_probe_failed",
            )
            return None

    def _safe_registered_file(self, relative_path):
        path = self.run_root / relative_path
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise ArtifactValidationError(
                "registered artifact must be a regular file: %s" % relative_path
            )
        try:
            path.resolve(strict=True).relative_to(self.run_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ArtifactValidationError(
                "registered artifact escapes the Run root: %s" % relative_path
            ) from exc
        return path, metadata

    def _last_metrics_record(self) -> Dict[str, Any]:
        validated = self._safe_registered_file(self._policy.metrics_path)
        if validated is None:
            return {}
        metrics_path, _ = validated
        last_record: Dict[str, Any] = {}
        try:
            with metrics_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        last_record = dict(json.loads(line))
        except (OSError, ValueError, TypeError) as exc:
            raise ArtifactValidationError(
                "%s metrics are not valid JSONL: %s"
                % (self._policy.name, exc)
            ) from exc
        return last_record
