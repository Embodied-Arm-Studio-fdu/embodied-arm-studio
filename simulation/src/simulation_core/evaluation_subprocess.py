"""Production launcher and result validation for final policy evaluation."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from .artifacts import ArtifactPaths
from .evaluation import wilson_confidence_interval
from .ppo import TrainingCancelled


EVALUATION_EPISODES = 100
EVALUATION_TIMEOUT_SECONDS = 3600.0
EVALUATION_REPLAY_TIMEOUT_SECONDS = 600.0
EVALUATION_REPLAY_FPS = 12
EVALUATION_VIDEO_PATH = "evaluation_video/successful_episode.mp4"
EVALUATION_VIDEO_PARTIAL_SUFFIX = ".partial.mp4"
_REQUIRED_SUMMARY_FIELDS = (
    "episode_count",
    "success_count",
    "success_rate",
    "timeout_rate",
    "failure_rate",
    "mean_episode_length",
    "mean_episode_return",
    "mean_final_distance",
    "mean_minimum_distance",
    "mean_object_displacement",
    "mean_action_norm",
    "action_saturation_rate",
)
_REQUIRED_EPISODE_FIELDS = (
    "seed", "success", "timeout", "failure", "episode_return",
    "episode_length", "final_distance", "minimum_distance",
    "object_displacement", "action_norm", "action_saturation_rate",
)
POLICY_SUCCESS_RATE = 0.80
POLICY_WILSON_LOWER_BOUND = 0.70


class EvaluationInfrastructureError(RuntimeError):
    """The evaluation process or its persisted result was invalid."""


class EvaluationVideoError(RuntimeError):
    """The isolated successful-episode replay could not produce video."""


def _assert_finite(value: Any, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise EvaluationInfrastructureError(
                "non-finite evaluation value at %s" % location
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, "%s.%s" % (location, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, "%s[%d]" % (location, index))


def validate_evaluation_outputs(run_root: Path) -> Mapping[str, Any]:
    """Validate the complete on-disk contract without judging policy quality."""
    output = Path(run_root) / "evaluation"
    summary_path = output / "summary.json"
    episodes_path = output / "episodes.jsonl"
    if not summary_path.is_file():
        raise EvaluationInfrastructureError("evaluation summary.json is missing")
    if not episodes_path.is_file():
        raise EvaluationInfrastructureError("evaluation episodes.jsonl is missing")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationInfrastructureError("evaluation summary.json is invalid") from exc
    missing = [name for name in _REQUIRED_SUMMARY_FIELDS if name not in summary]
    if missing:
        raise EvaluationInfrastructureError(
            "evaluation summary is missing fields: %s" % ", ".join(missing)
        )
    interval = summary.get("confidence_interval_95")
    if not isinstance(interval, Mapping) or not {"lower", "upper"} <= set(interval):
        raise EvaluationInfrastructureError("evaluation Wilson 95% CI is missing")
    if isinstance(summary["episode_count"], bool) or not isinstance(
        summary["episode_count"], int
    ) or summary["episode_count"] != EVALUATION_EPISODES:
        raise EvaluationInfrastructureError("evaluation did not complete 100 episodes")
    if isinstance(summary["success_count"], bool) or not isinstance(
        summary["success_count"], int
    ) or not (0 <= summary["success_count"] <= EVALUATION_EPISODES):
        raise EvaluationInfrastructureError("evaluation success_count is invalid")
    _assert_finite(summary, "summary")
    records = []
    try:
        with episodes_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise EvaluationInfrastructureError(
                        "blank evaluation episode record at line %d" % line_number
                    )
                record = json.loads(line)
                missing_episode = [
                    name for name in _REQUIRED_EPISODE_FIELDS if name not in record
                ]
                if missing_episode:
                    raise EvaluationInfrastructureError(
                        "evaluation episode is missing fields: %s"
                        % ", ".join(missing_episode)
                    )
                _assert_finite(record, "episodes[%d]" % (line_number - 1))
                records.append(record)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationInfrastructureError("evaluation episodes.jsonl is invalid") from exc
    if len(records) != EVALUATION_EPISODES:
        raise EvaluationInfrastructureError("evaluation episodes.jsonl must contain 100 records")
    success_count = sum(record["success"] is True for record in records)
    if any(type(record["success"]) is not bool for record in records):
        raise EvaluationInfrastructureError("evaluation episode success must be boolean")
    if success_count != summary["success_count"]:
        raise EvaluationInfrastructureError("evaluation success counts are inconsistent")
    expected_rate = success_count / EVALUATION_EPISODES
    if not math.isclose(float(summary["success_rate"]), expected_rate, abs_tol=1e-12):
        raise EvaluationInfrastructureError("evaluation success_rate is inconsistent")
    expected_lower, expected_upper = wilson_confidence_interval(
        success_count, EVALUATION_EPISODES
    )
    if interval.get("method") != "wilson" or not math.isclose(
        float(interval["lower"]), expected_lower, abs_tol=1e-12
    ) or not math.isclose(float(interval["upper"]), expected_upper, abs_tol=1e-12):
        raise EvaluationInfrastructureError("evaluation Wilson 95% CI is inconsistent")
    successful_seeds = sorted(
        int(record["seed"]) for record in records if record["success"] is True
    )
    selected_seed = summary.get("selected_success_seed")
    expected_seed = successful_seeds[0] if successful_seeds else None
    if selected_seed != expected_seed:
        raise EvaluationInfrastructureError(
            "evaluation selected_success_seed is not the minimum successful seed"
        )
    checkpoint_sha = summary.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha
    ):
        raise EvaluationInfrastructureError("evaluation checkpoint SHA-256 is invalid")
    video = summary.get("evaluation_video")
    if not isinstance(video, Mapping) or video.get("status") not in {
        "available", "unavailable",
    }:
        raise EvaluationInfrastructureError("evaluation video status is invalid")
    if video["status"] == "available":
        expected_video = {
            "status": "available",
            "path": EVALUATION_VIDEO_PATH,
            "seed": expected_seed,
            "success": True,
            "model_sha256": checkpoint_sha,
            "width": 640,
            "height": 480,
            "fps": EVALUATION_REPLAY_FPS,
        }
        if dict(video) != expected_video:
            raise EvaluationInfrastructureError("evaluation video metadata is inconsistent")
        if expected_seed is None or not (Path(run_root) / EVALUATION_VIDEO_PATH).is_file():
            raise EvaluationInfrastructureError("available evaluation video is missing")
    elif not isinstance(video.get("failure_reason"), str) or not video["failure_reason"]:
        raise EvaluationInfrastructureError(
            "unavailable evaluation video requires failure_reason"
        )
    return summary


def evaluation_contract(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the compact terminal contract and judge policy independently."""
    episode_count = int(summary["episode_count"])
    success_rate = float(summary["success_rate"])
    interval = summary["confidence_interval_95"]
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    is_reach = summary.get("task_id") == "s1_reach"
    success_threshold = 0.90 if is_reach else POLICY_SUCCESS_RATE
    wilson_threshold = 0.80 if is_reach else POLICY_WILSON_LOWER_BOUND
    passed = (
        episode_count >= EVALUATION_EPISODES
        and success_rate >= success_threshold
        and lower >= wilson_threshold
    )
    contract = {
        "status": "passed" if passed else "failed",
        "deterministic": True,
        "policy_passed": passed,
        "episode_count": episode_count,
        "success_count": int(summary["success_count"]),
        "success_rate": success_rate,
        "wilson_95_lower_bound": lower,
        "wilson_95_upper_bound": upper,
    }
    for name in (
        "mean_episode_length", "mean_episode_return", "mean_final_distance",
        "mean_minimum_distance", "mean_object_displacement", "timeout_rate",
        "failure_rate",
    ):
        value = summary.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                contract[name] = number
    return contract


class ProductionEvaluationLauncher:
    """Run the proven diagnostic evaluator in a fresh Python process."""

    def __init__(
        self,
        artifacts_root: Path,
        timeout_seconds: float = EVALUATION_TIMEOUT_SECONDS,
        task_id: str = "s1_push",
        popen_factory: Callable[..., Any] = subprocess.Popen,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("evaluation timeout must be positive")
        self._artifacts_root = Path(artifacts_root)
        if task_id not in (
            "s1_push", "s1_reach", "s1_grasp", "s1_lift",
            "s1_lower_release", "s1_place",
        ):
            raise ValueError("unsupported evaluation task_id: %s" % task_id)
        self._task_id = task_id
        self._timeout_seconds = float(timeout_seconds)
        self._popen_factory = popen_factory
        self._monotonic_clock = monotonic_clock

    def run(
        self,
        project_id: str,
        run_id: str,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> Mapping[str, Any]:
        run_root = ArtifactPaths(self._artifacts_root).run_root(project_id, run_id)
        checkpoint = run_root / "checkpoints" / "model_latest.pt"
        if not checkpoint.is_file():
            raise EvaluationInfrastructureError(
                "evaluation checkpoint model_latest.pt is missing"
            )
        script_name = {
            "s1_push": "push_diagnostic_evaluation.py",
            "s1_reach": "reach_diagnostic_evaluation.py",
            "s1_grasp": "manipulation_diagnostic_evaluation.py",
            "s1_lift": "manipulation_diagnostic_evaluation.py",
            "s1_lower_release": "manipulation_diagnostic_evaluation.py",
            "s1_place": "place_diagnostic_evaluation.py",
        }[self._task_id]
        script = Path(__file__).resolve().parents[2] / "scripts" / script_name
        simulation_root = script.parent.parent
        if not script.is_file():
            raise EvaluationInfrastructureError("evaluation entrypoint is missing")
        output = run_root / "evaluation"
        output.mkdir(parents=True, exist_ok=True)
        command = (
            sys.executable,
            str(script),
            "--run-root",
            str(run_root),
            "--episodes",
            str(EVALUATION_EPISODES),
        )
        if self._task_id.startswith("s1_") and self._task_id not in {
            "s1_push", "s1_reach",
        }:
            command += ("--task-id", self._task_id)
        return_code = self._run_process(
            command,
            output / "stdout.log",
            output / "stderr.log",
            simulation_root,
            self._timeout_seconds,
            cancel_requested,
            "evaluation",
        )
        if return_code != 0:
            raise EvaluationInfrastructureError(
                "evaluation subprocess exited with code %d" % return_code
            )
        summary = dict(validate_evaluation_outputs(run_root))
        seed = summary.get("selected_success_seed")
        if seed is None:
            summary["evaluation_video"] = {
                "status": "unavailable",
                "failure_reason": "no_successful_seed",
            }
            self._write_summary(output / "summary.json", summary)
            return validate_evaluation_outputs(run_root)
        video_path = run_root / EVALUATION_VIDEO_PATH
        replay_command = (
            sys.executable,
            str(script),
            "--run-root",
            str(run_root),
            "--replay-seed",
            str(seed),
            "--video-output",
            str(video_path),
            "--video-fps",
            str(EVALUATION_REPLAY_FPS),
        )
        if self._task_id.startswith("s1_") and self._task_id not in {
            "s1_push", "s1_reach",
        }:
            replay_command += ("--task-id", self._task_id)
        try:
            self._prepare_replay_target(video_path)
            replay_code = self._run_process(
                replay_command,
                output / "replay_stdout.log",
                output / "replay_stderr.log",
                simulation_root,
                EVALUATION_REPLAY_TIMEOUT_SECONDS,
                cancel_requested,
                "evaluation replay",
            )
            if replay_code != 0:
                reason = (
                    "replay_subprocess_sigsegv"
                    if replay_code == -11 else "replay_subprocess_exit_%d" % replay_code
                )
                raise EvaluationVideoError(reason)
            if not video_path.is_file():
                raise EvaluationVideoError("replay_video_missing")
            summary["evaluation_video"] = {
                "status": "available",
                "path": EVALUATION_VIDEO_PATH,
                "seed": seed,
                "success": True,
                "model_sha256": summary["checkpoint_sha256"],
                "width": 640,
                "height": 480,
                "fps": EVALUATION_REPLAY_FPS,
            }
        except EvaluationVideoError as exc:
            self._cleanup_replay_outputs(video_path)
            summary["evaluation_video"] = {
                "status": "unavailable",
                "failure_reason": str(exc),
            }
        self._write_summary(output / "summary.json", summary)
        return validate_evaluation_outputs(run_root)

    def _run_process(
        self, command, stdout_path, stderr_path, cwd, timeout_seconds,
        cancel_requested, label,
    ) -> int:
        started = self._monotonic_clock()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                process = self._popen_factory(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    shell=False,
                    cwd=str(cwd),
                )
            except (OSError, ValueError) as exc:
                error_type = (
                    EvaluationInfrastructureError
                    if label == "evaluation" else EvaluationVideoError
                )
                raise error_type("failed to start %s subprocess" % label) from exc
            while True:
                if cancel_requested():
                    self._stop_process(process)
                    raise TrainingCancelled()
                remaining = timeout_seconds - (self._monotonic_clock() - started)
                if remaining <= 0:
                    self._stop_process(process)
                    error_type = (
                        EvaluationInfrastructureError
                        if label == "evaluation" else EvaluationVideoError
                    )
                    raise error_type("%s_subprocess_timeout" % label.replace(" ", "_"))
                try:
                    return process.wait(timeout=min(0.25, remaining))
                except subprocess.TimeoutExpired:
                    continue

    @staticmethod
    def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
        temporary = path.with_name(".%s.tmp" % path.name)
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    @staticmethod
    def _cleanup_replay_partial(video_path: Path) -> None:
        partial = video_path.with_name(
            video_path.name + EVALUATION_VIDEO_PARTIAL_SUFFIX
        )
        try:
            partial.unlink()
        except OSError:
            pass

    @classmethod
    def _cleanup_replay_outputs(cls, video_path: Path) -> None:
        cls._cleanup_replay_partial(video_path)
        try:
            video_path.unlink()
        except OSError:
            pass

    @classmethod
    def _prepare_replay_target(cls, video_path: Path) -> None:
        cls._cleanup_replay_partial(video_path)
        try:
            video_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise EvaluationVideoError("replay_target_cleanup_failed") from exc

    @staticmethod
    def _stop_process(process: Any) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
