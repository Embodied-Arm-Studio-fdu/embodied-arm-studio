"""Best-effort H.264 Annex-B to MP4 recording after the single NVENC encoder."""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import queue
import select
import stat
import subprocess
import threading
import time
from typing import Callable, Optional


class Mp4RecorderState(str, Enum):
    NEW = "new"
    STARTING = "starting"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    PROBING = "probing"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class Mp4RecorderConfig:
    ffmpeg_path: Path
    output_path: Path
    width: int
    height: int
    fps: float
    queue_size: int = 64
    write_timeout: float = 1.0
    close_timeout: float = 8.0
    probe_timeout: float = 5.0
    fps_tolerance: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "ffmpeg_path", Path(self.ffmpeg_path))
        object.__setattr__(self, "output_path", Path(self.output_path))
        for name in ("width", "height", "queue_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        for name in (
            "fps", "write_timeout", "close_timeout", "probe_timeout",
            "fps_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        if self.output_path.suffix.lower() != ".mp4":
            raise ValueError("output_path must end in .mp4")

    @property
    def partial_path(self) -> Path:
        return self.output_path.with_name(self.output_path.name + ".partial")

    @property
    def ffprobe_path(self) -> Path:
        return self.ffmpeg_path.with_name("ffprobe")


@dataclass(frozen=True)
class Mp4ProbeResult:
    codec: str
    width: int
    height: int
    actual_fps: float
    duration: float


@dataclass(frozen=True)
class Mp4RecorderResult:
    state: Mp4RecorderState
    output_path: str = ""
    actual_fps: Optional[float] = None
    duration: Optional[float] = None
    failure_reason: Optional[str] = None


class Mp4RecorderError(RuntimeError):
    pass


def _finite_positive_number(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Mp4RecorderError("invalid_%s" % name) from exc
    if not math.isfinite(number) or number <= 0:
        raise Mp4RecorderError("invalid_%s" % name)
    return number


def _frame_rate(value) -> float:
    if not isinstance(value, str) or not value or value.upper() == "N/A":
        raise Mp4RecorderError("invalid_fps")
    try:
        fraction = Fraction(value)
        result = float(fraction)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise Mp4RecorderError("invalid_fps") from exc
    if not math.isfinite(result) or result <= 0:
        raise Mp4RecorderError("invalid_fps")
    return result


def probe_mp4(
    ffprobe_path: Path,
    path: Path,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    fps_tolerance: float = 0.1,
    timeout: float = 5.0,
    runner: Callable = subprocess.run,
) -> Mp4ProbeResult:
    """Validate one regular finalized candidate and return measured metadata."""
    target = Path(path)
    try:
        metadata = target.lstat()
    except OSError as exc:
        raise Mp4RecorderError("probe_file_unavailable") from exc
    if target.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise Mp4RecorderError("probe_file_invalid")
    expected = _finite_positive_number(expected_fps, "expected_fps")
    try:
        completed = runner(
            [
                str(ffprobe_path), "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=codec_name,width,height,avg_frame_rate:format=duration",
                "-of", "json", str(target),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Mp4RecorderError("probe_process_failed") from exc
    if completed.returncode != 0:
        raise Mp4RecorderError("probe_process_failed")
    try:
        payload = json.loads(completed.stdout)
        streams = payload["streams"]
        if not isinstance(streams, list) or len(streams) != 1:
            raise ValueError("unexpected stream count")
        stream = streams[0]
        codec = str(stream["codec_name"])
        width = int(stream["width"])
        height = int(stream["height"])
        actual_fps = _frame_rate(stream["avg_frame_rate"])
        duration = _finite_positive_number(payload["format"]["duration"], "duration")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise Mp4RecorderError("probe_payload_invalid") from exc
    if codec != "h264":
        raise Mp4RecorderError("probe_codec_mismatch")
    if width != expected_width or height != expected_height:
        raise Mp4RecorderError("probe_dimensions_mismatch")
    if abs(actual_fps - expected) > fps_tolerance:
        raise Mp4RecorderError("probe_fps_mismatch")
    return Mp4ProbeResult(codec, width, height, actual_fps, duration)


class Mp4Recorder:
    """Bounded best-effort recorder with exactly one subprocess shutdown owner."""

    def __init__(
        self,
        config: Mp4RecorderConfig,
        popen: Callable = subprocess.Popen,
        probe: Callable = probe_mp4,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._popen = popen
        self._probe = probe
        self._monotonic = monotonic
        self._queue = queue.Queue(maxsize=config.queue_size)
        self._state = Mp4RecorderState.NEW
        self._failure_reason: Optional[str] = None
        self._probe_result: Optional[Mp4ProbeResult] = None
        self._process = None
        self._writer = None
        self._accepting = False
        self._close_requested = threading.Event()
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_owner: Optional[str] = None
        self._shutdown_done = threading.Event()
        self._close_deadline: Optional[float] = None
        self._writer_deadline: Optional[float] = None
        self._closed = False

    @property
    def state(self) -> Mp4RecorderState:
        with self._state_lock:
            return self._state

    @property
    def failure_reason(self) -> Optional[str]:
        with self._state_lock:
            return self._failure_reason

    @property
    def shutdown_owner(self) -> Optional[str]:
        with self._state_lock:
            return self._shutdown_owner

    @property
    def writer_alive(self) -> bool:
        return bool(self._writer and self._writer.is_alive())

    def start(self) -> None:
        with self._state_lock:
            if self._state == Mp4RecorderState.RECORDING:
                return
            if self._state != Mp4RecorderState.NEW:
                raise Mp4RecorderError("recorder_not_startable")
            self._state = Mp4RecorderState.STARTING
        try:
            if not self.config.ffmpeg_path.is_file():
                raise Mp4RecorderError("ffmpeg_unavailable")
            if not self.config.ffprobe_path.is_file():
                raise Mp4RecorderError("ffprobe_unavailable")
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.config.partial_path.exists() or self.config.output_path.exists():
                raise Mp4RecorderError("recording_target_exists")
            self._process = self._popen(
                self.ffmpeg_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            if self._process.stdin is None:
                raise Mp4RecorderError("recorder_stdin_unavailable")
            os.set_blocking(self._process.stdin.fileno(), False)
            self._accepting = True
            self._set_state(Mp4RecorderState.RECORDING)
            self._writer = threading.Thread(
                target=self._writer_main,
                name="mp4-recorder-writer",
                daemon=True,
            )
            self._writer.start()
        except Exception as exc:
            reason = str(exc) if isinstance(exc, Mp4RecorderError) else "recorder_open_failed"
            self._set_failed(reason)
            self._accepting = False
            self._coordinate_shutdown("start")
            raise Mp4RecorderError(reason) from exc

    def ffmpeg_command(self):
        return [
            str(self.config.ffmpeg_path), "-hide_banner", "-loglevel", "error",
            "-nostdin", "-y", "-fflags", "+genpts", "-r", str(self.config.fps),
            "-f", "h264", "-i", "pipe:0", "-an", "-c:v", "copy",
            "-movflags", "+faststart", "-f", "mp4",
            str(self.config.partial_path),
        ]

    def offer(self, data: bytes) -> bool:
        if not isinstance(data, bytes) or not data:
            self._set_failed("recorder_packet_invalid")
            return False
        if not self._accepting or self.state != Mp4RecorderState.RECORDING:
            return False
        try:
            self._queue.put_nowait(data)
            return True
        except queue.Full:
            self._accepting = False
            self._set_failed("recorder_queue_full")
            self._close_requested.set()
            return False

    def record_external_failure(self, reason: str) -> None:
        controlled = reason if reason.startswith("recorder_") else "recorder_unexpected_failure"
        self._accepting = False
        self._set_failed(controlled)
        self._close_requested.set()

    def close(self) -> Mp4RecorderResult:
        with self._close_lock:
            if self._closed:
                return self.result()
            self._accepting = False
            close_started = self._monotonic()
            self._close_deadline = close_started + self.config.close_timeout
            # Reserve most of the total close budget for the shutdown owner to
            # close/reap the muxer, probe the candidate, and install or unlink
            # it. A writer already blocked on a full pipe must yield first.
            self._writer_deadline = close_started + self.config.close_timeout * 0.25
            self._close_requested.set()
            writer = self._writer
            if writer is not None:
                writer.join(timeout=max(0.0, self._remaining()))
            if writer is not None and writer.is_alive():
                self._set_failed("recorder_thread_cleanup_failed")
                self._coordinate_shutdown("close")
                writer.join(timeout=max(0.0, self._remaining()))
            elif not self._shutdown_done.is_set():
                self._coordinate_shutdown("close")
            if writer is not None and writer.is_alive():
                self._set_failed("recorder_thread_cleanup_failed")
            self._closed = True
            if self.state not in (Mp4RecorderState.READY, Mp4RecorderState.FAILED):
                self._set_state(Mp4RecorderState.CLOSED)
            return self.result()

    def result(self) -> Mp4RecorderResult:
        probe = self._probe_result
        return Mp4RecorderResult(
            state=self.state,
            output_path=str(self.config.output_path) if self.state == Mp4RecorderState.READY else "",
            actual_fps=probe.actual_fps if probe else None,
            duration=probe.duration if probe else None,
            failure_reason=self.failure_reason,
        )

    def _writer_main(self) -> None:
        try:
            while True:
                if self._close_requested.is_set() and self._queue.empty():
                    break
                if (
                    self._close_requested.is_set()
                    and self._writer_deadline is not None
                    and self._monotonic() >= self._writer_deadline
                ):
                    raise Mp4RecorderError("recorder_write_timeout")
                try:
                    data = self._queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                try:
                    self._write_packet(data)
                finally:
                    self._queue.task_done()
                if self.state == Mp4RecorderState.FAILED:
                    break
        except Exception as exc:
            reason = str(exc) if isinstance(exc, Mp4RecorderError) else "recorder_pipe_failed"
            self._set_failed(reason)
        finally:
            self._accepting = False
            self._coordinate_shutdown("writer")

    def _write_packet(self, data: bytes) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise Mp4RecorderError("recorder_pipe_failed")
        fd = process.stdin.fileno()
        offset = 0
        deadline = self._monotonic() + self.config.write_timeout
        while offset < len(data):
            if process.poll() is not None:
                raise Mp4RecorderError("recorder_pipe_failed")
            effective_deadline = deadline
            if self._close_requested.is_set() and self._writer_deadline is not None:
                effective_deadline = min(effective_deadline, self._writer_deadline)
            remaining = effective_deadline - self._monotonic()
            if remaining <= 0:
                raise Mp4RecorderError("recorder_write_timeout")
            if not select.select([], [fd], [], min(remaining, 0.02))[1]:
                continue
            try:
                written = os.write(fd, data[offset:])
            except BlockingIOError:
                continue
            if written <= 0:
                raise Mp4RecorderError("recorder_pipe_failed")
            offset += written

    def _coordinate_shutdown(self, owner: str) -> None:
        deadline = self._close_deadline or (
            self._monotonic() + self.config.close_timeout
        )
        acquired = self._shutdown_lock.acquire(timeout=max(0.0, deadline - self._monotonic()))
        if not acquired:
            self._set_failed("recorder_process_cleanup_failed")
            return
        try:
            if self._shutdown_done.is_set():
                return
            with self._state_lock:
                self._shutdown_owner = owner
            process = self._process
            if process is None:
                self._cleanup_partial_once()
                return
            if self.state != Mp4RecorderState.FAILED:
                self._set_state(Mp4RecorderState.FINALIZING)
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
            except OSError:
                self._set_failed("recorder_pipe_failed")
            returncode = self._wait_or_escalate(process, deadline)
            if returncode is None:
                self._set_failed("recorder_process_cleanup_failed")
            elif returncode != 0:
                self._set_failed("recorder_finalize_failed")
            if self.state != Mp4RecorderState.FAILED:
                self._set_state(Mp4RecorderState.PROBING)
                try:
                    self._probe_result = self._probe(
                        self.config.ffprobe_path,
                        self.config.partial_path,
                        self.config.width,
                        self.config.height,
                        self.config.fps,
                        fps_tolerance=self.config.fps_tolerance,
                        timeout=min(self.config.probe_timeout, max(0.001, deadline - self._monotonic())),
                    )
                    self._install_final()
                    self._set_state(Mp4RecorderState.READY)
                except Exception:
                    self._set_failed("recorder_probe_failed")
            if self.state != Mp4RecorderState.READY:
                self._cleanup_partial_once()
        finally:
            self._shutdown_done.set()
            self._shutdown_lock.release()

    def _wait_or_escalate(self, process, deadline: float):
        returncode = self._poll_until(
            process,
            min(deadline, self._monotonic() + max(0.0, deadline - self._monotonic()) * 0.6),
        )
        if returncode is not None:
            return self._reap_once(process, deadline)
        try:
            process.terminate()
        except OSError:
            pass
        returncode = self._poll_until(
            process,
            min(deadline, self._monotonic() + max(0.0, deadline - self._monotonic()) * 0.5),
        )
        if returncode is not None:
            return self._reap_once(process, deadline)
        try:
            process.kill()
        except OSError:
            pass
        return self._reap_once(process, deadline)

    def _poll_until(self, process, deadline: float):
        while self._monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                return returncode
            time.sleep(min(0.01, max(0.0, deadline - self._monotonic())))
        return process.poll()

    def _reap_once(self, process, deadline: float):
        try:
            return process.wait(timeout=max(0.001, deadline - self._monotonic()))
        except subprocess.TimeoutExpired:
            return None

    def _install_final(self) -> None:
        partial = self.config.partial_path
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(str(partial), str(self.config.output_path))
        directory_fd = os.open(str(self.config.output_path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _cleanup_partial_once(self) -> None:
        if self.state == Mp4RecorderState.READY:
            return
        try:
            self.config.partial_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            self._set_failed("recorder_partial_cleanup_failed")

    def _remaining(self) -> float:
        if self._close_deadline is None:
            return self.config.close_timeout
        return self._close_deadline - self._monotonic()

    def _set_state(self, state: Mp4RecorderState) -> None:
        with self._state_lock:
            if self._state == Mp4RecorderState.READY and state != Mp4RecorderState.READY:
                return
            self._state = state

    def _set_failed(self, reason: str) -> None:
        with self._state_lock:
            if self._state == Mp4RecorderState.READY:
                return
            if self._failure_reason is None:
                self._failure_reason = reason
            self._state = Mp4RecorderState.FAILED


class RecordingFanoutSink:
    """Write primary live output first and offer recorder data without blocking."""

    def __init__(
        self, primary, recorder: Optional[Mp4Recorder], diagnostic_callback=None
    ) -> None:
        self.primary = primary
        self.recorder = recorder
        self._diagnostic_callback = diagnostic_callback
        self.recorder_diagnostic: Optional[str] = None
        self._diagnostic_reported = False
        self._closed = False

    def start_recorder_best_effort(self) -> bool:
        if self.recorder is None:
            return False
        try:
            self.recorder.start()
            return True
        except Exception:
            self.recorder_diagnostic = (
                getattr(self.recorder, "failure_reason", None)
                or "recorder_start_unexpected"
            )
            self._record_diagnostic(self.recorder_diagnostic)
            return False

    def write_packet(self, packet) -> None:
        self.primary.write_packet(packet)
        if self.recorder is None:
            return
        try:
            if not self.recorder.offer(packet.data):
                self.recorder_diagnostic = self.recorder.failure_reason
                if self.recorder_diagnostic:
                    self._record_diagnostic(self.recorder_diagnostic)
        except Exception:
            self.recorder_diagnostic = "recorder_offer_unexpected"
            self._record_diagnostic(self.recorder_diagnostic)

    def close(self) -> None:
        if self._closed:
            return
        primary_error = None
        try:
            self.primary.close()
        except Exception as exc:
            primary_error = exc
        finally:
            if self.recorder is not None:
                try:
                    result = self.recorder.close()
                    self.recorder_diagnostic = result.failure_reason
                except Exception:
                    self.recorder_diagnostic = "recorder_close_unexpected"
                    self._record_diagnostic(self.recorder_diagnostic)
            self._closed = True
        if primary_error is not None:
            raise primary_error

    def _record_diagnostic(self, reason: str) -> None:
        try:
            if self.recorder is not None:
                self.recorder.record_external_failure(reason)
        except Exception:
            pass
        try:
            if self._diagnostic_callback is not None and not self._diagnostic_reported:
                self._diagnostic_reported = True
                self._diagnostic_callback(reason)
        except Exception:
            pass
