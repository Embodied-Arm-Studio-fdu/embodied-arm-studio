"""H.264 encoder ports, FFmpeg NVENC adapter, and worker lifecycle."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import select
import shlex
import subprocess
import threading
import time
from typing import Callable, List, Optional, Protocol, Sequence

import numpy as np

from .video_frame_pipeline import FramePipeline, PixelFormat, VideoFrame


class VideoEncoderError(RuntimeError):
    pass


class VideoEncoderStateError(VideoEncoderError):
    pass


@dataclass(frozen=True)
class EncoderConfig:
    codec: str = "h264"
    width: int = 640
    height: int = 480
    fps: float = 15.0
    bitrate: int = 2_000_000
    gop_size: int = 30
    pixel_format: PixelFormat = PixelFormat.BGRA8
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.codec != "h264":
            raise ValueError("only h264 is supported")
        for name in ("width", "height", "bitrate", "gop_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if isinstance(self.fps, bool) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        pixel_format = PixelFormat(self.pixel_format)
        object.__setattr__(self, "pixel_format", pixel_format)
        if pixel_format != PixelFormat.BGRA8:
            raise ValueError("current NVENC helper input must be BGRA8")
        if not self.device.startswith("cuda:") or not self.device[5:].isdigit():
            raise ValueError("device must use cuda:<non-negative-index>")

    @property
    def gpu_index(self) -> int:
        return int(self.device.split(":", 1)[1])


@dataclass(frozen=True)
class EncodedPacket:
    run_id: str
    task_id: str
    camera_id: str
    frame_index: int
    pts: float
    dts: Optional[float]
    is_keyframe: bool
    contains_codec_config: bool
    codec: str
    width: int
    height: int
    fps: float
    data: bytes
    bitstream_format: str = "annex-b"

    def __post_init__(self) -> None:
        for name in ("run_id", "task_id", "camera_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError("%s is required" % name)
        if self.frame_index < 0 or not math.isfinite(self.pts) or self.pts < 0:
            raise ValueError("frame_index and pts must be non-negative")
        if self.dts is not None and not math.isfinite(self.dts):
            raise ValueError("dts must be finite or None")
        if self.codec != "h264" or self.bitstream_format != "annex-b":
            raise ValueError("packet must be H.264 Annex-B")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("packet video dimensions/fps are invalid")
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("encoded packet data must be non-empty bytes")


class VideoEncoder(Protocol):
    def open(self, config: EncoderConfig) -> None: ...
    def encode(self, frame: VideoFrame) -> Sequence[EncodedPacket]: ...
    def flush(self) -> Sequence[EncodedPacket]: ...
    def close(self) -> None: ...


class EncodedVideoSink(Protocol):
    def write_packet(self, packet: EncodedPacket) -> None: ...
    def close(self) -> None: ...


class EncoderInputConverter:
    @staticmethod
    def to_bgra(frame: VideoFrame) -> np.ndarray:
        if frame.pixel_format == PixelFormat.BGRA8:
            data = frame.data
        elif frame.pixel_format == PixelFormat.RGBA8:
            data = frame.data[:, :, (2, 1, 0, 3)]
        elif frame.pixel_format == PixelFormat.RGB24:
            alpha = np.full((frame.height, frame.width, 1), 255, dtype=np.uint8)
            data = np.concatenate((frame.data[:, :, (2, 1, 0)], alpha), axis=2)
        else:
            raise ValueError("unsupported encoder input pixel format")
        result = np.ascontiguousarray(data, dtype=np.uint8)
        if result.shape != (frame.height, frame.width, 4):
            raise ValueError("BGRA encoder input has invalid shape")
        return result


def build_nvenc_helper(output_path: Path, source_path: Path = None) -> Path:
    source = source_path or Path(__file__).with_name("nvenc_helper.c")
    output = Path(output_path)
    flags = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "libavcodec", "libavutil", "libswscale"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if flags.returncode != 0:
        raise VideoEncoderError("FFmpeg development metadata unavailable: %s" % flags.stderr.strip())
    command = ["cc", "-O2", str(source), "-o", str(output)] + shlex.split(flags.stdout)
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise VideoEncoderError("failed to build NVENC helper: %s" % result.stderr.strip())
    return output


def _contains_h264_config(data: bytes) -> bool:
    return _contains_h264_nal(data, 7) or _contains_h264_nal(data, 8)


def _contains_h264_nal(data: bytes, nal_type: int) -> bool:
    index = 0
    while index + 4 <= len(data):
        if data[index:index + 4] == b"\x00\x00\x00\x01":
            nal = index + 4; index = nal
        elif data[index:index + 3] == b"\x00\x00\x01":
            nal = index + 3; index = nal
        else:
            index += 1; continue
        if nal < len(data) and data[nal] & 0x1F == nal_type:
            return True
    return False


class FFmpegNvencEncoder:
    """Online BGRA -> h264_nvenc adapter using an explicit FFmpeg executable.

    stdout is an Annex-B byte stream, so returned objects are byte chunks rather
    than assumed AVPacket/frame boundaries.  Every byte is forwarded to the sink.
    """

    def __init__(self, ffmpeg_path: Path, startup_timeout: float = 10.0) -> None:
        self.ffmpeg_path = Path(ffmpeg_path)
        self.startup_timeout = startup_timeout
        self.config = None
        self._process = None
        self._stderr_lines = []
        self._stderr_thread = None
        self._last_frame = None
        self._stdin_closed = False
        self._closed = False

    def open(self, config: EncoderConfig) -> None:
        if self._closed:
            raise VideoEncoderStateError("encoder is closed")
        if self._process is not None:
            return
        if not self.ffmpeg_path.is_file():
            raise VideoEncoderError("FFmpeg executable does not exist: %s" % self.ffmpeg_path)
        self.config = config
        args = [
            str(self.ffmpeg_path), "-hide_banner", "-loglevel", "verbose",
            "-f", "rawvideo", "-pixel_format", "bgra",
            "-video_size", "%dx%d" % (config.width, config.height),
            "-framerate", str(config.fps), "-i", "pipe:0", "-an",
            "-c:v", "h264_nvenc", "-gpu", str(config.gpu_index),
            "-preset", "p4", "-tune", "ll", "-rc", "cbr",
            "-b:v", str(config.bitrate), "-g", str(config.gop_size),
            "-bf", "0", "-delay", "0", "-zerolatency", "1",
            # A short GOP bounds mid-stream reader wait; forced keyframes are
            # IDR. This FFmpeg build's h264_nvenc has no repeat_headers AVOption.
            "-forced-idr", "1", "-aud", "1",
            "-f", "h264", "pipe:1",
        ]
        self._process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        os.set_blocking(self._process.stdout.fileno(), False)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def encode(self, frame: VideoFrame) -> Sequence[EncodedPacket]:
        self._require_open()
        if (frame.width, frame.height) != (self.config.width, self.config.height):
            raise ValueError("frame resolution does not match encoder config")
        bgra = EncoderInputConverter.to_bgra(frame)
        self._last_frame = frame
        try:
            self._process.stdin.write(bgra.tobytes())
            self._process.stdin.flush()
            packets = self._read_available(frame)
            if self._process.poll() not in (None, 0):
                raise VideoEncoderError(self._failure_message("FFmpeg NVENC process exited"))
            return packets
        except (BrokenPipeError, OSError) as exc:
            raise VideoEncoderError(self._failure_message("FFmpeg NVENC pipe failed: %s" % exc)) from exc

    def flush(self) -> Sequence[EncodedPacket]:
        self._require_open()
        if not self._stdin_closed:
            self._process.stdin.close()
            self._stdin_closed = True
        os.set_blocking(self._process.stdout.fileno(), True)
        data = self._process.stdout.read()
        packets = self._packets_for_chunk(data, self._last_frame) if data else []
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            self._process.terminate()
            raise VideoEncoderError("FFmpeg NVENC flush timed out") from exc
        if self._process.returncode != 0:
            raise VideoEncoderError(self._failure_message("FFmpeg NVENC flush failed"))
        return packets

    def close(self) -> None:
        if self._closed:
            return
        process = self._process
        try:
            if process is not None:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    try: process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try: process.wait(timeout=2)
                        except subprocess.TimeoutExpired: process.kill(); process.wait()
                for stream in (process.stdout, process.stderr):
                    if stream is not None: stream.close()
        finally:
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1)
            self._process = None; self._closed = True

    @property
    def diagnostics(self) -> str:
        return "\n".join(self._stderr_lines)

    def _read_available(self, frame: VideoFrame) -> List[EncodedPacket]:
        chunks = []
        while select.select([self._process.stdout], [], [], 0)[0]:
            try:
                chunk = os.read(self._process.stdout.fileno(), 65536)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.extend(self._packets_for_chunk(chunk, frame))
        return chunks

    def _packets_for_chunk(self, data: bytes, frame: Optional[VideoFrame]) -> List[EncodedPacket]:
        if not data:
            return []
        if frame is None:
            raise VideoEncoderError("encoded chunk has no source frame metadata")
        return [EncodedPacket(
            frame.run_id, frame.task_id, frame.camera_id, frame.frame_index,
            frame.presentation_timestamp, None, _contains_h264_nal(data, 5),
            _contains_h264_config(data), "h264", frame.width, frame.height,
            self.config.fps, bytes(data),
        )]

    def _require_open(self):
        if self._closed or self._process is None:
            raise VideoEncoderStateError("encoder is not open")

    def _drain_stderr(self):
        for line in iter(self._process.stderr.readline, b""):
            self._stderr_lines.append(line.decode("utf-8", "replace").rstrip())

    def _collect_remaining_stderr(self):
        if self._process and self._process.stderr:
            remainder = self._process.stderr.read().decode("utf-8", "replace")
            self._stderr_lines.extend(item for item in remainder.splitlines() if item)

    def _failure_message(self, prefix: str) -> str:
        return "%s: %s" % (prefix, " | ".join(self._stderr_lines[-20:]))


class FakeVideoEncoder:
    def __init__(self, fail_at: int = None) -> None:
        self.config = None; self.closed = False; self.flushed = False
        self.fail_at = fail_at; self.input_frames = 0

    def open(self, config): self.config = config
    def encode(self, frame):
        if self.fail_at is not None and self.input_frames == self.fail_at:
            raise VideoEncoderError("fake encode failure")
        self.input_frames += 1
        return [EncodedPacket(frame.run_id, frame.task_id, frame.camera_id,
            frame.frame_index, frame.presentation_timestamp, None,
            frame.frame_index == 0, frame.frame_index == 0, "h264",
            frame.width, frame.height, self.config.fps, b"\x00\x00\x00\x01\x65fake")]
    def flush(self): self.flushed = True; return []
    def close(self): self.closed = True


class RecordingEncodedSink:
    def __init__(self, fail_at: int = None):
        self.packets = []; self.closed = False; self.fail_at = fail_at
    def write_packet(self, packet):
        if self.fail_at is not None and len(self.packets) == self.fail_at:
            raise VideoEncoderError("fake sink failure")
        self.packets.append(packet)
    def close(self): self.closed = True


class AnnexBFileSink:
    def __init__(self, path: Path): self.path = Path(path); self._file = None
    def write_packet(self, packet):
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("wb")
        self._file.write(packet.data)
    def close(self):
        if self._file is not None: self._file.close(); self._file = None


@dataclass(frozen=True)
class EncoderStats:
    input_frames: int = 0; encoded_frames: int = 0; encoded_packets: int = 0
    dropped_frames: int = 0; keyframes: int = 0; encoded_bytes: int = 0
    encode_errors: int = 0; average_encode_ms: float = 0.0
    effective_fps: float = 0.0; approximate_bitrate: float = 0.0


class EncoderWorker:
    """Consumer-side thread; producer only writes the bounded FramePipeline."""
    def __init__(self, pipeline: FramePipeline, encoder: VideoEncoder,
                 config: EncoderConfig, sink: EncodedVideoSink,
                 error_callback=None) -> None:
        self.pipeline = pipeline; self.encoder = encoder; self.config = config; self.sink = sink
        self._error_callback = error_callback
        self.error = None; self._thread = None; self._stop = threading.Event()
        self._input = self._frames = self._packets = self._keys = self._bytes = self._errors = 0
        self._encode_seconds = 0.0; self._started = None; self._closed = False

    def start(self):
        self.encoder.open(self.config)
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="video-encoder", daemon=True)
        self._thread.start()

    def close(self):
        if self._closed: return
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=10)
        try:
            if self.error is None:
                self._write(self.encoder.flush())
        except Exception as exc:
            self._record_error(exc)
        finally:
            self.encoder.close(); self.sink.close(); self._closed = True

    def _run(self):
        try:
            while not self._stop.is_set() or self.pipeline.stats.queue_depth:
                frame = self.pipeline.poll()
                if frame is None:
                    self._stop.wait(0.002); continue
                self._input += 1; started = time.perf_counter()
                packets = self.encoder.encode(frame)
                self._encode_seconds += time.perf_counter() - started
                self._frames += 1; self._write(packets)
        except Exception as exc:
            self._record_error(exc); self._stop.set()

    def set_error_callback(self, callback):
        self._error_callback = callback
        if self.error is not None and callback is not None:
            self._notify_error(callback, self.error)

    def _record_error(self, error):
        self.error = error
        self._errors += 1
        if self._error_callback is not None:
            self._notify_error(self._error_callback, error)

    @staticmethod
    def _notify_error(callback, error):
        try:
            callback(error)
        except Exception:
            # Reporting failure must not escape the video-only boundary.
            pass

    def _write(self, packets):
        for packet in packets:
            self.sink.write_packet(packet); self._packets += 1
            self._keys += int(packet.is_keyframe); self._bytes += len(packet.data)

    @property
    def stats(self):
        elapsed = max(0.0, time.monotonic() - self._started) if self._started else 0.0
        return EncoderStats(
            self._input, self._frames, self._packets, self.pipeline.stats.dropped_frames,
            self._keys, self._bytes, self._errors,
            self._encode_seconds * 1000 / self._frames if self._frames else 0.0,
            self._frames / elapsed if elapsed else 0.0,
            self._bytes * 8 / elapsed if elapsed else 0.0,
        )
