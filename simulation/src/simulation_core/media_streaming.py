"""MediaMTX RTSP publishing boundary for Phase 6.4.

The publisher treats encoder output as an opaque, continuous Annex-B stream.
FFmpeg only muxes the stream into RTSP; it never decodes or re-encodes video.
"""

from dataclasses import dataclass
from enum import Enum
import ipaddress
import math
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Callable, Mapping, Optional
from urllib.parse import urlsplit

from .video_encoder import EncodedPacket, VideoEncoderError


_STREAM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class MediaRuntimeConfig:
    """Deployment-owned addresses used by one Simulation training child."""

    internal_rtsp_base: str
    public_playback_base: str
    ffmpeg_path: Path
    allow_loopback_playback: bool = False

    def __post_init__(self) -> None:
        rtsp = _clean_base(self.internal_rtsp_base, {"rtsp"}, "internal RTSP base")
        if not _is_loopback(rtsp.hostname):
            raise ValueError("internal RTSP base must use loopback")
        public = _clean_base(
            self.public_playback_base, {"http", "https"}, "public playback base"
        )
        if _is_loopback(public.hostname) and not self.allow_loopback_playback:
            raise ValueError("public playback base must not use loopback")
        object.__setattr__(self, "internal_rtsp_base", self.internal_rtsp_base.rstrip("/"))
        object.__setattr__(self, "public_playback_base", self.public_playback_base.rstrip("/"))
        object.__setattr__(self, "ffmpeg_path", Path(self.ffmpeg_path))

    @classmethod
    def from_environment(cls, values: Optional[Mapping[str, str]] = None):
        source = os.environ if values is None else values
        required = (
            "SIMULATION_STREAM_RTSP_BASE",
            "SIMULATION_STREAM_PUBLIC_BASE_URL",
            "SIMULATION_FFMPEG_PATH",
        )
        missing = [key for key in required if not str(source.get(key, "")).strip()]
        if missing:
            raise ValueError("missing video configuration: %s" % ",".join(missing))
        return cls(
            str(source[required[0]]),
            str(source[required[1]]),
            Path(str(source[required[2]])),
            allow_loopback_playback=(
                source.get("SIMULATION_STREAM_ALLOW_LOOPBACK_PLAYBACK") == "1"
            ),
        )


def _clean_base(value: str, schemes, label: str):
    parsed = urlsplit(value)
    if (
        parsed.scheme not in schemes
        or not parsed.hostname
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("%s must be a credential-free origin with an explicit port" % label)
    return parsed


def _is_loopback(hostname: Optional[str]) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class VideoStreamState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class MediaMtxPublisherConfig:
    stream_id: str
    ffmpeg_path: Path = Path("/opt/ffmpeg/bin/ffmpeg")
    rtsp_host: str = "127.0.0.1"
    rtsp_port: int = 18554
    fps: float = 15.0
    close_timeout: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not _STREAM_ID.fullmatch(self.stream_id):
            raise ValueError("stream_id must be a URL-safe single path segment")
        if not isinstance(self.rtsp_host, str) or not self.rtsp_host:
            raise ValueError("rtsp_host is required")
        if isinstance(self.rtsp_port, bool) or not isinstance(self.rtsp_port, int) or not 1 <= self.rtsp_port <= 65535:
            raise ValueError("rtsp_port must be in 1..65535")
        if isinstance(self.fps, bool) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not math.isfinite(self.close_timeout) or self.close_timeout <= 0:
            raise ValueError("close_timeout must be finite and positive")
        object.__setattr__(self, "ffmpeg_path", Path(self.ffmpeg_path))

    @property
    def rtsp_url(self) -> str:
        return "rtsp://%s:%d/%s" % (self.rtsp_host, self.rtsp_port, self.stream_id)


@dataclass(frozen=True)
class FakeStreamDescriptor:
    stream_id: str
    run_id: str
    task_id: str
    camera_id: str
    webrtc_url: str

    def __post_init__(self) -> None:
        if not _STREAM_ID.fullmatch(self.stream_id or ""):
            raise ValueError("invalid stream_id")
        for name in ("run_id", "task_id", "camera_id", "webrtc_url"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError("%s is required" % name)


class MediaMtxPublisherSink:
    """Write every Annex-B byte to an FFmpeg RTSP copy/remux process."""

    def __init__(self, config: MediaMtxPublisherConfig,
                 process_factory: Callable = subprocess.Popen) -> None:
        self.config = config
        self._process_factory = process_factory
        self._process = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_lines = []
        self._closed = False
        self.state = VideoStreamState.CREATED
        self.bytes_written = 0
        self.chunks_written = 0

    @property
    def command(self):
        # -c:v copy is the essential contract: the second FFmpeg is a muxer only.
        return [
            str(self.config.ffmpeg_path), "-hide_banner", "-loglevel", "warning",
            "-fflags", "+genpts", "-r", str(self.config.fps),
            "-f", "h264", "-i", "pipe:0", "-an", "-c:v", "copy",
            "-f", "rtsp", "-rtsp_transport", "tcp", self.config.rtsp_url,
        ]

    @property
    def diagnostics(self) -> str:
        return "\n".join(self._stderr_lines)

    def open(self) -> None:
        if self._closed:
            raise VideoEncoderError("RTSP publisher is closed")
        if self._process is not None:
            return
        if not self.config.ffmpeg_path.is_file():
            raise VideoEncoderError("FFmpeg executable does not exist: %s" % self.config.ffmpeg_path)
        self.state = VideoStreamState.STARTING
        self._process = self._process_factory(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="rtsp-publisher-stderr", daemon=True,
        )
        self._stderr_thread.start()

    def write_packet(self, packet: EncodedPacket) -> None:
        if packet.codec != "h264" or packet.bitstream_format != "annex-b":
            raise ValueError("publisher requires H.264 Annex-B")
        if self._process is None:
            self.open()
        if self._process.poll() is not None:
            self.state = VideoStreamState.ERROR
            raise VideoEncoderError(self._failure("RTSP publisher exited"))
        try:
            self._process.stdin.write(packet.data)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.state = VideoStreamState.ERROR
            raise VideoEncoderError(self._failure("RTSP publisher pipe failed: %s" % exc)) from exc
        self.bytes_written += len(packet.data)
        self.chunks_written += 1
        self.state = VideoStreamState.READY

    def flush(self) -> None:
        if self._process is not None and self._process.stdin and not self._process.stdin.closed:
            self._process.stdin.flush()

    def close(self) -> None:
        if self._closed:
            return
        self.state = VideoStreamState.STOPPING
        process = self._process
        try:
            if process is not None:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    try:
                        process.wait(timeout=self.config.close_timeout)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                if process.stderr is not None:
                    process.stderr.close()
        finally:
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=1)
            self._process = None
            self._closed = True
            self.state = VideoStreamState.STOPPED

    def _drain_stderr(self) -> None:
        for line in iter(self._process.stderr.readline, b""):
            self._stderr_lines.append(line.decode("utf-8", "replace").rstrip())

    def _failure(self, prefix: str) -> str:
        details = " | ".join(self._stderr_lines[-20:])
        return "%s%s" % (prefix, ": " + details if details else "")
