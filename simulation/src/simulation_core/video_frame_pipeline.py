"""CPU video-frame pipeline between Camera Sensors and future encoders."""

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
import math
import threading
import time
from typing import Any, Callable, Deque, Dict, Optional, Protocol

import numpy as np

from .isaac_gym_camera import CameraFrame, RGBA8


RGB24 = "RGB24"
BGRA8 = "BGRA8"


class FramePipelineError(RuntimeError):
    pass


class PixelFormat(str, Enum):
    RGBA8 = RGBA8
    RGB24 = RGB24
    BGRA8 = BGRA8


class FrameStorage(str, Enum):
    CPU = "cpu"
    GPU = "gpu"  # Reserved capability; VideoFrame currently accepts CPU only.


class DropPolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"


_CHANNELS = {PixelFormat.RGBA8: 4, PixelFormat.RGB24: 3, PixelFormat.BGRA8: 4}


@dataclass(frozen=True)
class VideoFrame:
    run_id: str
    task_id: str
    camera_id: str
    frame_index: int
    captured_at: datetime
    presentation_timestamp: float
    width: int
    height: int
    pixel_format: PixelFormat
    data: np.ndarray
    storage: FrameStorage = FrameStorage.CPU

    def __post_init__(self) -> None:
        for name in ("run_id", "task_id", "camera_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("%s must be a non-empty string" % name)
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        if not math.isfinite(self.presentation_timestamp) or self.presentation_timestamp < 0:
            raise ValueError("presentation_timestamp must be finite and non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        try:
            pixel_format = PixelFormat(self.pixel_format)
            storage = FrameStorage(self.storage)
        except ValueError as exc:
            raise ValueError("unsupported pixel format or frame storage") from exc
        object.__setattr__(self, "pixel_format", pixel_format)
        object.__setattr__(self, "storage", storage)
        if storage != FrameStorage.CPU:
            raise ValueError("GPU-backed VideoFrame is reserved for Phase 6.3")
        if not isinstance(self.data, np.ndarray) or self.data.dtype != np.uint8:
            raise ValueError("CPU frame data must be a numpy uint8 array")
        expected = (self.height, self.width, _CHANNELS[pixel_format])
        if self.data.shape != expected:
            raise ValueError("frame shape must be %r, got %r" % (expected, self.data.shape))
        if not self.data.flags.c_contiguous:
            raise ValueError("frame data must be C-contiguous")

    def metadata(self) -> Dict[str, Any]:
        """Return JSON-compatible metadata without pixel/runtime storage."""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "captured_at": self.captured_at.isoformat(),
            "presentation_timestamp": self.presentation_timestamp,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format.value,
            "storage": self.storage.value,
        }


class FrameConverter:
    """Explicit, non-reshaping CPU pixel format conversion."""

    @staticmethod
    def convert(frame: VideoFrame, output_format: PixelFormat) -> VideoFrame:
        output_format = PixelFormat(output_format)
        source = frame.pixel_format
        if source == output_format:
            data = np.ascontiguousarray(frame.data)
        elif source == PixelFormat.RGBA8 and output_format == PixelFormat.RGB24:
            data = np.ascontiguousarray(frame.data[:, :, :3])
        elif source == PixelFormat.RGBA8 and output_format == PixelFormat.BGRA8:
            data = np.ascontiguousarray(frame.data[:, :, (2, 1, 0, 3)])
        else:
            raise ValueError("unsupported conversion: %s -> %s" % (source.value, output_format.value))
        return replace(frame, pixel_format=output_format, data=data)


class FrameRateLimiter:
    """Non-blocking monotonic sampling policy; it never sleeps."""

    def __init__(self, target_fps: float = 15.0, clock: Callable[[], float] = None) -> None:
        if isinstance(target_fps, bool) or not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be finite and positive")
        self.target_fps = float(target_fps)
        self.interval = 1.0 / self.target_fps
        self._clock = clock or time.monotonic
        self._next_capture_at = None

    def should_capture(self, now: Optional[float] = None) -> bool:
        current = self._clock() if now is None else float(now)
        if not math.isfinite(current):
            raise ValueError("monotonic time must be finite")
        if self._next_capture_at is None:
            self._next_capture_at = current + self.interval
            return True
        if current + 1e-12 < self._next_capture_at:
            return False
        missed = max(1, int((current - self._next_capture_at) // self.interval) + 1)
        self._next_capture_at += missed * self.interval
        return True


@dataclass(frozen=True)
class FramePipelineStats:
    produced_frames: int = 0
    accepted_frames: int = 0
    dropped_frames: int = 0
    consumed_frames: int = 0
    queue_depth: int = 0
    last_frame_index: Optional[int] = None
    average_capture_interval: Optional[float] = None
    average_conversion_seconds: Optional[float] = None


class FrameConsumer(Protocol):
    def consume(self, frame: VideoFrame) -> None: ...


class RecordingFrameConsumer:
    """Test/smoke consumer that stores metadata only, never image data."""

    def __init__(self) -> None:
        self.records = []

    def consume(self, frame: VideoFrame) -> None:
        self.records.append({
            "frame_index": frame.frame_index,
            "presentation_timestamp": frame.presentation_timestamp,
            "pixel_format": frame.pixel_format.value,
            "shape": frame.data.shape,
        })


class FramePipeline:
    """Convert CameraFrames and retain a bounded, low-latency frame queue."""

    def __init__(
        self,
        run_id: str,
        task_id: str,
        output_format: PixelFormat = PixelFormat.RGB24,
        max_frames: int = 3,
        drop_policy: DropPolicy = DropPolicy.DROP_OLDEST,
        target_fps: float = 15.0,
        monotonic_clock: Callable[[], float] = None,
    ) -> None:
        if not run_id or not task_id:
            raise ValueError("run_id and task_id are required")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self.run_id, self.task_id = run_id, task_id
        self.output_format = PixelFormat(output_format)
        self.max_frames = max_frames
        self.drop_policy = DropPolicy(drop_policy)
        self._clock = monotonic_clock or time.monotonic
        self._started_at = None
        self.limiter = FrameRateLimiter(target_fps, self._clock)
        self._queue: Deque[VideoFrame] = deque()
        self._lock = threading.Lock()
        self._produced = self._accepted = self._dropped = self._consumed = 0
        self._last_index = None
        self._last_pts = None
        self._last_capture_time = None
        self._interval_total = 0.0
        self._interval_count = 0
        self._conversion_total = 0.0
        self._conversion_count = 0

    def maybe_produce(self, producer: Callable[[], CameraFrame], now: float = None):
        current = self._clock() if now is None else float(now)
        if not self.limiter.should_capture(current):
            return None
        return self.push_camera_frame(producer(), monotonic_time=current)

    def push_camera_frame(self, frame: CameraFrame, monotonic_time: float = None):
        current = self._clock() if monotonic_time is None else float(monotonic_time)
        if self._started_at is None:
            self._started_at = current
        pts = max(0.0, current - self._started_at)
        base = VideoFrame(
            self.run_id, self.task_id, frame.camera_id, frame.frame_index,
            frame.captured_at, pts, frame.width, frame.height,
            PixelFormat.RGBA8, np.ascontiguousarray(frame.data),
        )
        conversion_started = time.perf_counter()
        converted = FrameConverter.convert(base, self.output_format)
        conversion_elapsed = time.perf_counter() - conversion_started
        with self._lock:
            self._conversion_total += conversion_elapsed
            self._conversion_count += 1
        self.push(converted, capture_time=current)
        return converted

    def push(self, frame: VideoFrame, capture_time: float = None) -> bool:
        with self._lock:
            if self._last_index is not None and frame.frame_index <= self._last_index:
                raise ValueError("frame_index must be strictly increasing")
            if self._last_pts is not None and frame.presentation_timestamp < self._last_pts:
                raise ValueError("presentation timestamp must be monotonic")
            self._produced += 1
            self._last_index = frame.frame_index
            self._last_pts = frame.presentation_timestamp
            current = self._clock() if capture_time is None else capture_time
            if self._last_capture_time is not None:
                self._interval_total += current - self._last_capture_time
                self._interval_count += 1
            self._last_capture_time = current
            if len(self._queue) >= self.max_frames:
                self._dropped += 1
                if self.drop_policy == DropPolicy.DROP_NEWEST:
                    return False
                self._queue.popleft()
            self._queue.append(frame)
            self._accepted += 1
            return True

    def poll(self) -> Optional[VideoFrame]:
        with self._lock:
            if not self._queue:
                return None
            self._consumed += 1
            return self._queue.popleft()

    def consume_next(self, consumer: FrameConsumer) -> Optional[VideoFrame]:
        frame = self.poll()
        if frame is not None:
            consumer.consume(frame)
        return frame

    @property
    def stats(self) -> FramePipelineStats:
        with self._lock:
            average = self._interval_total / self._interval_count if self._interval_count else None
            conversion_average = (
                self._conversion_total / self._conversion_count
                if self._conversion_count else None
            )
            return FramePipelineStats(
                self._produced, self._accepted, self._dropped, self._consumed,
                len(self._queue), self._last_index, average, conversion_average,
            )
