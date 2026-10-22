"""Optional Isaac Gym Camera Sensor bound to an existing simulation runtime."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Callable, Optional, Tuple

import numpy as np


RGBA8 = "RGBA8"


class CameraError(RuntimeError):
    pass


class CameraStateError(CameraError):
    pass


@dataclass(frozen=True)
class IsaacGymCameraConfig:
    camera_id: str = "overview"
    width: int = 640
    height: int = 480
    horizontal_fov: float = 60.0
    position: Tuple[float, float, float] = (1.45, -1.35, 1.35)
    target: Tuple[float, float, float] = (0.38, 0.02, 0.58)

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("camera_id must be a non-empty string")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if not math.isfinite(self.horizontal_fov) or not 0 < self.horizontal_fov < 180:
            raise ValueError("horizontal_fov must be between 0 and 180 degrees")
        for name in ("position", "target"):
            value = tuple(getattr(self, name))
            if len(value) != 3 or not all(math.isfinite(float(v)) for v in value):
                raise ValueError("%s must contain three finite values" % name)
            object.__setattr__(self, name, value)
        if self.position == self.target:
            raise ValueError("camera position and target must differ")


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    width: int
    height: int
    pixel_format: str
    frame_index: int
    captured_at: datetime
    data: np.ndarray

    def __post_init__(self) -> None:
        if self.pixel_format != RGBA8:
            raise ValueError("pixel_format must be RGBA8")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        if not isinstance(self.data, np.ndarray) or self.data.dtype != np.uint8:
            raise ValueError("RGBA frame data must be a numpy uint8 array")
        if self.data.shape != (self.height, self.width, 4):
            raise ValueError(
                "RGBA frame shape must be (%d, %d, 4), got %r"
                % (self.height, self.width, self.data.shape)
            )


class IsaacGymCamera:
    """Own only a Camera Sensor; gym/sim/env remain owned by the Push runtime."""

    def __init__(
        self,
        gym: Any,
        sim: Any,
        env: Any,
        gymapi: Any,
        gymtorch: Any,
        config: Optional[IsaacGymCameraConfig] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.gym, self.sim, self.env, self.gymapi, self.gymtorch = gym, sim, env, gymapi, gymtorch
        self.config = config or IsaacGymCameraConfig()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._handle = None
        self._gpu_tensor = None
        self._next_index = 0
        self._closed = False

    @property
    def handle(self):
        return self._handle

    def initialize(self) -> "IsaacGymCamera":
        if self._closed:
            raise CameraStateError("camera is closed")
        if self._handle is not None:
            return self
        properties = self.gymapi.CameraProperties()
        properties.width = self.config.width
        properties.height = self.config.height
        properties.horizontal_fov = self.config.horizontal_fov
        properties.enable_tensors = True
        handle = self.gym.create_camera_sensor(self.env, properties)
        if handle is None or handle < 0:
            raise CameraError("Isaac Gym failed to create camera sensor")
        self._handle = handle
        try:
            self.gym.set_camera_location(
                handle,
                self.env,
                self.gymapi.Vec3(*self.config.position),
                self.gymapi.Vec3(*self.config.target),
            )
            descriptor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.env, handle, self.gymapi.IMAGE_COLOR
            )
            self._gpu_tensor = self.gymtorch.wrap_tensor(descriptor)
            if self._gpu_tensor is None:
                raise CameraError("Isaac Gym failed to acquire camera GPU tensor")
        except Exception:
            self.close()
            raise
        return self

    def capture(self) -> CameraFrame:
        if self._closed:
            raise CameraStateError("camera is closed")
        if self._handle is None:
            raise CameraStateError("camera must be initialized before capture")
        # Physics is advanced by Push. This sequence publishes its latest state
        # to graphics, renders sensors, then performs the intentional CPU copy.
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        try:
            # Phase 6.1 intentionally materializes a detached CPU ndarray.
            pixels = self._gpu_tensor.detach().cpu().numpy().reshape(
                self.config.height, self.config.width, 4
            ).astype(np.uint8, copy=True)
        finally:
            self.gym.end_access_image_tensors(self.sim)
        frame = CameraFrame(
            camera_id=self.config.camera_id,
            width=self.config.width,
            height=self.config.height,
            pixel_format=RGBA8,
            frame_index=self._next_index,
            captured_at=self._clock(),
            data=pixels,
        )
        self._next_index += 1
        return frame

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._handle is not None:
                self.gym.destroy_camera_sensor(self.sim, self.env, self._handle)
        finally:
            self._handle = None
            self._gpu_tensor = None
            self._closed = True
