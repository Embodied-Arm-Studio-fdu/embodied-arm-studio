#!/usr/bin/env python3
"""Short real RTX/Isaac Gym/S1 Push overview-camera validation."""

import argparse
import faulthandler
import signal
from datetime import datetime, timezone
from pathlib import Path
import os
import time

import numpy as np
from PIL import Image

from simulation_core.isaac_gym_camera import IsaacGymCameraConfig
from simulation_core.push_adapter import ACTION_DIM, IsaacGymPushBackend, PushAdapter, PushConfig
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig


def validate(frame):
    if frame.data.dtype != np.uint8 or frame.data.shape != (frame.height, frame.width, 4):
        raise RuntimeError("invalid RGBA frame layout")
    if not frame.data.any():
        raise RuntimeError("camera frame is all zero")
    if np.unique(frame.data.reshape(-1, 4), axis=0).shape[0] <= 1:
        raise RuntimeError("camera frame is a single color")


def save(frame, path):
    Image.fromarray(frame.data, mode="RGBA").save(str(path))


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/phase-6-1"))
    parser.add_argument("--graphics-device-id", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.graphics_device_id < 0:
        parser.error("--graphics-device-id must be non-negative")
    # Preview 4's graphics runtime otherwise follows a stale remote DISPLAY on
    # this server and can block in create_sim. Camera sensors are Vulkan
    # offscreen rendering and do not require a viewer/X11.
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault(
        "VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json"
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.artifacts_root / run_id / "camera"
    output.mkdir(parents=True, exist_ok=False)
    camera_config = IsaacGymCameraConfig()
    push_config = PushConfig(max_steps=max(300, args.steps + 1))
    backend = IsaacGymPushBackend(
        push_config,
        s1_config=S1IsaacGymConfig(
            graphics_device_id=args.graphics_device_id, headless=True
        ),
        camera_config=camera_config,
        stage_callback=lambda stage: print("camera_setup_stage=%s" % stage, flush=True),
    )
    adapter = PushAdapter(push_config, backend=backend)
    frames = []
    started = time.perf_counter()
    try:
        print("camera_smoke_stage=initialize", flush=True)
        adapter.initialize()
        print("camera_smoke_stage=reset", flush=True)
        adapter.reset()
        print("camera_smoke_stage=capture_first", flush=True)
        frames.append(backend.capture_camera_frame())
        print("camera_smoke_stage=control_steps", flush=True)
        action = (0.002, 0.001)
        for index in range(args.steps):
            adapter.step(action)
            if index == args.steps // 2:
                print("camera_smoke_stage=capture_middle", flush=True)
                frames.append(backend.capture_camera_frame())
        print("camera_smoke_stage=capture_last", flush=True)
        frames.append(backend.capture_camera_frame())
        for frame in frames:
            validate(frame)
        difference = float(np.abs(
            frames[-1].data.astype(np.int16) - frames[0].data.astype(np.int16)
        ).mean())
        if difference <= 0.0:
            raise RuntimeError("camera image did not change after physics/control steps")
        names = ("frame_000.png", "frame_001.png", "frame_last.png")
        for frame, name in zip(frames, names):
            save(frame, output / name)
        elapsed = time.perf_counter() - started
        first = frames[0]
        print("CAMERA_SMOKE_RESULT=success")
        print("camera_id=%s" % first.camera_id)
        print("resolution=%dx%d" % (first.width, first.height))
        print("pixel_format=%s" % first.pixel_format)
        print("frame_shape=%s" % (first.data.shape,))
        print("frame_dtype=%s" % first.data.dtype)
        print("frame_min=%d frame_max=%d frame_mean=%.4f" %
              (first.data.min(), first.data.max(), first.data.mean()))
        print("frame_difference=%.6f" % difference)
        print("capture_count=%d elapsed_seconds=%.4f cpu_copy=true" % (len(frames), elapsed))
        print("sample_frames=%s" % output.resolve())
    finally:
        adapter.close()
        print("CLEANUP=adapter_closed")


if __name__ == "__main__":
    main()
