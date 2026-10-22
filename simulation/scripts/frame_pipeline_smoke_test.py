#!/usr/bin/env python3
"""Real S1 Push CameraFrame -> bounded RGB24 VideoFrame smoke test."""

import argparse
import os
import time

from simulation_core.isaac_gym_camera import IsaacGymCameraConfig
from simulation_core.push_adapter import ACTION_DIM, IsaacGymPushBackend, PushAdapter, PushConfig
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig
from simulation_core.video_frame_pipeline import (
    DropPolicy, FramePipeline, PixelFormat, RecordingFrameConsumer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--graphics-device-id", type=int, default=0)
    args = parser.parse_args()
    if args.duration <= 0 or args.target_fps <= 0 or args.max_frames <= 0:
        parser.error("duration, target-fps and max-frames must be positive")
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")

    pipeline = FramePipeline(
        "phase-6-2-smoke", "phase-6-2-task", PixelFormat.RGB24,
        max_frames=args.max_frames, drop_policy=DropPolicy.DROP_OLDEST,
        target_fps=args.target_fps,
    )
    config = PushConfig(max_steps=100000)
    backend = IsaacGymPushBackend(
        config,
        s1_config=S1IsaacGymConfig(
            graphics_device_id=args.graphics_device_id, headless=True
        ),
        camera_config=IsaacGymCameraConfig(), frame_pipeline=pipeline,
    )
    adapter = PushAdapter(config, backend=backend)
    consumer = RecordingFrameConsumer()
    action = (0.0002, 0.0001)
    try:
        adapter.initialize(); adapter.reset()
        started = time.monotonic()
        while time.monotonic() - started < args.duration:
            adapter.step(action)
            produced = backend.maybe_capture_frame()
            if produced is not None:
                pipeline.consume_next(consumer)
        while pipeline.consume_next(consumer) is not None:
            pass
        stats = pipeline.stats
        if not consumer.records or backend.last_video_error is not None:
            raise RuntimeError("frame pipeline produced no valid consumable frames")
        pts = [item["presentation_timestamp"] for item in consumer.records]
        if any(right < left for left, right in zip(pts, pts[1:])):
            raise RuntimeError("presentation timestamps are not monotonic")
        last = consumer.records[-1]
        print("FRAME_PIPELINE_SMOKE_RESULT=success")
        print("target_fps=%s" % args.target_fps)
        print("produced_frames=%d" % stats.produced_frames)
        print("accepted_frames=%d" % stats.accepted_frames)
        print("consumed_frames=%d" % stats.consumed_frames)
        print("dropped_frames=%d" % stats.dropped_frames)
        print("queue_depth=%d" % stats.queue_depth)
        print("last_frame_index=%s" % stats.last_frame_index)
        print("pixel_format=%s" % last["pixel_format"])
        print("resolution=640x480")
        print("first_pts=%.6f" % pts[0])
        print("last_pts=%.6f" % pts[-1])
        print("average_rgba_to_rgb_ms=%.4f" %
              ((stats.average_conversion_seconds or 0.0) * 1000))
        print("average_capture_interval=%s" % stats.average_capture_interval)
        print("cpu_path=true gpu_to_cpu_copy=true")
    finally:
        adapter.close()
        print("CLEANUP=adapter_closed")


if __name__ == "__main__": main()
