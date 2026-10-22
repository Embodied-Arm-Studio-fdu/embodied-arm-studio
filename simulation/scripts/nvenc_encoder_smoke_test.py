#!/usr/bin/env python3
"""Real-time S1 Push Camera -> FramePipeline -> NVENC -> Annex-B smoke."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

from simulation_core.isaac_gym_camera import IsaacGymCameraConfig
from simulation_core.push_adapter import ACTION_DIM, IsaacGymPushBackend, PushAdapter, PushConfig
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig
from simulation_core.video_encoder import (
    AnnexBFileSink, EncoderConfig, EncoderWorker, FFmpegNvencEncoder,
    VideoEncoderError,
)
from simulation_core.video_frame_pipeline import DropPolicy, FramePipeline, PixelFormat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/phase-6-3-1"))
    parser.add_argument("--ffmpeg", type=Path,
                        default=Path("/opt/ffmpeg/bin/ffmpeg"))
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.artifacts_root / run_id / "video" / "overview.h264"
    config = EncoderConfig()
    adapter = None; worker = None
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    try:
        encoder = FFmpegNvencEncoder(args.ffmpeg)
        try:
            pipeline = FramePipeline("phase-6-3-1-smoke", "phase-6-3-1-task",
                PixelFormat.RGB24, max_frames=3, drop_policy=DropPolicy.DROP_OLDEST,
                target_fps=config.fps)
            push = PushConfig(max_steps=100000)
            backend = IsaacGymPushBackend(push,
                s1_config=S1IsaacGymConfig(graphics_device_id=0, headless=True),
                camera_config=IsaacGymCameraConfig(), frame_pipeline=pipeline)
            adapter = PushAdapter(push, backend=backend)
            sink = AnnexBFileSink(output)
            worker = EncoderWorker(pipeline, encoder, config, sink)
            try:
                worker.start()
            except VideoEncoderError as exc:
                print("NVENC_ENCODER_SMOKE_RESULT=failure")
                print("encoder_backend=ffmpeg_cli_h264_nvenc")
                print("hardware_encoder=h264_nvenc")
                print("failure_reason=%s" % str(exc).replace("\n", " | "))
                print("OVSTREAM_SMOKE_RESULT=not_available")
                raise
            adapter.initialize(); adapter.reset()
            started = time.monotonic()
            action = (0.0002, 0.0001)
            while time.monotonic() - started < args.duration:
                adapter.step(action); backend.maybe_capture_frame()
            worker.close()
            if worker.error: raise worker.error
            stats = worker.stats
        finally:
            if worker is not None: worker.close()
        decode = subprocess.run([str(args.ffmpeg), "-v", "error", "-f", "h264", "-i", str(output),
                                 "-f", "null", "-"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if decode.returncode != 0: raise RuntimeError("decode failed: %s" % decode.stderr)
        ffprobe = args.ffmpeg.with_name("ffprobe")
        probe = subprocess.run([
            str(ffprobe), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height",
            "-of", "default=noprint_wrappers=1", str(output),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        probe_fields = dict(line.split("=", 1) for line in probe.stdout.splitlines()
                            if "=" in line)
        if probe_fields != {"codec_name": "h264", "width": "640", "height": "480"}:
            raise RuntimeError("unexpected ffprobe stream: %r" % probe_fields)
        version = subprocess.run([str(args.ffmpeg), "-version"], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, check=True).stdout.splitlines()[0]
        print("NVENC_ENCODER_SMOKE_RESULT=success")
        print("encoder_backend=ffmpeg_cli_h264_nvenc")
        print("ffmpeg_path=%s" % args.ffmpeg)
        print("ffmpeg_version=%s" % version)
        print("hardware_encoder=h264_nvenc")
        print("resolution=640x480\ntarget_fps=15")
        print("input_frames=%d\nencoded_frames=%d\nencoded_packets_or_chunks=%d" %
              (stats.input_frames, stats.encoded_frames, stats.encoded_packets))
        print("keyframes=%d\nencoded_bytes=%d" % (stats.keyframes, stats.encoded_bytes))
        print("pipeline_dropped_frames=%d" % pipeline.stats.dropped_frames)
        print("average_encode_ms=%.4f" % stats.average_encode_ms)
        print("effective_fps=%.4f" % (stats.encoded_frames / args.duration))
        print("approximate_bitrate=%.0f" % (stats.encoded_bytes * 8 / args.duration))
        print("output_path=%s" % output.resolve())
        print("codec_name=%s\nwidth=%s\nheight=%s\ndecode_success=true" %
              (probe_fields["codec_name"], probe_fields["width"], probe_fields["height"]))
        print("decode_validation=success")
        print("OVSTREAM_SMOKE_RESULT=not_available")
    finally:
        if adapter is not None: adapter.close()
        print("CLEANUP=complete")


if __name__ == "__main__": main()
