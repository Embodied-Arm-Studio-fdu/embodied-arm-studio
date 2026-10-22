#!/usr/bin/env python3
"""Real S1 Camera -> NVENC -> RTSP -> MediaMTX -> WebRTC smoke."""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from simulation_core.isaac_gym_camera import IsaacGymCameraConfig
from simulation_core.media_streaming import MediaMtxPublisherConfig, MediaMtxPublisherSink
from simulation_core.push_adapter import ACTION_DIM, IsaacGymPushBackend, PushAdapter, PushConfig
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig
from simulation_core.video_encoder import EncoderConfig, EncoderWorker, FFmpegNvencEncoder
from simulation_core.video_frame_pipeline import DropPolicy, FramePipeline, PixelFormat
from webrtc_test_client import receive_video_sync


RUN_ID = "phase-6-4-smoke"
TASK_ID = "phase-6-4-smoke-task"
STREAM_ID = "phase-6-4-smoke-overview"


def wait_tcp(port, process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None: raise RuntimeError("MediaMTX exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2): return
        except OSError: time.sleep(.05)
    raise RuntimeError("MediaMTX port %d was not ready" % port)


def stop(process):
    if process is None or process.poll() is not None: return
    process.terminate()
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--mediamtx", type=Path, default=Path("/opt/mediamtx/mediamtx"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/ffmpeg/bin/ffmpeg"))
    parser.add_argument("--config", type=Path, default=Path("config/mediamtx_phase_6_4.yml"))
    args = parser.parse_args()
    mediamtx = adapter = worker = None
    client_result = {}; client_error = []
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    try:
        mediamtx = subprocess.Popen([str(args.mediamtx), str(args.config)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wait_tcp(18554, mediamtx); wait_tcp(18889, mediamtx)
        pipeline = FramePipeline(RUN_ID, TASK_ID, PixelFormat.RGB24,
            max_frames=3, drop_policy=DropPolicy.DROP_OLDEST, target_fps=15)
        push = PushConfig(max_steps=100000)
        backend = IsaacGymPushBackend(push,
            s1_config=S1IsaacGymConfig(graphics_device_id=0, headless=True),
            camera_config=IsaacGymCameraConfig(), frame_pipeline=pipeline)
        adapter = PushAdapter(push, backend=backend)
        publisher = MediaMtxPublisherSink(MediaMtxPublisherConfig(STREAM_ID,
            ffmpeg_path=args.ffmpeg, rtsp_port=18554, fps=15))
        worker = EncoderWorker(pipeline, FFmpegNvencEncoder(args.ffmpeg),
                               EncoderConfig(), publisher)
        worker.start(); publisher.open()
        adapter.initialize(); adapter.reset()

        def client():
            deadline = time.monotonic() + max(args.duration - 1, 3)
            while time.monotonic() < deadline:
                try:
                    client_result["value"] = receive_video_sync(
                        "http://127.0.0.1:18889/%s/whep" % STREAM_ID,
                        frames=10, timeout=5)
                    return
                except Exception as exc:
                    client_error[:] = [exc]; time.sleep(.25)
        client_thread = threading.Thread(target=client, name="phase-6-4-test-client", daemon=True)
        client_thread.start()
        started = time.monotonic()
        action = (0.0002, 0.0001)
        while time.monotonic() - started < args.duration:
            adapter.step(action); backend.maybe_capture_frame()
        client_thread.join(timeout=5)
        worker.close()
        if worker.error: raise worker.error
        if "value" not in client_result:
            raise RuntimeError("WebRTC client did not decode video: %s" %
                               (client_error[-1] if client_error else "timeout"))
        result = client_result["value"]; stats = worker.stats
        print("PHASE_6_4_SMOKE_RESULT=success")
        print("run_id=%s\ntask_id=%s\nstream_id=%s" % (RUN_ID, TASK_ID, STREAM_ID))
        print("camera_id=overview\nresolution=640x480\ntarget_fps=15")
        print("encoder=h264_nvenc\nhardware_encoder=true")
        print("rtsp_publish=true\nmediamtx_running=true")
        print("whep_status=%d\nwebrtc_connected=true\nvideo_track_received=true" % result.whep_status)
        print("browser_video_width=%d\nbrowser_video_height=%d" % (result.width, result.height))
        print("frames_encoded=%d\nframes_published_or_bytes=%d\nframes_received_or_decoded=%d" %
              (stats.encoded_frames, publisher.bytes_written, result.frames_decoded))
        print("pipeline_dropped_frames=%d" % pipeline.stats.dropped_frames)
        print("effective_encoded_fps=%.3f\napproximate_bitrate=%.0f" %
              (stats.effective_fps, stats.approximate_bitrate))
        print("frontend_mode=mock\nbackend_mode=mock")
    finally:
        if worker is not None: worker.close()
        if adapter is not None: adapter.close()
        stop(mediamtx)
        print("CLEANUP=complete")


if __name__ == "__main__": main()
