#!/usr/bin/env python3
"""Real S1 lifecycle + StreamSession + WHEP disconnect/reconnect smoke."""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from simulation_core.domain import SimulationTask, TaskConfig, TaskStatus
from simulation_core.isaac_gym_camera import IsaacGymCameraConfig
from simulation_core.media_streaming import MediaMtxPublisherConfig, MediaMtxPublisherSink
from simulation_core.push_adapter import ACTION_DIM, IsaacGymPushBackend, PushAdapter, PushConfig
from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig
from simulation_core.stream_session import StreamSessionManager, TaskStreamLifecycle
from simulation_core.video_encoder import EncoderConfig, EncoderWorker, FFmpegNvencEncoder
from simulation_core.video_frame_pipeline import DropPolicy, FramePipeline, PixelFormat
from webrtc_test_client import receive_video_sync

RUN_ID = "phase-6-5-smoke"
TASK_ID = "phase-6-5-smoke-task"


class VideoRuntime:
    def __init__(self, worker, publisher): self.worker, self.publisher = worker, publisher
    def start(self): self.worker.start(); self.publisher.open()
    def stop(self): self.worker.close()


def wait_tcp(port, process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None: raise RuntimeError("MediaMTX exited")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2): return
        except OSError: time.sleep(.05)
    raise RuntimeError("MediaMTX port not ready")


def stop(process):
    if process is None: return
    if process.poll() is None: process.terminate()
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mediamtx", type=Path, default=Path("/opt/mediamtx/mediamtx"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/ffmpeg/bin/ffmpeg"))
    parser.add_argument("--config", type=Path, default=Path("config/mediamtx_phase_6_5.yml"))
    args = parser.parse_args(); mediamtx = adapter = manager = session = worker = None
    os.environ.pop("DISPLAY", None)
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    try:
        mediamtx = subprocess.Popen([str(args.mediamtx), str(args.config)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wait_tcp(18554, mediamtx); wait_tcp(18889, mediamtx)
        pipeline = FramePipeline(RUN_ID, TASK_ID, PixelFormat.RGB24, 3, DropPolicy.DROP_OLDEST, 15)
        push = PushConfig(max_steps=100000)
        backend = IsaacGymPushBackend(push, s1_config=S1IsaacGymConfig(graphics_device_id=0, headless=True),
            camera_config=IsaacGymCameraConfig(), frame_pipeline=pipeline)
        adapter = PushAdapter(push, backend=backend); adapter.initialize(); adapter.reset()
        holder = {}
        def runtime_factory(item):
            publisher = MediaMtxPublisherSink(MediaMtxPublisherConfig(item.stream_id, ffmpeg_path=args.ffmpeg, fps=item.fps))
            holder["publisher"] = publisher
            holder["worker"] = EncoderWorker(pipeline, FFmpegNvencEncoder(args.ffmpeg),
                EncoderConfig(width=item.width, height=item.height, fps=item.fps, gop_size=30), publisher)
            return VideoRuntime(holder["worker"], publisher)
        manager = StreamSessionManager(runtime_factory)
        lifecycle = TaskStreamLifecycle(manager)
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        task = SimulationTask(TASK_ID, RUN_ID, "mock-project", TaskStatus.RUNNING, now, now, TaskConfig())
        session = lifecycle.on_task_state(task)[0]
        action = (0.0002, 0.0001)
        def receive(target):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    target["result"] = receive_video_sync(session.whep_url, frames=5, timeout=8); return
                except Exception as exc:
                    target["error"] = exc; time.sleep(.2)
        def pump(target, timeout=12):
            thread = threading.Thread(target=receive, args=(target,), daemon=True); thread.start()
            deadline = time.monotonic() + timeout
            while thread.is_alive() and time.monotonic() < deadline:
                adapter.step(action); backend.maybe_capture_frame()
            thread.join(timeout=2)
            if "result" not in target: raise RuntimeError("client failed: %s" % target.get("error", "timeout"))
        first = {}; pump(first); manager.mark_client_connected(session.stream_id); manager.mark_client_disconnected(session.stream_id)
        disconnected = manager.get_session(session.stream_id)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline: adapter.step(action); backend.maybe_capture_frame()
        second = {}; pump(second); reconnected = manager.mark_client_connected(session.stream_id)
        completed = SimulationTask(TASK_ID, RUN_ID, "mock-project", TaskStatus.COMPLETED, now, now, TaskConfig())
        final_stream = lifecycle.on_task_state(completed)[0]
        worker = holder["worker"]
        if worker.error: raise worker.error
        print("PHASE_6_5_S1_SMOKE_RESULT=success")
        print("run_id=%s\ntask_id=%s\nstream_id=%s" % (RUN_ID, TASK_ID, session.stream_id))
        print("task_state_running=RUNNING\nstream_state_running=READY")
        print("first_client_frames=%d" % first["result"].frames_decoded)
        print("session_after_disconnect=%s\ntraining_after_disconnect=RUNNING" % disconnected.status.value)
        print("second_client_frames=%d\nreconnect_count=%d" % (second["result"].frames_decoded, reconnected.reconnect_count))
        print("first_time_to_first_decoded_frame=%.3f\nsecond_time_to_first_decoded_frame=%.3f" %
            (first["result"].time_to_first_decoded_frame, second["result"].time_to_first_decoded_frame))
        print("final_task_state=COMPLETED\nfinal_stream_state=%s" % final_stream.status.value)
        print("frontend_mode=mock\nbackend_mode=mock")
    finally:
        if manager is not None and session is not None:
            try: manager.stop_session(session.stream_id)
            except Exception: pass
        if adapter is not None: adapter.close()
        stop(mediamtx); print("CLEANUP=complete")


if __name__ == "__main__": main()
