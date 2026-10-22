#!/usr/bin/env python3
"""Real Reach Camera -> NVENC -> RTSP runtime smoke without PPO."""

import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from isaacgym import gymapi as _gymapi  # noqa: F401

from simulation_core.domain import ReachTrainingTaskConfig, TaskStatus
from simulation_core.stream_session import RecordingStreamGateway
from simulation_core.training_task_runner import TrainingTaskRunner, TrainingTaskSpec
from simulation_core.video_runtime import TaskVideoRuntimeFactory


ROOT=Path(__file__).resolve().parents[1]


def wait_port(port,process):
    for _ in range(200):
        if process.poll() is not None: raise RuntimeError("MediaMTX exited")
        try:
            with socket.create_connection(("127.0.0.1",port),.1): return
        except OSError: time.sleep(.05)
    raise RuntimeError("MediaMTX port not ready")


def main():
    log=ROOT/"artifacts/m2_reach_stream_smoke_mediamtx.log"; log.parent.mkdir(exist_ok=True)
    media=subprocess.Popen(["/opt/mediamtx/mediamtx",str(ROOT/"config/mediamtx_phase_6_5.yml")],stdout=log.open("wb"),stderr=subprocess.STDOUT)
    try:
        wait_port(18554,media); wait_port(18889,media)
        os.environ["SIMULATION_STREAM_RTSP_BASE"]="rtsp://127.0.0.1:18554"
        os.environ["SIMULATION_STREAM_PUBLIC_BASE_URL"]="http://192.0.2.10:18889"
        os.environ["SIMULATION_FFMPEG_PATH"]="/opt/ffmpeg/bin/ffmpeg"
        config=ReachTrainingTaskConfig(task_type="training",environment_type="s1_reach",seed=19,
            rollout_steps=1,num_updates=1,checkpoint_every=1,episode_steps=128,
            physics_steps_per_action=8,update_epochs=1,batch_size=1,device="cuda:0",
            video_enabled=True,run_mode="smoke")
        task=TrainingTaskSpec("m2-reach-stream-task","m2-reach-stream-smoke","m2",config)
        env=TrainingTaskRunner._create_environment(config,task); gateway=RecordingStreamGateway()
        lifecycle=None
        try:
            env.initialize(); env.reset()
            lifecycle=TaskVideoRuntimeFactory(ROOT/"artifacts",stream_gateway=gateway).create(env,task)
            running=lifecycle.on_status(task.run_id,task.task_id,TaskStatus.RUNNING)
            for _ in range(90):
                state=env.backend.read_robot_state(); ee=tuple(float(v) for v in state.ee_link_state)
                delta=(env.target_xy[0]-ee[0],env.target_xy[1]-ee[1]); norm=math.hypot(*delta)
                action=(0.,0.) if norm<1e-9 else (delta[0]/norm,delta[1]/norm)
                _,_,done,_=env.step(action)
                if done: env.reset()
                time.sleep(.01)
            time.sleep(2)
            stopped=lifecycle.on_status(task.run_id,task.task_id,TaskStatus.COMPLETED)
            lifecycle=None
        finally:
            if lifecycle is not None: lifecycle.on_status(task.run_id,task.task_id,TaskStatus.FAILED)
            env.close()
        video=ROOT/"artifacts/m2/m2-reach-stream-smoke/video/overview.mp4"
        history=list(gateway.records)
        text=log.read_text(encoding="utf-8",errors="replace")
        output={"running_descriptor":running[0].__dict__,"stopped_descriptor":stopped[0].__dict__,
                "history":history,"video":str(video),"video_exists":video.is_file(),
                "video_bytes":video.stat().st_size if video.is_file() else 0,
                "rtsp_publisher_connected":"is publishing to path" in text or "publisher" in text.lower(),
                "mediamtx_log":str(log)}
        output["passed"]=output["video_exists"] and output["video_bytes"]>0 and output["rtsp_publisher_connected"]
        print("S1_REACH_STREAM_SMOKE="+json.dumps(output,sort_keys=True,default=str),flush=True)
        if not output["passed"]: raise SystemExit(2)
    finally:
        if media.poll() is None: media.terminate()
        try: media.wait(timeout=10)
        except subprocess.TimeoutExpired: media.kill(); media.wait()


if __name__=="__main__": main()
