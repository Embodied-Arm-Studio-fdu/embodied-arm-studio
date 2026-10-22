#!/usr/bin/env python3
"""Submit and observe the M2 20k run through HTTP, Redis/RQ and GPUWorker."""

import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import httpx

from simulation_core import (
    RedisCancellationStore, RedisRecordingBackendClient, RedisTaskRepository,
    RQTaskQueue, SimulationService, TaskStatus, create_http_app,
)


ROOT=Path(__file__).resolve().parents[1]
REDIS_URL="redis://127.0.0.1:6379/0"; NAMESPACE="m2-reach-production"
QUEUE="m2-reach-production"; TOKEN="m2-production-token"
RUN_ID="phase-m2-s1-reach-20k-seed7-v2"; PROJECT_ID="phase-m2"


def wait_tcp(port, process, timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None: raise RuntimeError("process exited before port %d"%port)
        try:
            with socket.create_connection(("127.0.0.1",port),timeout=.2): return
        except OSError: time.sleep(.05)
    raise TimeoutError("port %d not ready"%port)


async def submit(service):
    app=create_http_app(service,TOKEN)
    transport=httpx.ASGITransport(app=app,raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport,base_url="http://simulation") as client:
        payload={"run_id":RUN_ID,"project_id":PROJECT_ID,"config":{
            "task_type":"training","environment_type":"s1_reach","seed":7,
            "rollout_steps":200,"num_updates":100,"checkpoint_every":10,
            "episode_steps":128,"physics_steps_per_action":8,"update_epochs":1,
            "batch_size":16,"device":"cuda:0","video_enabled":True,
            "camera_id":"overview","video_fps":15.0,"video_width":640,"video_height":480,
            "run_mode":"training","action_delta_m":.005,
            "target_center_xy_m":[.36,0.],"target_half_extent_xy_m":[.03,.08],
            "success_distance_m":.02,"success_hold_steps":3}}
        response=await client.post("/simulation/tasks",headers={"Authorization":"Bearer "+TOKEN},json=payload)
        if response.status_code not in (200,202): raise RuntimeError("HTTP submit failed: %s %s"%(response.status_code,response.text))
        catalog=await client.get("/simulation/task-catalog",headers={"Authorization":"Bearer "+TOKEN})
        return response.json(),catalog.json()


def stop(process):
    if process is None:return
    if process.poll() is None: process.terminate()
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)


def main():
    redis=media=worker=None
    try:
        redis=subprocess.Popen(["redis-server","--port","6379","--save","","--appendonly","no"],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        wait_tcp(6379,redis)
        media=subprocess.Popen(["/opt/mediamtx/mediamtx",str(ROOT/"config/mediamtx_phase_6_5.yml")],stdout=ROOT.joinpath("artifacts/m2_mediamtx.log").open("wb"),stderr=subprocess.STDOUT)
        wait_tcp(18554,media); wait_tcp(18889,media)
        repository=RedisTaskRepository.from_url(REDIS_URL,NAMESPACE)
        repository.client.flushdb()
        cancellation=RedisCancellationStore.from_url(REDIS_URL,NAMESPACE)
        queue=RQTaskQueue.from_url(REDIS_URL,str(ROOT/"artifacts"),QUEUE,NAMESPACE,job_timeout=7200)
        backend=RedisRecordingBackendClient(repository.client,NAMESPACE)
        service=SimulationService(repository,queue,backend,cancellation_store=cancellation,artifacts_root=ROOT/"artifacts")
        submitted,catalog=asyncio.run(submit(service)); task_id=submitted["task_id"]
        print("PRODUCTION_SUBMIT="+json.dumps(submitted,sort_keys=True),flush=True)
        print("PRE_GATE_CATALOG="+json.dumps(catalog,sort_keys=True),flush=True)
        env=dict(os.environ); env["PYTHONPATH"]=str(ROOT/"src")
        env.pop("DISPLAY",None)
        env["SIMULATION_HEARTBEAT_INTERVAL"]="2"
        env["SIMULATION_STREAM_RTSP_BASE"]="rtsp://127.0.0.1:18554"
        env["SIMULATION_STREAM_PUBLIC_BASE_URL"]="http://192.0.2.10:18889"
        env["SIMULATION_FFMPEG_PATH"]="/opt/ffmpeg/bin/ffmpeg"
        env.setdefault("VK_ICD_FILENAMES","/usr/share/vulkan/icd.d/nvidia_icd.json")
        worker_log=ROOT/"artifacts/m2_reach_rq_worker.log"; worker_log.parent.mkdir(exist_ok=True)
        worker=subprocess.Popen(["rq","worker","--burst","-u",REDIS_URL,"-n","m2-reach-worker","--logging_level","INFO",QUEUE],cwd=str(ROOT),env=env,stdout=worker_log.open("wb"),stderr=subprocess.STDOUT)
        deadline=time.monotonic()+7200; last=(-1,None)
        while time.monotonic()<deadline:
            task=service.get_task(task_id)
            events=backend.events_for_run(RUN_ID)
            metrics=[item for item in events if item.get("type")=="metric"]
            marker=(len(metrics),task.status.value)
            if marker!=last:
                latest=metrics[-1].get("payload",{}) if metrics else {}
                print("PRODUCTION_PROGRESS="+json.dumps({"status":task.status.value,"metric_count":len(metrics),"latest":latest},sort_keys=True),flush=True); last=marker
            if task.is_terminal: break
            time.sleep(2)
        else: raise TimeoutError("production task timeout")
        callbacks=backend.terminal_callbacks_for_run(RUN_ID)
        run_root=ROOT/"artifacts"/PROJECT_ID/RUN_ID
        result={"task_id":task_id,"status":task.status.value,"run_root":str(run_root),
                "callbacks":callbacks,"worker_exit":worker.poll(),
                "manifest_exists":(run_root/"manifest.json").is_file(),
                "latest_exists":(run_root/"checkpoints/model_latest.pt").is_file(),
                "best_exists":(run_root/"checkpoints/model_best.pt").is_file(),
                "evaluation_summary_exists":(run_root/"evaluation/summary.json").is_file(),
                "evaluation_video_exists":(run_root/"evaluation_video/successful_episode.mp4").is_file()}
        print("S1_REACH_PRODUCTION_GATE="+json.dumps(result,sort_keys=True),flush=True)
        if task.status!=TaskStatus.COMPLETED: raise SystemExit(2)
    finally:
        stop(worker); stop(media); stop(redis)


if __name__=="__main__": main()
