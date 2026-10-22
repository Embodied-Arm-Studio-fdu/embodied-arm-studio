#!/usr/bin/env python3
"""M3 HTTP/RQ/GPUWorker/Camera/NVENC/RTSP production and stop gate."""
import asyncio, json, os, socket, subprocess, time
from pathlib import Path
import httpx
from simulation_core import RedisCancellationStore, RedisRecordingBackendClient, RedisTaskRepository, RQTaskQueue, SimulationService, TaskStatus, create_http_app

ROOT=Path(__file__).resolve().parents[1]; REDIS="redis://127.0.0.1:6379/0"; NS="m3-demo-production"; QUEUE=NS; TOKEN="m3-token"; PROJECT="phase-m3"
CASES=(
 ("demo_s1_target_pose","m3-target-final",{}),
 ("demo_s1_line_tracking","m3-line-final",{}),
 ("demo_s1_circle_tracking","m3-circle-final",{}),
 ("demo_s1_joint_position","m3-joint-home-final",{"pose":"home"}),
 ("demo_s1_joint_position","m3-joint-side-final",{"pose":"side"}),
)
def wait_port(port,process):
 for _ in range(300):
  if process.poll() is not None: raise RuntimeError("process exited")
  try:
   with socket.create_connection(("127.0.0.1",port),.1): return
  except OSError: time.sleep(.05)
 raise TimeoutError(port)
def stop_process(p):
 if p and p.poll() is None: p.terminate()
 if p:
  try:p.wait(10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
async def request(service,method,path,payload=None):
 async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_http_app(service,TOKEN),raise_app_exceptions=False),base_url="http://simulation") as c:
  return await c.request(method,path,headers={"Authorization":"Bearer "+TOKEN},json=payload)
def main():
 redis=media=worker=None
 try:
  redis=subprocess.Popen(["redis-server","--port","6379","--save","","--appendonly","no"],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT);wait_port(6379,redis)
  media_log=ROOT/"artifacts/m3_mediamtx.log";media_log.parent.mkdir(exist_ok=True)
  media=subprocess.Popen(["/opt/mediamtx/mediamtx",str(ROOT/"config/mediamtx_phase_6_5.yml")],stdout=media_log.open("wb"),stderr=subprocess.STDOUT);wait_port(18554,media);wait_port(18889,media)
  repo=RedisTaskRepository.from_url(REDIS,NS);repo.client.flushdb();cancel=RedisCancellationStore.from_url(REDIS,NS);queue=RQTaskQueue.from_url(REDIS,str(ROOT/"artifacts"),QUEUE,NS,job_timeout=900);backend=RedisRecordingBackendClient(repo.client,NS);service=SimulationService(repo,queue,backend,cancellation_store=cancel,artifacts_root=ROOT/"artifacts")
  env=dict(os.environ);env["PYTHONPATH"]=str(ROOT/"src");env.pop("DISPLAY",None);env.update({"SIMULATION_HEARTBEAT_INTERVAL":"1","SIMULATION_STREAM_RTSP_BASE":"rtsp://127.0.0.1:18554","SIMULATION_STREAM_PUBLIC_BASE_URL":"http://192.0.2.10:18889","SIMULATION_FFMPEG_PATH":"/opt/ffmpeg/bin/ffmpeg","VK_ICD_FILENAMES":"/usr/share/vulkan/icd.d/nvidia_icd.json"})
  worker_log=ROOT/"artifacts/m3_rq_worker.log";worker=subprocess.Popen(["rq","worker","-u",REDIS,"-n","m3-worker","--logging_level","INFO",QUEUE],cwd=str(ROOT),env=env,stdout=worker_log.open("wb"),stderr=subprocess.STDOUT)
  results={}
  for task_type,run_id,extra in CASES:
   config={"task_type":"demo","environment_type":task_type,"duration_seconds":10,"seed":7,"physics_steps_per_command":1,"video_enabled":True,"camera_id":"overview","width":640,"height":480,"fps":15};config.update(extra)
   response=asyncio.run(request(service,"POST","/simulation/tasks",{"run_id":run_id,"project_id":PROJECT,"config":config}));
   if response.status_code!=202: raise RuntimeError(response.text)
   task_id=response.json()["task_id"];deadline=time.monotonic()+300
   while time.monotonic()<deadline:
    task=service.get_task(task_id)
    if task.is_terminal:break
    time.sleep(.5)
   root=ROOT/"artifacts"/PROJECT/run_id;manifest=root/"manifest.json";video=root/"video/overview.mp4"
   callbacks=backend.terminal_callbacks_for_run(run_id)
   results[run_id]={"task_id":task_id,"status":task.status.value,"manifest":str(manifest),"manifest_exists":manifest.is_file(),"video":str(video),"video_bytes":video.stat().st_size if video.is_file() else 0,"checkpoint_dir_exists":(root/"checkpoints").exists(),"final_metrics":callbacks[-1].get("metrics",{}) if callbacks else {}}
  stop_run="m3-stop-final";config={"task_type":"demo","environment_type":"demo_s1_target_pose","duration_seconds":30,"video_enabled":False}
  response=asyncio.run(request(service,"POST","/simulation/tasks",{"run_id":stop_run,"project_id":PROJECT,"config":config}));stop_id=response.json()["task_id"]
  while service.get_task(stop_id).status not in (TaskStatus.RUNNING,TaskStatus.FAILED):time.sleep(.2)
  stop_response=asyncio.run(request(service,"POST","/simulation/tasks/%s/stop"%stop_id));deadline=time.monotonic()+60
  while not service.get_task(stop_id).is_terminal and time.monotonic()<deadline:time.sleep(.2)
  results[stop_run]={"task_id":stop_id,"stop_http":stop_response.status_code,"status":service.get_task(stop_id).status.value}
  media_text=media_log.read_text(errors="replace");catalog=asyncio.run(request(service,"GET","/simulation/task-catalog")).json()
  output={"runs":results,"rtsp_publisher_seen":"is publishing to path" in media_text,"catalog":catalog,"worker_alive":worker.poll() is None,"media_log":str(media_log),"worker_log":str(worker_log)}
  output["passed"]=all(v.get("status")=="completed" and v.get("manifest_exists") and v.get("video_bytes",0)>0 and not v.get("checkpoint_dir_exists") for k,v in results.items() if k!=stop_run) and results[stop_run]["status"]=="stopped" and output["rtsp_publisher_seen"]
  print("S1_DEMO_PRODUCTION_GATE="+json.dumps(output,sort_keys=True));
  if not output["passed"]:raise SystemExit(2)
 finally:stop_process(worker);stop_process(media);stop_process(redis)
if __name__=="__main__":main()
