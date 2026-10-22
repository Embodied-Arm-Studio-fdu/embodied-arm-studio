#!/usr/bin/env python3
"""M3.5 production Camera/NVENC/RTSP acceptance for four S1 demos."""
import asyncio, json, os, socket, subprocess, time
from pathlib import Path
import httpx
from simulation_core import RedisCancellationStore, RedisRecordingBackendClient, RedisTaskRepository, RQTaskQueue, SimulationService, create_http_app

ROOT=Path(__file__).resolve().parents[1]; REDIS="redis://127.0.0.1:6379/0"; NS="m3-5-visual"; TOKEN="m35-token"; PROJECT="phase-m3-5"
CASES=(
 ("demo_s1_target_pose","target-pose-quality-v2",12,{}),
 ("demo_s1_line_tracking","line-tracking-quality-v2",20,{}),
 ("demo_s1_circle_tracking","circle-tracking-quality-v2",24,{}),
 ("demo_s1_joint_position","joint-home-quality-v2",30,{"pose":"home"}),
)
def wait_port(port,p):
 for _ in range(300):
  if p.poll() is not None: raise RuntimeError("service exited")
  try:
   with socket.create_connection(("127.0.0.1",port),.1):return
  except OSError:time.sleep(.05)
 raise TimeoutError(port)
def stop(p):
 if p and p.poll() is None:p.terminate()
 if p:
  try:p.wait(10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
async def submit(service,task_id,run_id,duration,extra):
 config={"task_type":"demo","environment_type":task_id,"duration_seconds":duration,"seed":7,"physics_steps_per_command":1,"video_enabled":True,"camera_id":"overview","width":640,"height":480,"fps":15};config.update(extra)
 async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_http_app(service,TOKEN),raise_app_exceptions=False),base_url="http://simulation") as client:
  response=await client.post("/simulation/tasks",headers={"Authorization":"Bearer "+TOKEN},json={"run_id":run_id,"project_id":PROJECT,"config":config})
  if response.status_code!=202:raise RuntimeError(response.text)
  return response.json()["task_id"]
def main():
 redis=media=worker=None
 try:
  redis=subprocess.Popen(["redis-server","--port","6379","--save","","--appendonly","no"],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT);wait_port(6379,redis)
  media_log=ROOT/"artifacts/phase-m3-5-mediamtx.log";media_log.parent.mkdir(exist_ok=True)
  media=subprocess.Popen(["/opt/mediamtx/mediamtx",str(ROOT/"config/mediamtx_phase_6_5.yml")],stdout=media_log.open("wb"),stderr=subprocess.STDOUT);wait_port(18554,media);wait_port(18889,media)
  repo=RedisTaskRepository.from_url(REDIS,NS);repo.client.flushdb();cancel=RedisCancellationStore.from_url(REDIS,NS);queue=RQTaskQueue.from_url(REDIS,str(ROOT/"artifacts"),NS,NS,job_timeout=900);backend=RedisRecordingBackendClient(repo.client,NS);service=SimulationService(repo,queue,backend,cancellation_store=cancel,artifacts_root=ROOT/"artifacts")
  env=dict(os.environ);env["PYTHONPATH"]=str(ROOT/"src");env.pop("DISPLAY",None);env.update({"SIMULATION_HEARTBEAT_INTERVAL":"1","SIMULATION_STREAM_RTSP_BASE":"rtsp://127.0.0.1:18554","SIMULATION_STREAM_PUBLIC_BASE_URL":"http://192.0.2.10:18889","SIMULATION_FFMPEG_PATH":"/opt/ffmpeg/bin/ffmpeg","VK_ICD_FILENAMES":"/usr/share/vulkan/icd.d/nvidia_icd.json"})
  worker_log=ROOT/"artifacts/phase-m3-5-worker.log";worker=subprocess.Popen(["rq","worker","-u",REDIS,"-n","m35-worker","--logging_level","INFO",NS],cwd=str(ROOT),env=env,stdout=worker_log.open("wb"),stderr=subprocess.STDOUT)
  results={}
  for definition,run_id,duration,extra in CASES:
   task_id=asyncio.run(submit(service,definition,run_id,duration,extra));deadline=time.monotonic()+300
   while time.monotonic()<deadline:
    task=service.get_task(task_id)
    if task.is_terminal:break
    time.sleep(.5)
   root=ROOT/"artifacts"/PROJECT/run_id;video=root/"video/overview.mp4";callbacks=backend.terminal_callbacks_for_run(run_id)
   results[definition]={"run_id":run_id,"status":task.status.value,"video":str(video),"video_bytes":video.stat().st_size if video.is_file() else 0,"manifest":str(root/"manifest.json"),"metrics":callbacks[-1].get("metrics",{}) if callbacks else {}}
  media_text=media_log.read_text(errors="replace");output={"runs":results,"rtsp_publishers":media_text.count("is publishing to path"),"media_log":str(media_log),"worker_log":str(worker_log)}
  output["passed"]=all(v["status"]=="completed" and v["video_bytes"]>0 and Path(v["manifest"]).is_file() for v in results.values()) and output["rtsp_publishers"]>=4
  print("S1_DEMO_VISUAL_QUALITY_GATE="+json.dumps(output,sort_keys=True));
  if not output["passed"]:raise SystemExit(2)
 finally:stop(worker);stop(media);stop(redis)
if __name__=="__main__":main()
