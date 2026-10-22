#!/usr/bin/env python3
"""M4 all-arm production video and cooperative-stop gate."""
import asyncio,json,os,socket,subprocess,time
from pathlib import Path
import httpx
from simulation_core import RedisCancellationStore,RedisRecordingBackendClient,RedisTaskRepository,RQTaskQueue,SimulationService,TaskStatus,create_http_app
ROOT=Path(__file__).resolve().parents[1];REDIS="redis://127.0.0.1:6379/0";NS="m4-joint-sweep";TOKEN="m4-token";PROJECT="phase-m4"
def wait_port(port,p):
 for _ in range(300):
  if p.poll() is not None:raise RuntimeError("service exited")
  try:
   with socket.create_connection(("127.0.0.1",port),.1):return
  except OSError:time.sleep(.05)
 raise TimeoutError(port)
def stop_process(p):
 if p and p.poll() is None:p.terminate()
 if p:
  try:p.wait(10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
async def call(service,method,path,payload=None):
 async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_http_app(service,TOKEN),raise_app_exceptions=False),base_url="http://simulation") as client:return await client.request(method,path,headers={"Authorization":"Bearer "+TOKEN},json=payload)
def config(duration,video):return {"task_type":"demo","environment_type":"demo_s1_joint_sweep","duration_seconds":duration,"joint_index":"all_arm","speed_scale":.5,"physics_steps_per_command":1,"video_enabled":video,"camera_id":"overview","width":640,"height":480,"fps":15}
def submit(service,run_id,duration,video):
 response=asyncio.run(call(service,"POST","/simulation/tasks",{"run_id":run_id,"project_id":PROJECT,"config":config(duration,video)}))
 if response.status_code!=202:raise RuntimeError(response.text)
 return response.json()["task_id"]
def main():
 redis=media=worker=None
 try:
  redis=subprocess.Popen(["redis-server","--port","6379","--save","","--appendonly","no"],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT);wait_port(6379,redis)
  media_log=ROOT/"artifacts/phase-m4-mediamtx.log";media_log.parent.mkdir(exist_ok=True);media=subprocess.Popen(["/opt/mediamtx/mediamtx",str(ROOT/"config/mediamtx_phase_6_5.yml")],stdout=media_log.open("wb"),stderr=subprocess.STDOUT);wait_port(18554,media);wait_port(18889,media)
  repo=RedisTaskRepository.from_url(REDIS,NS);repo.client.flushdb();cancel=RedisCancellationStore.from_url(REDIS,NS);queue=RQTaskQueue.from_url(REDIS,str(ROOT/"artifacts"),NS,NS,job_timeout=900);backend=RedisRecordingBackendClient(repo.client,NS);service=SimulationService(repo,queue,backend,cancellation_store=cancel,artifacts_root=ROOT/"artifacts")
  env=dict(os.environ);env["PYTHONPATH"]=str(ROOT/"src");env.pop("DISPLAY",None);env.update({"SIMULATION_HEARTBEAT_INTERVAL":"1","SIMULATION_STREAM_RTSP_BASE":"rtsp://127.0.0.1:18554","SIMULATION_STREAM_PUBLIC_BASE_URL":"http://192.0.2.10:18889","SIMULATION_FFMPEG_PATH":"/opt/ffmpeg/bin/ffmpeg","VK_ICD_FILENAMES":"/usr/share/vulkan/icd.d/nvidia_icd.json"})
  worker_log=ROOT/"artifacts/phase-m4-worker.log";worker=subprocess.Popen(["rq","worker","-u",REDIS,"-n","m4-worker","--logging_level","INFO",NS],cwd=str(ROOT),env=env,stdout=worker_log.open("wb"),stderr=subprocess.STDOUT)
  run_id="joint-sweep-all-arm";task_id=submit(service,run_id,70,True);deadline=time.monotonic()+180
  while time.monotonic()<deadline:
   task=service.get_task(task_id)
   if task.is_terminal:break
   time.sleep(.5)
  root=ROOT/"artifacts"/PROJECT/run_id;video=root/"video/overview.mp4";metrics_path=root/"metrics/demo.jsonl"
  metrics=json.loads(metrics_path.read_text().splitlines()[-1]) if metrics_path.is_file() else {}
  stop_run="joint-sweep-stop";stop_id=submit(service,stop_run,120,False)
  while service.get_task(stop_id).status not in (TaskStatus.RUNNING,TaskStatus.FAILED):time.sleep(.2)
  stop_response=asyncio.run(call(service,"POST","/simulation/tasks/%s/stop"%stop_id));deadline=time.monotonic()+60
  while not service.get_task(stop_id).is_terminal and time.monotonic()<deadline:time.sleep(.2)
  text=media_log.read_text(errors="replace");output={"status":task.status.value,"task_id":task_id,"run_id":run_id,"metrics":metrics,"video":str(video),"video_bytes":video.stat().st_size if video.is_file() else 0,"manifest":str(root/"manifest.json"),"checkpoint_dir_exists":(root/"checkpoints").exists(),"rtsp_publisher_seen":"is publishing to path" in text,"stop":{"http_status":stop_response.status_code,"status":service.get_task(stop_id).status.value,"task_id":stop_id},"worker_log":str(worker_log),"media_log":str(media_log)}
  output["passed"]=task.status==TaskStatus.COMPLETED and metrics.get("sweep_completed") is True and metrics.get("non_finite_count")==0 and metrics.get("limit_clamp_count")==0 and metrics.get("final_home_error_rad",1)<.01 and output["video_bytes"]>0 and Path(output["manifest"]).is_file() and not output["checkpoint_dir_exists"] and output["rtsp_publisher_seen"] and output["stop"]["status"]=="stopped"
  print("S1_JOINT_SWEEP_PRODUCTION_GATE="+json.dumps(output,sort_keys=True));
  if not output["passed"]:raise SystemExit(2)
 finally:stop_process(worker);stop_process(media);stop_process(redis)
if __name__=="__main__":main()
