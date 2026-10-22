#!/usr/bin/env python3
"""Real M5 HTTP/RQ/GPU unified video and browser-WHEP acceptance gate."""
import asyncio, json, os, socket, subprocess, sys, time
from pathlib import Path

import httpx
from simulation_core import (
    RedisCancellationStore, RedisRecordingBackendClient,
    RedisRecordingStreamGateway, RedisTaskRepository, RQTaskQueue,
    SimulationService, TaskStatus, create_http_app,
)

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT.parent / "apps" / "web"
REDIS = "redis://127.0.0.1:6379/0"
NS = "m5-video-platform"
TOKEN = "m5-token"
PROJECT = "phase-m5"
MEDIA_API = "http://127.0.0.1:19997"


def wait_port(port, process, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("service exited while waiting for port %d" % port)
        try:
            with socket.create_connection(("127.0.0.1", port), .2):
                return
        except OSError:
            time.sleep(.1)
    raise TimeoutError(port)


def stop_process(process):
    if process and process.poll() is None:
        process.terminate()
    if process:
        try:
            process.wait(15)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()


async def request(app, method, path, payload=None):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://simulation", timeout=15,
    ) as client:
        return await client.request(
            method, path, headers={"Authorization": "Bearer " + TOKEN}, json=payload,
        )


def demo_config(task_id, duration=12, width=640, height=480, fps=15):
    values = {"task_type": "demo", "environment_type": task_id,
              "duration_seconds": duration, "seed": 7,
              "physics_steps_per_command": 1, "video_enabled": True,
              "camera_id": "overview", "width": width, "height": height,
              "fps": fps}
    if task_id == "demo_s1_joint_position": values["pose"] = "home"
    if task_id == "demo_s1_joint_sweep":
        values.update({"joint_index": 0, "speed_scale": .5})
    return values


def training_config(task_id, browser=False):
    values = {"task_type": "training", "environment_type": task_id,
              "seed": 7, "rollout_steps": 200,
              "num_updates": 10 if browser else 2, "checkpoint_every": 1,
              "episode_steps": 128,
              "physics_steps_per_action": 8,
              "update_epochs": 1, "batch_size": 16, "device": "cuda:0",
              "video_enabled": True, "camera_id": "overview",
              "video_fps": 30 if browser else 15,
              "video_width": 1280 if browser else 640,
              "video_height": 720 if browser else 480, "run_mode": "smoke"}
    return values


def browser_gate(descriptor, task_id, run_id, evidence_path):
    env = dict(os.environ)
    env.update({
        "WHEP_SMOKE_URL": descriptor["whep_url"],
        "WHEP_SMOKE_STREAM_ID": descriptor["stream_id"],
        "WHEP_EVIDENCE_RUN_ID": run_id,
        "WHEP_EVIDENCE_TASK_ID": task_id,
        "WHEP_EVIDENCE_CAMERA_ID": descriptor["camera_id"],
        "WHEP_EVIDENCE_PATH": str(evidence_path),
        "WHEP_MEDIAMTX_API_ORIGIN": MEDIA_API,
        "WHEP_FRONTEND_ORIGIN": "http://192.0.2.10:5173",
        "WHEP_FRONTEND_FORWARD_ORIGIN": "http://127.0.0.1:5173",
    })
    result = subprocess.run(
        ["npx", "playwright", "test", "--config=playwright.whep.config.ts",
         "--project=chromium", "--reporter=line"], cwd=str(WEB), env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    return {"passed": result.returncode == 0, "exit_code": result.returncode,
            "evidence": str(evidence_path), "output_tail": result.stdout[-3000:]}


def media_path_ready(stream_id):
    try:
        response = httpx.get(MEDIA_API + "/v3/paths/get/" + stream_id,
                             timeout=3, trust_env=False)
        payload = response.json() if response.status_code == 200 else {}
        return response.status_code == 200 and payload.get("ready") is True, response.status_code
    except Exception:
        return False, 0


def whep_gate(descriptor):
    site = "/home/user/opt/phase64-aiortc/site"
    if site not in sys.path: sys.path.insert(0, site)
    from webrtc_test_client import receive_video_sync
    error = None
    for _ in range(60):
        try:
            result = receive_video_sync(
                descriptor["whep_url"], expected_width=descriptor["width"],
                expected_height=descriptor["height"], frames=10, timeout=15,
            )
            break
        except Exception as exc:
            error = exc
            time.sleep(.25)
    else:
        raise RuntimeError("WHEP negotiation did not become ready: %s" % error)
    return {"passed": result.connected and result.video_track_received
            and result.frames_decoded >= 10 and result.whep_status == 201,
            "whep_status": result.whep_status,
            "resource_location": result.resource_location,
            "connected": result.connected,
            "video_track_received": result.video_track_received,
            "frames_decoded": result.frames_decoded,
            "resolution": [result.width, result.height],
            "first_pts": result.first_pts, "last_pts": result.last_pts,
            "time_to_first_decoded_frame": result.time_to_first_decoded_frame}


def probe_mp4(path):
    result = subprocess.run(
        ["/home/user/opt/ffmpeg-nvenc/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames",
         "-of", "json", str(path)], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(result.stdout) if result.returncode == 0 else {"error": result.stderr}


def run():
    processes = []; output = {"schema_version": "m5-video-gate/v1", "runs": []}
    before = subprocess.run(["pgrep", "-af", "ffmpeg|mediamtx|rq worker"], text=True,
                            stdout=subprocess.PIPE).stdout.splitlines()
    try:
        redis = subprocess.Popen(["redis-server", "--port", "6379", "--save", "",
                                  "--appendonly", "no"], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.STDOUT); processes.append(redis); wait_port(6379, redis)
        media_log = ROOT / "artifacts/phase-m5-mediamtx.log"; media_log.parent.mkdir(exist_ok=True)
        media_handle = media_log.open("wb")
        media = subprocess.Popen(["/opt/mediamtx/mediamtx",
                                  str(ROOT / "config/mediamtx_phase_6_5.yml")],
                                 stdout=media_handle, stderr=subprocess.STDOUT)
        processes.append(media); wait_port(18554, media); wait_port(18889, media); wait_port(19997, media)
        if os.environ.get("M5_BROWSER") == "1":
            vite_log = ROOT / "artifacts/phase-m5-vite.log"; vite_handle = vite_log.open("wb")
            vite = subprocess.Popen(["npx", "vite", "--host", "0.0.0.0"], cwd=str(WEB),
                                    stdout=vite_handle, stderr=subprocess.STDOUT)
            processes.append(vite); wait_port(5173, vite)
        repository = RedisTaskRepository.from_url(REDIS, NS); repository.client.flushdb()
        cancellation = RedisCancellationStore.from_url(REDIS, NS)
        queue = RQTaskQueue.from_url(REDIS, str(ROOT / "artifacts"), NS, NS, job_timeout=900)
        backend = RedisRecordingBackendClient(repository.client, NS)
        streams = RedisRecordingStreamGateway(repository.client, NS)
        service = SimulationService(repository, queue, backend,
                                    cancellation_store=cancellation,
                                    artifacts_root=ROOT / "artifacts")
        app = create_http_app(service, TOKEN, stream_gateway=streams)
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT / "src"); env.pop("DISPLAY", None)
        env.update({"SIMULATION_HEARTBEAT_INTERVAL": "1",
                    "SIMULATION_STREAM_RTSP_BASE": "rtsp://127.0.0.1:18554",
                    "SIMULATION_STREAM_PUBLIC_BASE_URL": "http://192.0.2.10:18889",
                    "SIMULATION_FFMPEG_PATH": "/opt/ffmpeg/bin/ffmpeg",
                    "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json"})
        worker_log = ROOT / "artifacts/phase-m5-worker.log"; worker_handle = worker_log.open("wb")
        worker = subprocess.Popen(["rq", "worker", "-u", REDIS, "-n", "m5-worker",
                                   "--logging_level", "INFO", NS], cwd=str(ROOT), env=env,
                                  stdout=worker_handle, stderr=subprocess.STDOUT)
        processes.append(worker)
        cases = ["s1_push", "s1_reach", "demo_s1_target_pose",
                 "demo_s1_line_tracking", "demo_s1_circle_tracking",
                 "demo_s1_joint_position", "demo_s1_joint_sweep"]
        selected = os.environ.get("M5_CASES")
        if selected:
            requested = [item.strip() for item in selected.split(",") if item.strip()]
            cases = [item for item in cases if item in requested]
        for index, task_name in enumerate(cases):
            browser = (
                task_name in ("s1_reach", "demo_s1_circle_tracking")
                and os.environ.get("M5_BROWSER") == "1"
            )
            run_id = "m5pass-%02d-%s" % (index + 1, task_name.replace("_", "-"))
            config = (training_config(task_name, browser) if task_name.startswith("s1_")
                      else demo_config(task_name, 70 if browser else 12,
                                       1280 if browser else 640,
                                       720 if browser else 480, 30 if browser else 15))
            response = asyncio.run(request(app, "POST", "/simulation/tasks",
                                           {"run_id": run_id, "project_id": PROJECT, "config": config}))
            if response.status_code != 202: raise RuntimeError(response.text)
            task_id = response.json()["task_id"]; descriptor = None; path_ready = False
            browser_result = None; whep_result = None
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                task = service.get_task(task_id)
                if task.status == TaskStatus.RUNNING and descriptor is None:
                    stream_response = asyncio.run(request(
                        app, "POST", "/simulation/tasks/%s/stream-session" % task_id,
                        {"camera_id": "overview"}))
                    if stream_response.status_code == 200:
                        descriptor = stream_response.json()
                        for _ in range(40):
                            path_ready, path_status = media_path_ready(descriptor["stream_id"])
                            if path_ready: break
                            time.sleep(.1)
                        whep_result = whep_gate(descriptor)
                        if browser and os.environ.get("M5_BROWSER") == "1":
                            evidence = ROOT / "artifacts" / "phase-m5-browser" / (task_name + ".json")
                            browser_result = browser_gate(
                                descriptor, task_name, run_id, evidence,
                            )
                        if os.environ.get("M5_STOP") == "1":
                            break
                if task.is_terminal: break
                time.sleep(.25)
            if descriptor is None: raise RuntimeError("no READY stream descriptor for " + task_name)
            if not service.get_task(task_id).is_terminal:
                asyncio.run(request(app, "POST", "/simulation/tasks/%s/stop" % task_id))
            terminal_deadline = time.monotonic() + 120
            while not service.get_task(task_id).is_terminal and time.monotonic() < terminal_deadline:
                time.sleep(.25)
            task = service.get_task(task_id)
            root = ROOT / "artifacts" / PROJECT / run_id
            video = root / "video" / "overview.mp4"
            history = streams.history_for_run(run_id)
            output["runs"].append({"run_id": run_id, "task_id": task_id,
                "environment_type": task_name, "camera_id": descriptor["camera_id"],
                "resolution": [descriptor["width"], descriptor["height"]],
                "fps": descriptor["fps"], "codec": "h264_nvenc",
                "stream_id": descriptor["stream_id"], "whep_endpoint": descriptor["whep_url"],
                "descriptor_fields": sorted(descriptor), "mediamtx_path_ready": path_ready,
                "stream_history_statuses": [item["status"] for item in history],
                "mp4_path": str(video), "mp4_bytes": video.stat().st_size if video.is_file() else 0,
                "mp4_probe": probe_mp4(video) if video.is_file() else {},
                "final_status": task.status.value, "whep": whep_result,
                "browser": browser_result})
        catalog = asyncio.run(request(app, "GET", "/simulation/task-catalog"))
        output["catalog"] = catalog.json(); output["catalog_status"] = catalog.status_code
        output["mediamtx_log"] = str(media_log); output["worker_log"] = str(worker_log)
        output["browser_tasks"] = [r["environment_type"] for r in output["runs"] if r["browser"]]
        output["passed"] = (
            catalog.status_code == 200 and len(output["catalog"].get("tasks", [])) == 7
            and all(r["mp4_bytes"] > 0 and r["mediamtx_path_ready"]
                    and "rtsp_internal_url" not in r["descriptor_fields"]
                    and r["final_status"] in ("completed", "stopped")
                    for r in output["runs"])
            and all(r["whep"] and r["whep"]["passed"] for r in output["runs"])
        )
    finally:
        for process in reversed(processes): stop_process(process)
        for handle_name in ("worker_handle", "vite_handle", "media_handle"):
            handle = locals().get(handle_name)
            if handle: handle.close()
        time.sleep(1)
        output["processes_before"] = before
        output["processes_after"] = subprocess.run(
            ["pgrep", "-af", "ffmpeg|mediamtx|rq worker"], text=True,
            stdout=subprocess.PIPE).stdout.splitlines()
    destination = ROOT / os.environ.get(
        "M5_RUNTIME_PATH", "doc/reports/phase_M5_video_platform_integration_runtime.json"
    )
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("M5_UNIFIED_VIDEO_GATE=" + json.dumps(output, sort_keys=True))
    if not output.get("passed"): raise SystemExit(2)


if __name__ == "__main__": run()
