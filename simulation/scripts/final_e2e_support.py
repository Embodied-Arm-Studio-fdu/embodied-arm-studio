#!/usr/bin/env python3
"""Shared Phase 6.6 Mock Backend -> Redis/RQ final E2E harness."""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from simulation_core import (
    RedisCancellationStore, RedisRecordingBackendClient,
    RedisRecordingStreamGateway, RedisTaskRepository, RQTaskQueue,
    SimulationService, StreamDescriptor, TaskStatus, TrainingTaskConfig,
)
from simulation_core.input_events import (
    MockInputWebSocketClient, MockInputWebSocketServer, RecordingInputEventSink,
)


ROOT = Path(__file__).resolve().parents[1]
REDIS_URL = "redis://127.0.0.1:6379/0"


def configure_aiortc_test_site():
    """Load the isolated Phase 6.4 client dependencies for test code only."""
    value = os.environ.get("PHASE64_AIORTC_SITE")
    if not value:
        raise RuntimeError("PHASE64_AIORTC_SITE is required for the WHEP test client")
    site = Path(value).resolve()
    if not site.is_dir():
        raise RuntimeError("PHASE64_AIORTC_SITE is not a directory: %s" % site)
    site_text = str(site)
    if site_text not in sys.path:
        sys.path.insert(0, site_text)

    import aiortc
    import av

    for name, module in (("aiortc", aiortc), ("av", av)):
        module_path = Path(module.__file__).resolve()
        if site != module_path and site not in module_path.parents:
            raise RuntimeError("%s loaded outside PHASE64_AIORTC_SITE: %s" %
                               (name, module_path))
    print("aiortc_version=%s" % aiortc.__version__)
    print("aiortc_module_path=%s" % Path(aiortc.__file__).resolve())
    print("av_version=%s" % av.__version__)
    print("av_module_path=%s" % Path(av.__file__).resolve())


def process_snapshot(root_pid):
    """Capture the current Linux descendant tree without changing process state."""
    records = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            fields = {}
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key] = value.strip()
            pid = int(fields["Pid"]); ppid = int(fields["PPid"])
            cmdline = (status_path.parent / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace").strip()
            records[pid] = {"pid": pid, "ppid": ppid, "state": fields.get("State", ""),
                            "name": fields.get("Name", ""), "cmdline": cmdline}
        except (FileNotFoundError, PermissionError, KeyError, ValueError, OSError):
            continue
    selected = set(); frontier = {int(root_pid)}
    while frontier:
        selected.update(frontier)
        frontier = {pid for pid, item in records.items()
                    if item["ppid"] in frontier and pid not in selected}
    return [records[pid] for pid in sorted(selected) if pid in records]


def stop_process(process):
    if process is None: return
    if process.poll() is None: process.terminate()
    try: process.wait(timeout=8)
    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=3)


def wait_tcp(port, process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None: raise RuntimeError("MediaMTX exited")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2): return
        except OSError: time.sleep(.05)
    raise TimeoutError("port %d not ready" % port)


def wait_task(service, task_id, statuses, timeout=120):
    statuses = set(statuses); deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = service.get_task(task_id)
        if task.status in statuses: return task
        time.sleep(.05)
    raise TimeoutError("task did not reach %s" % [item.value for item in statuses])


def wait_stream(gateway, run_id, status="READY", timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = gateway.history_for_run(run_id)
        for item in history:
            if item["status"] == status: return item
        time.sleep(.05)
    raise TimeoutError("stream did not reach %s" % status)


def descriptor(record):
    return StreamDescriptor(record["stream_id"], record["run_id"], record["task_id"], record["status"],
        record["webrtc_url"], record["whep_url"], record["camera_id"],
        int(record["width"]), int(record["height"]), float(record["fps"]))


def input_event(number, kind, run_id, task_id, **fields):
    value = {"event_id": "input-%d" % number, "run_id": run_id,
        "task_id": task_id, "type": kind,
        "occurred_at": datetime.now(timezone.utc).isoformat()}
    value.update(fields); return value


async def send_inputs(run_id, task_id):
    sink = RecordingInputEventSink(); server = MockInputWebSocketServer(sink)
    await server.start()
    try:
        values = [input_event(1, "KEYBOARD", run_id, task_id, key="W", action="pressed"),
            input_event(2, "KEYBOARD", run_id, task_id, key="W", action="released"),
            input_event(3, "MOUSE_MOVE", run_id, task_id, x=.4, y=.6),
            input_event(4, "MOUSE_BUTTON", run_id, task_id, button="left", action="pressed", x=.4, y=.6),
            input_event(5, "MOUSE_BUTTON", run_id, task_id, button="left", action="released", x=.4, y=.6)]
        replies = await MockInputWebSocketClient("ws://127.0.0.1:%d" % server.port).send(values)
        if not all(item["ok"] for item in replies): raise RuntimeError("valid input rejected")
        return len(sink.events)
    finally: await server.close()


def validate_artifacts(task, callback, partial=False):
    run_root = ROOT / "artifacts" / task.project_id / task.run_id
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file(): raise RuntimeError("manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["final_status"] != task.status.value: raise RuntimeError("manifest status mismatch")
    for item in manifest["artifacts"]:
        path = run_root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise RuntimeError("artifact missing/size mismatch: %s" % path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]: raise RuntimeError("artifact hash mismatch")
    if not partial:
        required = {"config/training_config.json", "logs/stdout.log", "logs/stderr.log",
            "logs/task.log", "metrics/training.jsonl", "checkpoints/model_latest.pt",
            "checkpoints/model_best.pt"}
        if not required.issubset({item["path"] for item in manifest["artifacts"]}):
            raise RuntimeError("completed artifacts incomplete")
    for item in callback["artifacts"]:
        if not (run_root / item["path"]).is_file(): raise RuntimeError("callback artifact missing")
    return manifest


class Harness:
    def __init__(self, scenario):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.scenario = scenario
        self.run_id = "phase-6-6-%s-%s" % (scenario, stamp)
        self.namespace = "phase66:%s:%s" % (scenario, stamp)
        self.queue_name = "phase66-%s-%s" % (scenario, stamp)
        self.project_id = "phase-6-6"
        self.mediamtx = self.rq_worker = None
        self.repository = RedisTaskRepository.from_url(REDIS_URL, self.namespace)
        self.cancellation = RedisCancellationStore.from_url(REDIS_URL, self.namespace)
        self.queue = RQTaskQueue.from_url(REDIS_URL, str(ROOT / "artifacts"),
            self.queue_name, self.namespace)
        self.backend = RedisRecordingBackendClient(self.repository.client, self.namespace)
        self.streams = RedisRecordingStreamGateway(self.repository.client, self.namespace)
        self.service = SimulationService(self.repository, self.queue, self.backend,
            cancellation_store=self.cancellation, artifacts_root=ROOT / "artifacts")

    def start_mediamtx(self):
        self.mediamtx = subprocess.Popen(["/opt/mediamtx/mediamtx",
            str(ROOT / "config/mediamtx_phase_6_5.yml")], stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)
        wait_tcp(18554, self.mediamtx); wait_tcp(18889, self.mediamtx)

    def start_worker(self):
        env = dict(os.environ); env["PYTHONPATH"] = str(ROOT / "src")
        env["SIMULATION_HEARTBEAT_INTERVAL"] = "0.5"
        if self.scenario == "failed":
            env["SIMULATION_TEST_FAILURE_INJECTION"] = "training_after_running"
        env.pop("DISPLAY", None)
        env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
        self.rq_worker = subprocess.Popen(["rq", "worker", "--burst", "-u", REDIS_URL,
            "-n", "phase66-%s" % self.scenario, "--logging_level", "WARNING",
            self.queue_name], cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)

    def submit(self, updates):
        config = TrainingTaskConfig(task_type="training", environment_type="s1_push",
            seed=66, rollout_steps=16, num_updates=updates, checkpoint_every=1,
            episode_steps=64, physics_steps_per_action=2, update_epochs=1,
            batch_size=16, video_enabled=True, run_mode="smoke")
        return self.service.create_task(self.run_id, self.project_id, config.to_dict())

    def cleanup(self):
        # A burst worker normally exits after it has wait/reaped its work horse.
        # Give that bookkeeping a bounded chance before the generic terminate
        # fallback, especially when a later optional client check raises.
        if self.rq_worker is not None and self.rq_worker.poll() is None:
            try: self.rq_worker.wait(timeout=8)
            except subprocess.TimeoutExpired: pass
        stop_process(self.rq_worker); stop_process(self.mediamtx)


def run_scenario(scenario):
    harness = Harness(scenario); task = None
    try:
        harness.start_mediamtx()
        updates = 20 if scenario == "completed" else 10000
        task = harness.submit(updates)
        harness.start_worker()
        ready = wait_stream(harness.streams, task.run_id, "READY")
        runtime_processes = process_snapshot(harness.rq_worker.pid)
        input_count = asyncio.run(send_inputs(task.run_id, task.task_id))
        video = None
        if scenario == "completed":
            configure_aiortc_test_site()
            sys.path.insert(0, str(ROOT / "scripts"))
            from webrtc_test_client import receive_video_sync
            deadline = time.monotonic() + 30
            last_error = None
            while time.monotonic() < deadline:
                try:
                    video = receive_video_sync(
                        descriptor(ready).whep_url, frames=5, timeout=15
                    )
                    break
                except Exception as exc:
                    last_error = exc; time.sleep(.2)
            if video is None:
                raise RuntimeError("WHEP client never received video: %s" % last_error)
        elif scenario == "stopped":
            wait_task(harness.service, task.task_id, {TaskStatus.RUNNING})
            harness.service.request_stop(task.task_id)
        final_status = {"completed": TaskStatus.COMPLETED, "stopped": TaskStatus.STOPPED,
                        "failed": TaskStatus.FAILED}[scenario]
        final = wait_task(harness.service, task.task_id, {final_status})
        stopped_stream = wait_stream(harness.streams, task.run_id, "STOPPED")
        events = harness.backend.events_for_run(task.run_id)
        callbacks = harness.backend.terminal_callbacks_for_run(task.run_id)
        if len(callbacks) != 1 or callbacks[0]["status"] != final_status.value:
            raise RuntimeError("terminal callback mismatch")
        sequences = [item["sequence"] for item in events] + [callbacks[0]["sequence"]]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise RuntimeError("run sequence is not strictly increasing")
        manifest = validate_artifacts(final, callbacks[0], partial=scenario != "completed")
        redis_pid = int(harness.repository.client.info("server")["process_id"])
        return {"harness": harness, "task": final, "ready": ready,
            "stopped_stream": stopped_stream, "events": events,
            "callback": callbacks[0], "manifest": manifest, "input_count": input_count,
            "video": video, "runtime_processes": runtime_processes,
            "redis_pid": redis_pid, "rq_worker_pid": harness.rq_worker.pid,
            "mediamtx_pid": harness.mediamtx.pid}
    except Exception:
        harness.cleanup(); raise


def finish(result): result["harness"].cleanup()
