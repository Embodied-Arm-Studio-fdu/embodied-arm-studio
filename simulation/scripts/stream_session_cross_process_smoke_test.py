#!/usr/bin/env python3
"""Real Redis/RQ/HTTP cross-process stream-session smoke orchestration."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types

import httpx
import redis


ROOT = Path(__file__).resolve().parents[1]


def wait_tcp(port, process, label, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("%s exited with %s" % (label, process.returncode))
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.2):
                return
        except OSError:
            time.sleep(.05)
    raise TimeoutError("%s did not listen on port %d" % (label, port))


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def request(client, method, path, token, **kwargs):
    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = "Bearer " + token
    response = client.request(method, path, headers=headers, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError("%s %s returned %s: %s" % (
            method, path, response.status_code, response.text,
        ))
    return response


def wait_task(client, task_id, token, wanted, timeout=180):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = request(
            client, "GET", "/simulation/tasks/" + task_id, token
        )
        last = response.json()
        if last["status"] in wanted:
            return last
        time.sleep(.1)
    raise TimeoutError("task last status %r, wanted %r" % (last, wanted))


def wait_redis_stream(connection, key, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = connection.get(key)
        if value is not None:
            record = json.loads(value)
            if record.get("status") in ("STARTING", "READY"):
                return record
        time.sleep(.1)
    raise TimeoutError("stream descriptor was not published at %s" % key)


def load_backend_client():
    """Read the real backend client without importing unrelated backend DB code."""
    source = Path("/home/user/embodied-arm-studio/robotics/simulation")
    robotics = types.ModuleType("robotics")
    robotics.__path__ = []
    package = types.ModuleType("robotics.simulation")
    package.__path__ = [str(source)]
    sys.modules["robotics"] = robotics
    sys.modules["robotics.simulation"] = package
    for name in ("contracts", "client"):
        qualified = "robotics.simulation." + name
        spec = importlib.util.spec_from_file_location(
            qualified, source / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return sys.modules["robotics.simulation.client"].SimulationClient


def process_tree(root_pid):
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,comm=,args="], text=True
    )
    rows = []
    parents = {int(root_pid)}
    changed = True
    parsed = []
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) >= 3:
            parsed.append((int(fields[0]), int(fields[1]), fields[2],
                           fields[3] if len(fields) == 4 else ""))
    while changed:
        changed = False
        for row in parsed:
            if row[1] in parents and row[0] not in parents:
                parents.add(row[0])
                changed = True
    for row in parsed:
        if row[0] in parents:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-port", type=int, default=16389)
    parser.add_argument("--http-port", type=int, default=18091)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    namespace = "phase62-cross-process:%s" % stamp
    queue_name = "phase62-cross-process-%s" % stamp
    run_id = "phase62-cross-process-%s" % stamp
    project_id = "phase-6-2-cross-process"
    token = "phase62-smoke-token-" + stamp
    redis_url = "redis://127.0.0.1:%d/0" % args.redis_port
    base_url = "http://127.0.0.1:%d" % args.http_port
    temp_root = Path(tempfile.mkdtemp(prefix="phase62-cross-process-"))
    artifacts_root = temp_root / "artifacts"
    logs = temp_root / "logs"
    logs.mkdir(parents=True)
    processes = {"redis": None, "mediamtx": None, "http": None, "rq": None}
    handles = []
    task_id = None
    worker_tree = []

    def start(name, command, env=None):
        handle = (logs / (name + ".log")).open("w", encoding="utf-8")
        handles.append(handle)
        processes[name] = subprocess.Popen(
            command, cwd=str(ROOT), env=env, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        return processes[name]

    try:
        redis_process = start("redis", [
            "redis-server", "--port", str(args.redis_port), "--bind", "127.0.0.1",
            "--save", "", "--appendonly", "no", "--dir", str(temp_root),
        ])
        wait_tcp(args.redis_port, redis_process, "redis-server")
        connection = redis.Redis.from_url(redis_url, decode_responses=True)
        if connection.ping() is not True:
            raise RuntimeError("real Redis PING failed")

        mediamtx = start("mediamtx", [
            "/opt/mediamtx/mediamtx",
            str(ROOT / "config" / "mediamtx_phase_6_5.yml"),
        ])
        wait_tcp(18554, mediamtx, "MediaMTX")
        wait_tcp(18889, mediamtx, "MediaMTX WebRTC")

        common_env = dict(os.environ)
        common_env.update({
            "PYTHONPATH": str(ROOT / "src"),
            "SIMULATION_REDIS_URL": redis_url,
            "SIMULATION_REDIS_NAMESPACE": namespace,
            "SIMULATION_RQ_QUEUE": queue_name,
            "SIMULATION_SERVICE_TOKEN": token,
            "SIMULATION_HTTP_HOST": "127.0.0.1",
            "SIMULATION_HTTP_PORT": str(args.http_port),
            "SIMULATION_ARTIFACTS_ROOT": str(artifacts_root),
            "SIMULATION_HEARTBEAT_INTERVAL": "0.5",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        })
        common_env.pop("DISPLAY", None)
        http_process = start(
            "http", [sys.executable, "scripts/http_adapter.py"], common_env
        )
        wait_tcp(args.http_port, http_process, "HTTP Adapter")

        with httpx.Client(base_url=base_url, timeout=10, trust_env=False) as client:
            config = {
                "task_type": "training", "environment_type": "s1_push",
                "seed": 62, "rollout_steps": 16, "num_updates": 10000,
                "checkpoint_every": 1, "episode_steps": 64,
                "physics_steps_per_action": 2, "update_epochs": 1,
                "batch_size": 16, "device": "cuda:0", "video_enabled": True,
                "camera_id": "overview", "video_fps": 15.0,
                "video_width": 640, "video_height": 480,
            }
            created_response = request(
                client, "POST", "/simulation/tasks", token,
                json={"run_id": run_id, "project_id": project_id, "config": config},
            )
            created = created_response.json()
            task_id = created["task_id"]

            rq_process = start("rq", [
                "rq", "worker", "--burst", "-u", redis_url,
                "-n", "phase62-cross-process-worker-" + stamp,
                "--logging_level", "INFO", queue_name,
            ], common_env)

            running = wait_task(
                client, task_id, token, {"running", "failed", "completed"},
                timeout=args.timeout,
            )
            if running["status"] != "running":
                raise RuntimeError("task did not stay RUNNING: %s" % running)

            stream_key = "%s:run:%s:stream" % (namespace, run_id)
            history_key = "%s:run:%s:stream_history" % (namespace, run_id)
            redis_descriptor = wait_redis_stream(
                connection, stream_key, timeout=args.timeout
            )
            before = wait_task(client, task_id, token, {"running"}, timeout=10)
            history_before = connection.llen(history_key)

            first_response = request(
                client, "POST",
                "/simulation/tasks/%s/stream-session" % task_id, token,
                json={"camera_id": "overview"},
            )
            second_response = request(
                client, "POST",
                "/simulation/tasks/%s/stream-session" % task_id, token,
                json={"camera_id": "overview"},
            )
            first, second = first_response.json(), second_response.json()
            required = {
                "stream_id", "run_id", "task_id", "camera_id", "status",
                "webrtc_url", "whep_url", "width", "height", "fps",
            }
            if set(first) != required or set(second) != required:
                raise RuntimeError("StreamDescriptor fields mismatch")
            if first != redis_descriptor or second != redis_descriptor:
                raise RuntimeError("HTTP descriptor is not the Redis descriptor")
            if first["stream_id"] != second["stream_id"]:
                raise RuntimeError("stream-session request was not idempotent")
            if first["run_id"] != run_id or first["task_id"] != task_id:
                raise RuntimeError("descriptor identity mismatch")
            if first["camera_id"] != "overview" or not first["stream_id"]:
                raise RuntimeError("descriptor camera/stream ID invalid")
            if first["width"] <= 0 or first["height"] <= 0 or first["fps"] <= 0:
                raise RuntimeError("descriptor dimensions/fps invalid")
            if first["status"] not in ("STARTING", "READY"):
                raise RuntimeError("descriptor status invalid")
            history_after = connection.llen(history_key)
            after = wait_task(client, task_id, token, {"running"}, timeout=10)
            if history_before != history_after:
                raise RuntimeError("HTTP queries published another stream record")

            SimulationClient = load_backend_client()
            with SimulationClient(base_url, service_token=token) as backend_client:
                backend_first = backend_client.create_stream_session(
                    task_id, "overview"
                )
                backend_second = backend_client.create_stream_session(
                    task_id, "overview"
                )
            if backend_first.stream_id != backend_second.stream_id:
                raise RuntimeError("Backend SimulationClient was not idempotent")
            if backend_first.stream_id != first["stream_id"]:
                raise RuntimeError("Backend Client descriptor differs from standard HTTP")

            tree = process_tree(rq_process.pid)
            worker_tree = list(tree)
            if http_process.pid == rq_process.pid:
                raise RuntimeError("HTTP and RQ unexpectedly share a PID")

            stopped_request = request(
                client, "POST", "/simulation/tasks/%s/stop" % task_id, token
            ).json()
            final = wait_task(
                client, task_id, token, {"stopped", "completed", "failed"},
                timeout=args.timeout,
            )

        rq_process.wait(timeout=30)
        print("SMOKE_RESULT=PASS")
        print("redis_url=%s" % redis_url)
        print("namespace=%s" % namespace)
        print("queue=%s" % queue_name)
        print("redis_pid=%s" % redis_process.pid)
        print("http_pid=%s" % http_process.pid)
        print("rq_pid=%s" % rq_process.pid)
        print("rq_process_tree=%s" % json.dumps(tree))
        print("run_id=%s" % run_id)
        print("task_id=%s" % task_id)
        print("project_id=%s" % project_id)
        print("initial_status=%s" % created["status"])
        print("before_status=%s" % before["status"])
        print("after_status=%s" % after["status"])
        print("stream_key=%s" % stream_key)
        print("history_key=%s" % history_key)
        print("history_before_queries=%s" % history_before)
        print("history_after_queries=%s" % history_after)
        print("stream_id_1=%s" % first["stream_id"])
        print("stream_id_2=%s" % second["stream_id"])
        print("backend_stream_id_1=%s" % backend_first.stream_id)
        print("backend_stream_id_2=%s" % backend_second.stream_id)
        print("descriptor=%s" % json.dumps(first, sort_keys=True))
        print("stop_request_status=%s" % stopped_request["status"])
        print("final_status=%s" % final["status"])
    finally:
        for name in ("rq", "http", "mediamtx", "redis"):
            stop_process(processes[name])
        for handle in handles:
            handle.close()
        leftovers = [
            (name, process.pid) for name, process in processes.items()
            if process is not None and process.poll() is None
        ]
        if leftovers:
            print("cleanup_leftovers=%s" % leftovers, file=sys.stderr)
        else:
            print("cleanup_leftovers=[]")
        descendant_pids = [str(row[0]) for row in worker_tree[1:]]
        descendant_residue = ""
        if descendant_pids:
            checked = subprocess.run(
                ["ps", "-p", ",".join(descendant_pids),
                 "-o", "pid=,ppid=,stat=,comm=,args="],
                text=True, stdout=subprocess.PIPE, check=False,
            )
            descendant_residue = checked.stdout.strip()
        print("cleanup_descendant_residue=%s" % (
            descendant_residue if descendant_residue else "[]"
        ))
        shutil.rmtree(str(temp_root), ignore_errors=True)
        if descendant_residue and sys.exc_info()[0] is None:
            raise RuntimeError(
                "RQ/training descendant residue remained after cleanup: %s" %
                descendant_residue
            )


if __name__ == "__main__":
    main()
