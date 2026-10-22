#!/usr/bin/env python3
"""Real-GPU cooperative-stop smoke test for the isolated training Worker."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import time

from simulation_core import (
    FakeBackendClient,
    FakeScheduler,
    GPUWorker,
    InMemoryTaskRepository,
    SimulationService,
    TaskStatus,
    TrainingTaskConfig,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default="phase-4-3")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--num-updates", type=int, default=10000)
    parser.add_argument("--run-seconds", type=float, default=1.0)
    parser.add_argument("--running-timeout", type=float, default=60.0)
    parser.add_argument("--stop-timeout", type=float, default=10.0)
    return parser


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    args = build_parser().parse_args()
    run_id = args.run_id or "gpu-worker-stop-%s" % datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    repository = InMemoryTaskRepository()
    backend = FakeBackendClient()
    service = SimulationService(repository, FakeScheduler(), backend)
    config = TrainingTaskConfig(
        task_type="training",
        environment_type="s1_push",
        seed=7,
        rollout_steps=args.rollout_steps,
        num_updates=args.num_updates,
        checkpoint_every=1,
        episode_steps=32,
        physics_steps_per_action=2,
        update_epochs=1,
        batch_size=min(16, args.rollout_steps),
    )
    task = service.create_task(run_id, args.project_id, config.to_dict())
    worker = GPUWorker(
        service, args.artifacts_root, stop_timeout=args.stop_timeout
    )
    outcome = {"result": None, "error": None, "child_exit_time": None}

    def run_worker():
        try:
            outcome["result"] = worker.run(task.task_id)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            outcome["child_exit_time"] = utc_timestamp()

    parent_pid = os.getpid()
    thread = threading.Thread(target=run_worker, name="gpu-worker-smoke-driver")
    thread.start()
    deadline = time.monotonic() + args.running_timeout
    while service.get_task(task.task_id).status != TaskStatus.RUNNING:
        if not thread.is_alive():
            raise RuntimeError("Worker exited before running: %r" % outcome["error"])
        if time.monotonic() >= deadline:
            raise TimeoutError("task did not enter running before timeout")
        time.sleep(0.02)

    child_pid = worker.active_child_pid
    if args.run_seconds > 0:
        time.sleep(args.run_seconds)
    stop_request_time = utc_timestamp()
    stop_started = time.monotonic()
    stopped_task = worker.stop(task.task_id)
    stop_elapsed = time.monotonic() - stop_started
    thread.join(max(5.0, args.stop_timeout + 2.0))
    if thread.is_alive():
        raise RuntimeError("Worker thread remained alive after stop")
    if outcome["error"] is not None:
        raise outcome["error"]
    result = outcome["result"]
    child_alive = False
    if child_pid is not None:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            pass
        else:
            child_alive = True

    metrics_exists = bool(result.metrics_path) and Path(result.metrics_path).is_file()
    checkpoint_exists = bool(result.checkpoint_path) and Path(
        result.checkpoint_path
    ).is_file()
    print("GPU_WORKER_STOP_RESULT=success")
    print("parent_pid=%d" % parent_pid)
    print("child_pid=%s" % child_pid)
    print("stop_request_time=%s" % stop_request_time)
    print("child_exit_time=%s" % outcome["child_exit_time"])
    print("stop_elapsed_seconds=%.6f" % stop_elapsed)
    print("child_exit_code=%s" % worker.last_exit_code)
    print(
        "status_history=%s"
        % " -> ".join(item.status.value for item in backend.reported_tasks)
    )
    print("final_task_status=%s" % stopped_task.status.value)
    print("training_steps=%d" % result.training_steps)
    print("update_steps=%d" % result.update_steps)
    print("metrics_path=%s" % result.metrics_path)
    print("metrics_exists=%s" % metrics_exists)
    print("checkpoint_path=%s" % result.checkpoint_path)
    print("checkpoint_exists=%s" % checkpoint_exists)
    print("forced_termination=%s" % worker.last_stop_forced)
    print("child_alive_after_cleanup=%s" % child_alive)
    print("worker_idle=%s" % (not worker.is_busy))

    if stopped_task.status != TaskStatus.STOPPED or result.status != TaskStatus.STOPPED:
        raise RuntimeError("cooperative stop did not produce stopped")
    if worker.last_stop_forced:
        raise RuntimeError("smoke test unexpectedly used terminate fallback")
    if child_alive or worker.is_busy or worker.active_child_pid is not None:
        raise RuntimeError("training process or Worker state was not cleaned")


if __name__ == "__main__":
    main()
