#!/usr/bin/env python3
"""Redis/RQ queue smoke scenarios; requires one external simulation-gpu worker."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from simulation_core import (
    FakeBackendClient,
    RedisCancellationStore,
    RedisTaskRepository,
    RQTaskQueue,
    SimulationService,
    TaskStatus,
    TrainingTaskConfig,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    parser.add_argument("--queue-name", default="simulation-gpu")
    parser.add_argument("--namespace")
    parser.add_argument("--project-id", default="phase-4-5")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--scenario",
        choices=("sequential", "queued-cancel", "running-stop", "all"),
        default="all",
    )
    return parser


def config(num_updates):
    return TrainingTaskConfig(
        task_type="training",
        environment_type="s1_push",
        seed=41,
        rollout_steps=16,
        num_updates=num_updates,
        checkpoint_every=1,
        episode_steps=32,
        physics_steps_per_action=2,
        update_epochs=1,
        batch_size=16,
    ).to_dict()


def wait_for(service, task_id, statuses, timeout):
    statuses = set(statuses)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = service.get_task(task_id)
        if task.status in statuses:
            return task
        time.sleep(0.05)
    raise TimeoutError(
        "task %s did not reach %s" % (
            task_id,
            ",".join(sorted(status.value for status in statuses)),
        )
    )


def assert_manifest(root, task, status):
    manifest_path = root / task.project_id / task.run_id / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("manifest missing: %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["task_id"] != task.task_id or manifest["final_status"] != status:
        raise RuntimeError("manifest identity/status mismatch: %s" % manifest_path)
    return manifest_path


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest_files(manifest_path, required_paths=()):
    run_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = {item["path"]: item for item in manifest["artifacts"]}
    for relative_path, item in metadata.items():
        artifact_path = run_root / relative_path
        if not artifact_path.is_file():
            raise RuntimeError("manifest artifact missing: %s" % artifact_path)
        if artifact_path.stat().st_size != item["size_bytes"]:
            raise RuntimeError("manifest size mismatch: %s" % artifact_path)
        if file_sha256(artifact_path) != item["sha256"]:
            raise RuntimeError("manifest sha256 mismatch: %s" % artifact_path)
    missing = set(required_paths).difference(metadata)
    if missing:
        raise RuntimeError("required artifacts missing: %s" % sorted(missing))
    return manifest


def task_log_evidence(root, task):
    task_log = root / task.project_id / task.run_id / "logs" / "task.log"
    records = [
        json.loads(line)
        for line in task_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    history = ["queued"]
    child_pid = None
    for record in records:
        if record["event"] == "task_started":
            status = record.get("status")
            child_pid = record.get("child_pid") or child_pid
        elif record["event"] == "status_changed":
            status = record.get("status")
        elif record["event"] == "final_result":
            status = record.get("final_status")
        else:
            status = None
        if status and history[-1] != status:
            history.append(status)
    return records, history, child_pid


def wait_for_file(service, task_id, path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = service.get_task(task_id)
        if path.is_file():
            return task
        if task.is_terminal:
            raise RuntimeError("task became %s before %s existed" % (task.status.value, path))
        time.sleep(0.05)
    raise TimeoutError("file did not appear: %s" % path)


def wait_for_job_status(queue, task_id, statuses, timeout):
    statuses = set(statuses)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = queue.get_job_status(task_id)
        if status in statuses:
            return status
        time.sleep(0.05)
    raise TimeoutError("RQ job %s did not reach %s" % (task_id, sorted(statuses)))


def sequential_scenario(service, queue, root, prefix, project_id, timeout):
    first = service.create_task(
        "%s-sequential-a" % prefix, project_id, config(num_updates=1)
    )
    second = service.create_task(
        "%s-sequential-b" % prefix, project_id, config(num_updates=1)
    )
    wait_for(service, first.task_id, {TaskStatus.RUNNING}, timeout)
    if service.get_task(second.task_id).status != TaskStatus.QUEUED:
        raise RuntimeError("task B did not remain queued while A was running")
    first_done = wait_for(service, first.task_id, {TaskStatus.COMPLETED}, timeout)
    second_done = wait_for(service, second.task_id, {TaskStatus.COMPLETED}, timeout)
    first_manifest_path = assert_manifest(root, first_done, "completed")
    second_manifest_path = assert_manifest(root, second_done, "completed")
    required = {
        "checkpoints/model_latest.pt",
        "checkpoints/model_best.pt",
        "metrics/training.jsonl",
        "config/training_config.json",
        "logs/stdout.log",
        "logs/stderr.log",
        "logs/task.log",
    }
    first_manifest = validate_manifest_files(first_manifest_path, required)
    second_manifest = validate_manifest_files(second_manifest_path, required)
    if first_manifest_path.parent == second_manifest_path.parent:
        raise RuntimeError("sequential runs did not use isolated artifact roots")
    first_end = datetime.fromisoformat(first_manifest["ended_at"])
    second_start = datetime.fromisoformat(second_manifest["started_at"])
    if first_end > second_start:
        raise RuntimeError("A/B training artifact intervals overlapped")
    _, first_history, first_child_pid = task_log_evidence(root, first_done)
    _, second_history, second_child_pid = task_log_evidence(root, second_done)
    print("sequential_a_task_id=%s" % first.task_id)
    print("sequential_b_task_id=%s" % second.task_id)
    print("sequential_a_run_id=%s" % first.run_id)
    print("sequential_b_run_id=%s" % second.run_id)
    print("sequential_a_child_pid=%s" % first_child_pid)
    print("sequential_b_child_pid=%s" % second_child_pid)
    print("sequential_a_history=%s" % "->".join(first_history))
    print("sequential_b_history=%s" % "->".join(second_history))
    print("sequential_artifact_intervals_overlap=false")
    print("sequential_a_job_status=%s" % queue.get_job_status(first.task_id))
    print("sequential_b_job_status=%s" % queue.get_job_status(second.task_id))


def queued_cancel_scenario(service, queue, root, prefix, project_id, timeout):
    first = service.create_task(
        "%s-queued-cancel-a" % prefix, project_id, config(num_updates=4)
    )
    second = service.create_task(
        "%s-queued-cancel-b" % prefix, project_id, config(num_updates=1)
    )
    wait_for(service, first.task_id, {TaskStatus.RUNNING}, timeout)
    stopped = service.request_stop(second.task_id)
    if stopped.status != TaskStatus.STOPPED:
        raise RuntimeError("queued task B did not transition directly to stopped")
    assert_manifest(root, stopped, "stopped")
    run_root = root / stopped.project_id / stopped.run_id
    if (run_root / "checkpoints/model_latest.pt").exists():
        raise RuntimeError("queued cancellation created a fake checkpoint")
    if (run_root / "metrics/training.jsonl").exists():
        raise RuntimeError("queued cancellation created fake training metrics")
    first_done = wait_for(service, first.task_id, {TaskStatus.COMPLETED}, timeout)
    records, history, child_pid = task_log_evidence(root, stopped)
    if child_pid is not None:
        raise RuntimeError("queued cancellation recorded a training child PID")
    _, first_history, first_child_pid = task_log_evidence(root, first_done)
    print("queued_cancel_a_task_id=%s" % first.task_id)
    print("queued_cancel_b_task_id=%s" % second.task_id)
    print("queued_cancel_a_child_pid=%s" % first_child_pid)
    print("queued_cancel_b_child_pid=%s" % child_pid)
    print("queued_cancel_a_history=%s" % "->".join(first_history))
    print("queued_cancel_b_history=%s" % "->".join(history))
    print("queued_cancel_b_task_started_events=%s" % sum(
        record["event"] == "task_started" and record.get("child_pid") is not None
        for record in records
    ))
    print("queued_cancel_job_status=%s" % queue.get_job_status(second.task_id))


def running_stop_scenario(
    service, queue, cancellation, root, prefix, project_id, timeout
):
    task = service.create_task(
        "%s-running-stop" % prefix, project_id, config(num_updates=10000)
    )
    wait_for(service, task.task_id, {TaskStatus.RUNNING}, timeout)
    run_root = root / task.project_id / task.run_id
    latest_checkpoint = run_root / "checkpoints" / "model_latest.pt"
    wait_for_file(service, task.task_id, latest_checkpoint, timeout)
    stop_requested_at = datetime.now(timezone.utc)
    stopping = service.request_stop(task.task_id)
    cancel_flag_after_request = cancellation.is_requested(task.task_id)
    if stopping.status not in (TaskStatus.STOPPING, TaskStatus.STOPPED):
        raise RuntimeError("running stop did not enter stopping/stopped")
    if not cancel_flag_after_request:
        raise RuntimeError("Redis cancellation flag was not set")
    stopped = wait_for(service, task.task_id, {TaskStatus.STOPPED}, timeout)
    stopped_at = datetime.now(timezone.utc)
    job_status = wait_for_job_status(queue, task.task_id, {"finished"}, timeout)
    manifest_path = assert_manifest(root, stopped, "stopped")
    manifest = validate_manifest_files(
        manifest_path,
        {"checkpoints/model_latest.pt", "metrics/training.jsonl", "logs/task.log"},
    )
    records, history, child_pid = task_log_evidence(root, stopped)
    events = {record["event"] for record in records}
    if not {"stop_requested", "stop_observed"}.issubset(events):
        raise RuntimeError("training child did not log cooperative stop evidence")
    print("running_stop_task_id=%s" % task.task_id)
    print("running_stop_run_id=%s" % task.run_id)
    print("running_stop_child_pid=%s" % child_pid)
    if stopping.status == TaskStatus.STOPPING and "stopping" not in history:
        history.insert(-1, "stopping")
    print("running_stop_request_status=%s" % stopping.status.value)
    print("running_stop_final_status=%s" % stopped.status.value)
    print("running_stop_history=%s" % "->".join(history))
    print("running_stop_job_status=%s" % job_status)
    print("running_stop_cancel_flag_after_request=%s" % str(cancel_flag_after_request).lower())
    print("running_stop_cancel_flag_after_finish=%s" % str(cancellation.is_requested(task.task_id)).lower())
    print("running_stop_elapsed_seconds=%.6f" % (stopped_at - stop_requested_at).total_seconds())
    print("running_stop_training_steps=%s" % manifest["training_steps"])
    print("running_stop_update_steps=%s" % manifest["update_steps"])


def main():
    args = build_parser().parse_args()
    print("submitter_pid=%s" % os.getpid())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    namespace = args.namespace or "simulation-smoke:%s" % stamp
    repository = RedisTaskRepository.from_url(args.redis_url, namespace=namespace)
    cancellation = RedisCancellationStore.from_url(
        args.redis_url, namespace=namespace
    )
    queue = RQTaskQueue.from_url(
        args.redis_url,
        artifacts_root=str(args.artifacts_root),
        queue_name=args.queue_name,
        namespace=namespace,
    )
    service = SimulationService(
        repository,
        queue,
        FakeBackendClient(),
        cancellation_store=cancellation,
        artifacts_root=args.artifacts_root,
    )
    if args.scenario in ("sequential", "all"):
        sequential_scenario(
            service, queue, args.artifacts_root, stamp, args.project_id, args.timeout
        )
    if args.scenario in ("queued-cancel", "all"):
        queued_cancel_scenario(
            service, queue, args.artifacts_root, stamp, args.project_id, args.timeout
        )
    if args.scenario in ("running-stop", "all"):
        running_stop_scenario(
            service,
            queue,
            cancellation,
            args.artifacts_root,
            stamp,
            args.project_id,
            args.timeout,
        )
    print("RQ_QUEUE_SMOKE_RESULT=success")
    print("namespace=%s" % namespace)


if __name__ == "__main__":
    main()
