#!/usr/bin/env python3
"""Real-GPU short training and ArtifactFinalizer integration smoke test."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

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
    parser.add_argument("--project-id", default="phase-4-4")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--num-updates", type=int, default=2)
    return parser


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    args = build_parser().parse_args()
    run_id = args.run_id or "artifact-finalization-%s" % datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    repository = InMemoryTaskRepository()
    backend = FakeBackendClient()
    service = SimulationService(repository, FakeScheduler(), backend)
    config = TrainingTaskConfig(
        task_type="training",
        environment_type="s1_push",
        seed=11,
        rollout_steps=args.rollout_steps,
        num_updates=args.num_updates,
        checkpoint_every=1,
        episode_steps=32,
        physics_steps_per_action=2,
        update_epochs=1,
        batch_size=min(16, args.rollout_steps),
    )
    task = service.create_task(run_id, args.project_id, config.to_dict())
    worker = GPUWorker(service, args.artifacts_root)
    parent_pid = os.getpid()
    result = worker.run(task.task_id)
    child_pid = worker.last_child_pid
    run_root = Path(result.artifact_root)
    expected_paths = (
        "checkpoints/model_latest.pt",
        "checkpoints/model_best.pt",
        "metrics/training.jsonl",
        "logs/stdout.log",
        "logs/stderr.log",
        "logs/task.log",
        "config/training_config.json",
        "manifest.json",
    )
    missing = [path for path in expected_paths if not (run_root / path).is_file()]
    if missing:
        raise RuntimeError("missing finalized artifacts: %s" % ", ".join(missing))

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    manifest_metadata_valid = True
    for item in manifest["artifacts"]:
        relative_path = Path(item["path"])
        path = run_root / relative_path
        if relative_path.is_absolute():
            manifest_metadata_valid = False
        if path.stat().st_size != item["size_bytes"]:
            manifest_metadata_valid = False
        if file_sha256(path) != item["sha256"]:
            manifest_metadata_valid = False

    child_alive = False
    if child_pid is not None:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            pass
        else:
            child_alive = True

    status_history = " -> ".join(
        reported.status.value for reported in backend.reported_tasks
    )
    print("ARTIFACT_FINALIZATION_RESULT=success")
    print("parent_pid=%d" % parent_pid)
    print("child_pid=%s" % child_pid)
    print("child_exit_code=%s" % worker.last_exit_code)
    print("status_history=%s" % status_history)
    print("final_status=%s" % result.status.value)
    print("training_steps=%d" % result.training_steps)
    print("update_steps=%d" % result.update_steps)
    print("artifact_root=%s" % result.artifact_root)
    for relative_path in expected_paths:
        print("artifact=%s size=%d" % (
            relative_path,
            (run_root / relative_path).stat().st_size,
        ))
    print("manifest_artifact_count=%d" % len(manifest["artifacts"]))
    print("manifest_metadata_valid=%s" % manifest_metadata_valid)
    print("child_alive_after_cleanup=%s" % child_alive)
    print("worker_idle=%s" % (not worker.is_busy))

    expected_history = (
        "queued -> initializing -> running -> finalizing -> completed"
    )
    if status_history != expected_history:
        raise RuntimeError("unexpected status history: %s" % status_history)
    if result.status != TaskStatus.COMPLETED:
        raise RuntimeError("training did not complete")
    if manifest["final_status"] != "completed" or not manifest_metadata_valid:
        raise RuntimeError("manifest validation failed")
    if child_alive or worker.is_busy or worker.active_child_pid is not None:
        raise RuntimeError("training process or Worker state was not cleaned")


if __name__ == "__main__":
    main()
