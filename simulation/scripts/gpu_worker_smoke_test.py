#!/usr/bin/env python3
"""Explicit real-GPU smoke test for the isolated single-task GPU Worker."""

import argparse
import os
from pathlib import Path

from simulation_core import (
    FakeBackendClient,
    FakeScheduler,
    GPUWorker,
    InMemoryTaskRepository,
    SimulationService,
    TrainingTaskConfig,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default="phase-4-2")
    parser.add_argument("--run-id", default="gpu-worker-smoke")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--num-updates", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
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
    task = service.create_task(args.run_id, args.project_id, config.to_dict())
    worker = GPUWorker(service, args.artifacts_root)

    print("parent_pid=%d" % os.getpid(), flush=True)
    try:
        result = worker.run(task.task_id)
    finally:
        print(
            "status_history=%s"
            % " -> ".join(item.status.value for item in backend.reported_tasks),
            flush=True,
        )
        print("child_pid=%s" % worker.last_child_pid, flush=True)
        print("child_exit_code=%s" % worker.last_exit_code, flush=True)
        print(
            "final_task_status=%s" % service.get_task(task.task_id).status.value,
            flush=True,
        )
        print("worker_idle=%s" % (not worker.is_busy), flush=True)

    print("GPU_WORKER_RESULT=success")
    print("training_steps=%d" % result.training_steps)
    print("update_steps=%d" % result.update_steps)
    print("metrics=%s" % result.metrics_path)
    print("checkpoint=%s" % result.checkpoint_path)
    print("final_metrics=%s" % dict(result.final_metrics_summary))


if __name__ == "__main__":
    main()
