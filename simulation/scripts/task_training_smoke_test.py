#!/usr/bin/env python3
"""Explicit real-GPU smoke test for task-driven S1 Push PPO training."""

import argparse
from pathlib import Path

from simulation_core import (
    FakeBackendClient,
    FakeScheduler,
    InMemoryTaskRepository,
    SimulationService,
    TaskStatus,
    TrainingTaskConfig,
    TrainingTaskRunner,
    TrainingTaskSpec,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default="phase-4-1")
    parser.add_argument("--run-id", default="task-training-smoke")
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
    scheduler = FakeScheduler()
    backend = FakeBackendClient()
    service = SimulationService(repository, scheduler, backend)
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
    runner = TrainingTaskRunner(
        args.artifacts_root,
        progress_callback=lambda status: service.transition_task_status(
            task.task_id, status
        ),
    )
    try:
        service.transition_task_status(task.task_id, TaskStatus.INITIALIZING)
        result = runner.run(
            TrainingTaskSpec.from_task(task),
        )
        service.transition_task_status(task.task_id, TaskStatus.COMPLETED)
    finally:
        print("STATUS_HISTORY=%s" % " -> ".join(item.status.value for item in backend.reported_tasks))
    print("TASK_TRAINING_RESULT=success")
    print("task_id=%s run_id=%s status=%s" % (result.task_id, result.run_id, result.status.value))
    print("training_steps=%d update_steps=%d" % (result.training_steps, result.update_steps))
    print("metrics=%s" % result.metrics_path)
    print("checkpoint=%s" % result.checkpoint_path)
    print("final_metrics=%s" % dict(result.final_metrics_summary))


if __name__ == "__main__":
    main()
