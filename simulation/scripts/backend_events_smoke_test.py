#!/usr/bin/env python3
"""Real-GPU Phase 5.2 smoke test using the in-memory recording backend."""

import argparse
import json
from pathlib import Path

from simulation_core import (
    FakeScheduler, GPUWorker, InMemoryTaskRepository, RecordingBackendClient,
    SimulationService, TaskStatus, TrainingTaskConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default="phase-5-2")
    parser.add_argument("--run-id", default="backend-events-smoke")
    parser.add_argument("--artifacts-root", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    args = parser.parse_args()
    backend = RecordingBackendClient()
    service = SimulationService(InMemoryTaskRepository(), FakeScheduler(), backend)
    task = service.create_task(args.run_id, args.project_id, TrainingTaskConfig(
        task_type="training", environment_type="s1_push", seed=7,
        rollout_steps=16, num_updates=2, checkpoint_every=1,
        episode_steps=32, physics_steps_per_action=2, update_epochs=1,
        batch_size=16,
    ).to_dict())
    worker = GPUWorker(service, args.artifacts_root)
    result = worker.run(task.task_id)
    events = backend.events_for_run(task.run_id)
    types = [event.type.value for event in events]
    states = [event.payload.get("status") for event in events if event.type.value == "state"]
    sequences = [event.sequence for event in events]
    assert result.status == TaskStatus.COMPLETED
    assert states == ["initializing", "running", "finalizing"]
    assert types.count("metric") == 2
    assert sequences == sorted(sequences) and len(sequences) == len(set(sequences))
    assert all(event.event_id for event in events)
    assert Path(result.metrics_path).is_file()
    assert Path(result.checkpoint_path).is_file()
    assert Path(result.manifest_path).is_file()
    assert not worker.is_busy and worker.last_exit_code == 0
    print(json.dumps([event.to_dict() for event in events], indent=2))
    print("BACKEND_EVENTS_SMOKE_RESULT=success")


if __name__ == "__main__":
    main()
