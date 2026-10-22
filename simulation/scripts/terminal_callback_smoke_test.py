#!/usr/bin/env python3
"""Real-GPU completed callback/outbox smoke test without HTTP."""

import argparse
import json
from pathlib import Path

from simulation_core import (
    DeliveryStatus, FakeScheduler, GPUWorker, InMemoryTaskRepository,
    RecordingBackendClient, SimulationService, TaskStatus, TrainingTaskConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default="phase-5-3")
    parser.add_argument("--run-id", default="terminal-callback-smoke")
    parser.add_argument(
        "--artifacts-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    args = parser.parse_args()
    backend = RecordingBackendClient()
    service = SimulationService(InMemoryTaskRepository(), FakeScheduler(), backend)
    task = service.create_task(args.run_id, args.project_id, TrainingTaskConfig(
        task_type="training", environment_type="s1_push", seed=7,
        rollout_steps=16, num_updates=2, checkpoint_every=1,
        episode_steps=32, physics_steps_per_action=2,
        update_epochs=1, batch_size=16,
    ).to_dict())
    worker = GPUWorker(service, args.artifacts_root)
    result = worker.run(task.task_id)
    events = backend.events_for_run(task.run_id)
    callbacks = backend.terminal_callbacks_for_run(task.run_id)
    record = service.callback_outbox.get_by_task_id(task.task_id)
    assert result.status == TaskStatus.COMPLETED
    assert [event.type.value for event in events] == [
        "state", "state", "metric", "metric", "state"
    ]
    assert len(callbacks) == 1
    callback = callbacks[0]
    assert [event.sequence for event in events] + [callback.sequence] == list(range(1, 7))
    assert callback.event_id and callback.event_id not in {event.event_id for event in events}
    assert callback.model == {
        "type": "checkpoint_best", "path": "checkpoints/model_best.pt"
    }
    assert "manifest.json" in [item["path"] for item in callback.artifacts]
    assert callback.metrics == result.final_metrics_summary
    assert record.delivery_status == DeliveryStatus.DELIVERED
    assert Path(result.best_checkpoint_path).is_file()
    assert Path(result.manifest_path).is_file()
    assert not worker.is_busy and worker.last_exit_code == 0
    print(json.dumps({
        "events": [event.to_dict() for event in events],
        "callback": callback.to_dict(), "outbox": record.to_dict(),
        "worker_idle": not worker.is_busy, "child_exit_code": worker.last_exit_code,
    }, indent=2))
    print("TERMINAL_CALLBACK_SMOKE_RESULT=success")


if __name__ == "__main__":
    main()
