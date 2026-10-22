"""Canonical simulation Run stop coordination."""

from typing import Any, Optional

from robotics.runs import (
    ExecutionMode,
    Run,
    RunService,
    RunStatus,
    RunStopNotAllowed,
    RunTaskMismatch,
)

from .dispatch import SimulationSnapshotMismatch


class SimulationStopUnavailable(RuntimeError):
    pass


class SimulationStopService:
    def __init__(self, run_service: RunService, client: Optional[Any]) -> None:
        self._run_service = run_service
        self._client = client

    def stop(self, run_id: str) -> Run:
        run = self._run_service.get_run(run_id)
        if run.execution_mode != ExecutionMode.SIMULATION:
            raise RunTaskMismatch("Only simulation Runs use Simulation stop")
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.STOPPED}:
            return run
        if run.status in {RunStatus.INITIALIZING, RunStatus.FINALIZING}:
            raise RunStopNotAllowed(run.run_id, run.status)
        if run.status == RunStatus.QUEUED and run.simulation_task_id is None:
            return self._run_service.request_stop(run.run_id)

        stopping = self._run_service.request_stop(run.run_id)
        if stopping.simulation_task_id is None:
            raise RunTaskMismatch("stopping simulation Run has no bound task_id")
        if self._client is None:
            raise SimulationStopUnavailable("Simulation stop is not configured")
        task = self._client.stop_task(stopping.simulation_task_id)
        if (
            task.task_id != stopping.simulation_task_id
            or task.run_id != stopping.run_id
            or task.project_id != stopping.project_id
        ):
            raise SimulationSnapshotMismatch(
                "Simulation stop snapshot does not match the persisted Run"
            )
        # The HTTP response is diagnostic acknowledgement only. Terminal Callback
        # remains the sole authority that converges stopping to stopped.
        return self._run_service.get_run(stopping.run_id)
