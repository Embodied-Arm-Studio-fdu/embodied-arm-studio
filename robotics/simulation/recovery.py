"""Bounded one-shot reconciliation for recoverable simulation Runs."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from robotics.runs import ExecutionMode, RunService, RunStatus

from .dispatch import SimulationDispatchService
from .stop import SimulationStopService


@dataclass(frozen=True)
class SimulationRecoveryResult:
    examined: int
    dispatched: int
    stop_retried: int
    failed: int


class SimulationRecoveryService:
    def __init__(
        self,
        run_service: RunService,
        dispatch_service: SimulationDispatchService,
        stop_service: SimulationStopService,
        *,
        max_runs: int = 100,
    ) -> None:
        if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 1:
            raise ValueError("max_runs must be a positive integer")
        self._run_service = run_service
        self._dispatch_service = dispatch_service
        self._stop_service = stop_service
        self._max_runs = max_runs

    def reconcile_simulation_runs(self) -> SimulationRecoveryResult:
        examined = dispatched = stop_retried = failed = 0
        # Apply the cap after selecting recoverable Runs. Terminal history must
        # not hide old unbound records from startup reconciliation.
        candidates = [run for run in reversed(self._run_service.list_runs())
                      if run.execution_mode == ExecutionMode.SIMULATION and (
                          (run.status == RunStatus.QUEUED and run.simulation_task_id is None)
                          or (run.status == RunStatus.STOPPING and run.simulation_task_id is not None))]
        for run in candidates[: self._max_runs]:
            examined += 1
            if run.execution_mode != ExecutionMode.SIMULATION:
                continue
            try:
                if run.status == RunStatus.QUEUED and run.simulation_task_id is None:
                    if datetime.now(timezone.utc) - run.created_at >= timedelta(minutes=15):
                        # Historical intents need diagnosis, not a surprise GPU run.
                        self._dispatch_service.inspect_stale_dispatch(run.run_id)
                    else:
                        self._dispatch_service.dispatch(run.run_id)
                        dispatched += 1
                elif run.status == RunStatus.STOPPING and run.simulation_task_id is not None:
                    self._stop_service.stop(run.run_id)
                    stop_retried += 1
            except Exception:
                # One failed upstream call or binding must not block other Runs or startup.
                failed += 1
        return SimulationRecoveryResult(
            examined=examined,
            dispatched=dispatched,
            stop_retried=stop_retried,
            failed=failed,
        )
