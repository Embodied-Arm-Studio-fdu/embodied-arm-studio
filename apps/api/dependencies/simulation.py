"""FastAPI composition for the outbound Simulation HTTP boundary."""

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from robotics.runs import RunService
from robotics.simulation import (
    SimulationClient,
    SimulationConfigurationError,
    SimulationDispatchService,
    SimulationRecoveryService,
    SimulationStopService,
    StreamSessionService,
)


@dataclass(frozen=True)
class SimulationAssembly:
    client: Optional[Any]
    dispatch_service: Optional[SimulationDispatchService]
    stop_service: SimulationStopService
    recovery_service: Optional[SimulationRecoveryService]
    stream_service: StreamSessionService


def build_simulation_assembly(
    run_service: RunService,
    *,
    client: Optional[Any] = None,
    environment: Optional[Mapping[str, str]] = None,
    dispatch_authorizer: Optional[Any] = None,
) -> SimulationAssembly:
    source = os.environ if environment is None else environment
    raw_max_runs = source.get("SIMULATION_RECOVERY_MAX_RUNS", "100")
    try:
        max_runs = int(raw_max_runs)
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(
            "SIMULATION_RECOVERY_MAX_RUNS must be a positive integer"
        ) from exc
    if max_runs < 1:
        raise SimulationConfigurationError(
            "SIMULATION_RECOVERY_MAX_RUNS must be a positive integer"
        )
    if client is not None:
        if not callable(getattr(client, "create_task", None)):
            raise SimulationConfigurationError("simulation client must implement create_task")
        dispatch = SimulationDispatchService(
            run_service, client, dispatch_authorizer
        )
        stop = SimulationStopService(run_service, client)
        return SimulationAssembly(
            client=client,
            dispatch_service=dispatch,
            stop_service=stop,
            recovery_service=SimulationRecoveryService(
                run_service, dispatch, stop, max_runs=max_runs
            ),
            stream_service=StreamSessionService(run_service, client),
        )

    base_url = source.get("SIMULATION_BASE_URL")
    token = source.get("SIMULATION_SERVICE_TOKEN")
    if not base_url or not token:
        return SimulationAssembly(
            client=None,
            dispatch_service=None,
            stop_service=SimulationStopService(run_service, None),
            recovery_service=None,
            stream_service=StreamSessionService(run_service, None),
        )
    raw_timeout = source.get("SIMULATION_TIMEOUT_SECONDS", "5")
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(
            "SIMULATION_TIMEOUT_SECONDS must be a positive number"
        ) from exc
    resolved = SimulationClient(base_url, timeout=timeout, service_token=token)
    dispatch = SimulationDispatchService(
        run_service, resolved, dispatch_authorizer
    )
    stop = SimulationStopService(run_service, resolved)
    return SimulationAssembly(
        client=resolved,
        dispatch_service=dispatch,
        stop_service=stop,
        recovery_service=SimulationRecoveryService(
            run_service, dispatch, stop, max_runs=max_runs
        ),
        stream_service=StreamSessionService(run_service, resolved),
    )
