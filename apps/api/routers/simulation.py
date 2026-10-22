"""Authenticated read-only proxy for Simulation Service capabilities."""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

from robotics.auth import User
from robotics.simulation import (
    SimulationConfigurationError,
    SimulationNetworkError,
    SimulationRejectedError,
    SimulationResponseError,
    SimulationServerError,
    SimulationTimeoutError,
)


def create_simulation_router(
    simulation_client: Optional[Any], current_user_provider: Any
) -> APIRouter:
    router = APIRouter(tags=["simulation"])

    @router.get("/api/simulation/task-catalog")
    def get_task_catalog(
        _user: User = Depends(current_user_provider),
    ) -> Any:
        get_catalog = getattr(simulation_client, "get_task_catalog", None)
        if not callable(get_catalog):
            raise HTTPException(
                status_code=503, detail="Simulation Service is unavailable"
            )
        try:
            return get_catalog().to_dict()
        except (
            SimulationTimeoutError,
            SimulationNetworkError,
            SimulationServerError,
        ) as exc:
            raise HTTPException(
                status_code=503, detail="Simulation Service is unavailable"
            ) from exc
        except (
            SimulationRejectedError,
            SimulationResponseError,
            SimulationConfigurationError,
        ) as exc:
            raise HTTPException(
                status_code=502, detail="Simulation Service returned an invalid response"
            ) from exc

    return router
