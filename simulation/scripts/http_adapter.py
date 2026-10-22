#!/usr/bin/env python3
"""Run the Simulation HTTP Adapter using environment configuration."""

import os

import uvicorn

from simulation_core.http_api import create_app_from_env


if __name__ == "__main__":
    uvicorn.run(
        create_app_from_env(),
        host=os.environ.get("SIMULATION_HTTP_HOST", "127.0.0.1"),
        port=int(os.environ.get("SIMULATION_HTTP_PORT", "8090")),
    )
