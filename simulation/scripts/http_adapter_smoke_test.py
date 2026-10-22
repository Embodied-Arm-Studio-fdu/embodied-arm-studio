#!/usr/bin/env python3
"""Real-loopback smoke using the backend's existing SimulationClient."""

import os
import socket
import threading
import time

import uvicorn

from robotics.simulation import SimulationClient
from simulation_core import (
    FakeBackendClient,
    FakeScheduler,
    InMemoryTaskRepository,
    SimulationService,
    create_http_app,
)


def main():
    # The backend client intentionally honors deployment proxy settings; this
    # loopback-only smoke must bypass any workstation HTTP proxy.
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    token = "smoke-only-service-token"
    service = SimulationService(
        InMemoryTaskRepository(),
        FakeScheduler(),
        FakeBackendClient(),
        task_id_factory=lambda: "task-http-smoke",
    )
    app = create_http_app(service, token)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 5
        while not server.started and time.time() < deadline:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("HTTP Adapter did not start")
        with SimulationClient(
            "http://127.0.0.1:%d" % port, timeout=2, service_token=token
        ) as client:
            first = client.create_task("run-http-smoke", "project-http-smoke", {"seed": 7})
            duplicate = client.create_task("run-http-smoke", "project-http-smoke", {"seed": 7})
            fetched = client.get_task(first.task_id)
            stopped = client.stop_task(first.task_id)
        assert first.task_id == duplicate.task_id == fetched.task_id == stopped.task_id
        assert stopped.status.value == "stopped"
        print("HTTP_ADAPTER_SMOKE_RESULT=success")
        print("backend_caller=robotics.simulation.SimulationClient")
        print("task_id=%s" % first.task_id)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()


if __name__ == "__main__":
    main()
