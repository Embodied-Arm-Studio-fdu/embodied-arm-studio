"""Manual real-loopback smoke: Platform FastAPI -> Simulation HTTP Adapter."""

import socket
import sys
import threading
import time
import types
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn

from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs import InMemoryRunRepository
# Load only the HTTP/domain package modules. simulation_core.__init__ also exports
# GPU camera helpers whose optional runtime dependencies are intentionally absent
# from the backend venv used by this no-GPU smoke.
simulation_package = types.ModuleType("simulation_core")
simulation_package.__path__ = [
    str(Path(__file__).resolve().parents[1] / "simulation" / "src" / "simulation_core")
]
sys.modules["simulation_core"] = simulation_package
from simulation_core.adapters import (  # noqa: E402
    FakeBackendClient,
    FakeScheduler,
    InMemoryTaskRepository,
)
from simulation_core.http_api import create_http_app  # noqa: E402
from simulation_core.service import SimulationService  # noqa: E402
from tests.http_client import ASGITestClient
from tests.test_simulation_dispatch_api import MemoryCreationService


def main():
    token = "phase-5-4-smoke-only"
    service = SimulationService(
        InMemoryTaskRepository(),
        FakeScheduler(),
        FakeBackendClient(),
        task_id_factory=lambda: "task-platform-http-smoke",
    )
    adapter = create_http_app(service, token)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    server = uvicorn.Server(
        uvicorn.Config(adapter, log_level="error", lifespan="off")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.time() + 5
        while not server.started and time.time() < deadline:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("Simulation HTTP Adapter did not start")
        repository = InMemoryRunRepository()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            platform = create_app(
                root / "runs",
                tick_seconds=0,
                project_root=root / "projects",
                run_repository=repository,
                run_creation_service=MemoryCreationService(repository),
                current_user_provider=StaticCurrentUserProvider("smoke-user"),
                simulation_environment={
                    "SIMULATION_BASE_URL": "http://127.0.0.1:%d" % listener.getsockname()[1],
                    "SIMULATION_SERVICE_TOKEN": token,
                    "SIMULATION_TIMEOUT_SECONDS": "2",
                },
            )
            with ASGITestClient(platform) as client:
                response = client.post(
                    "/api/v1/runs",
                    headers={"Idempotency-Key": "phase-5-4-http-smoke"},
                    json={
                        "project_id": "project-platform-http-smoke",
                        "execution_mode": "simulation",
                        "run_kind": "train",
                        "parameters": {"seed": 7},
                        "project_config_snapshot": {"environment_type": "s1_push"},
                        "code": "print('http smoke')\n",
                    },
                )
            assert response.status_code == 202, response.text
            assert response.json()["simulation_task_id"] == "task-platform-http-smoke"
            assert len(repository.list_runs()) == 1
        print("PHASE_5_4_PLATFORM_HTTP_SMOKE=success")
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()


if __name__ == "__main__":
    main()
