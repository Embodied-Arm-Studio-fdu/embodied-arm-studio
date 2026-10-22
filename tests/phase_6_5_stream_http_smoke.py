#!/usr/bin/env python3
"""Phase 6.5 Platform-to-Simulation stream control-plane smoke."""

import os
import socket
import sys
import threading
import time
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import uvicorn
from sqlalchemy import delete, select

# Importing apps.api.main creates its default app. This unused module-level app
# is explicitly memory-backed for this test; the smoke app below is PostgreSQL.
os.environ.setdefault("RUN_REPOSITORY_MODE", "memory")

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs import ExecutionMode, RunKind, RunStatus
from robotics.runs.db_models import RunRow
from tests.http_client import ASGITestClient
from tests.postgres_safety import safe_test_database_url


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
from simulation_core.domain import TaskStatus  # noqa: E402
from simulation_core.http_api import create_http_app  # noqa: E402
from simulation_core.service import SimulationService  # noqa: E402
from simulation_core.stream_session import (  # noqa: E402
    RecordingStreamGateway,
    StreamSessionManager,
)


class NoopStreamRuntime:
    """Control-plane-only runtime: no Camera, encoder, or MediaMTX."""

    def start(self):
        return None

    def stop(self):
        return None


class PublishedRecordingStreamGateway(RecordingStreamGateway):
    """Isolated simulation test gateway with the production read boundary."""

    def get_for_run(self, run_id):
        for record in reversed(self.records):
            if record["run_id"] == run_id:
                return dict(record)
        return None


def main():
    database_url = safe_test_database_url()
    if database_url is None:
        raise RuntimeError(
            "Phase 6.5 smoke requires TEST_DATABASE_URL and POSTGRES_TEST_OPT_IN=1"
        )
    prefix = "p65-smoke-%s" % uuid4().hex
    task_id = prefix + "-task"
    token = "phase-6-5-stream-smoke-only"

    repository = InMemoryTaskRepository()
    simulation_service = SimulationService(
        repository,
        FakeScheduler(),
        FakeBackendClient(),
        task_id_factory=lambda: task_id,
    )
    manager = StreamSessionManager(
        lambda session: NoopStreamRuntime(),
        webrtc_base="http://127.0.0.1:18889",
    )
    gateway = PublishedRecordingStreamGateway()

    adapter = create_http_app(simulation_service, token, stream_gateway=gateway)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    server = uvicorn.Server(uvicorn.Config(adapter, log_level="error", lifespan="off"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()

    cleanup_engine = create_postgres_engine(database_url)
    cleanup_factory = create_session_factory(cleanup_engine)
    platform = None
    run_id = None
    smoke_succeeded = False
    try:
        deadline = time.time() + 5
        while not server.started and time.time() < deadline:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("Simulation HTTP Adapter did not start")

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            platform = create_app(
                root / "runs",
                tick_seconds=0,
                project_root=root / "projects",
                run_repository_mode="postgresql",
                database_url=database_url,
                current_user_provider=StaticCurrentUserProvider(prefix + "-user"),
                simulation_environment={
                    "SIMULATION_BASE_URL": "http://127.0.0.1:%d"
                    % listener.getsockname()[1],
                    "SIMULATION_SERVICE_TOKEN": token,
                    "SIMULATION_TIMEOUT_SECONDS": "2",
                    "SIMULATION_RECOVERY_MAX_RUNS": "20",
                },
            )
            service = platform.state.run_service
            run = service.create_run(
                user_id=prefix + "-user",
                project_id=prefix + "-project",
                execution_mode=ExecutionMode.SIMULATION,
                run_kind=RunKind.TRAIN,
                parameters={},
                project_config_snapshot={},
                code="",
            )
            run_id = run.run_id
            service.bind_simulation_task(run.run_id, task_id)
            service.transition_status(run.run_id, RunStatus.INITIALIZING)
            service.transition_status(run.run_id, RunStatus.RUNNING)

            task = simulation_service.create_task(
                run_id, prefix + "-project", {}
            )
            simulation_service.transition_task_status(
                task.task_id, TaskStatus.INITIALIZING
            )
            simulation_service.transition_task_status(
                task.task_id, TaskStatus.RUNNING
            )
            session = manager.create_session(
                run_id, task_id, "overview", 640, 480, 15.0
            )
            ready = manager.start_session(session.stream_id)
            gateway.publish(ready)

            with ASGITestClient(platform) as client:
                url = "/api/v1/runs/%s/stream?camera_id=overview" % run_id
                first = client.get(url)
                second = client.get(url)
                assert first.status_code == 200, first.text
                assert second.status_code == 200, second.text
                assert first.json() == second.json()
                assert first.json()["stream_id"] == ready.stream_id
                assert first.json()["status"] == "READY"
                assert simulation_service.get_task(task_id).status == TaskStatus.RUNNING
                assert service.get_run(run_id).status == RunStatus.RUNNING

        smoke_succeeded = True
        print("PHASE_6_5_STREAM_HTTP_SMOKE=success")
        print("stream_status=READY")
        print("stream_id_reused=true")
        print("postgresql=test")
        print("stream_gateway=isolated_recording")
        print("mediamtx_camera_encoder_started=false")
    finally:
        if platform is not None:
            client = getattr(platform.state, "simulation_client", None)
            if client is not None:
                client.close()
        if run_id is not None:
            with cleanup_factory() as cleanup:
                deleted = cleanup.execute(
                    delete(RunRow).where(RunRow.run_id == run_id)
                )
                cleanup.commit()
                if smoke_succeeded:
                    assert deleted.rowcount == 1
                    assert cleanup.scalar(
                        select(RunRow.run_id).where(RunRow.run_id == run_id)
                    ) is None
        cleanup_engine.dispose()
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        if smoke_succeeded:
            print("test_postgresql_cleanup=complete")


if __name__ == "__main__":
    main()
