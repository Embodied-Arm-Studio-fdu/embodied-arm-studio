"""Manual Phase 5 closure smoke with test PostgreSQL and real loopback HTTP."""

import os
import socket
import sys
import threading
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import uvicorn
from sqlalchemy import delete

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs.db_models import (
    RunArtifactRow,
    RunEventRow,
    RunIdempotencyKeyRow,
    RunRow,
)
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
from simulation_core.http_api import create_http_app  # noqa: E402
from simulation_core.service import SimulationService  # noqa: E402


def main():
    database_url = safe_test_database_url()
    if database_url is None:
        raise RuntimeError(
            "Phase 5.5 smoke requires TEST_DATABASE_URL and POSTGRES_TEST_OPT_IN=1"
        )
    prefix = "p55-smoke-%s" % uuid4().hex
    task_id = "task-" + prefix
    simulation_token = "phase-5-5-simulation-smoke-only"
    internal_token = "phase-5-5-internal-smoke-only"
    simulation_service = SimulationService(
        InMemoryTaskRepository(),
        FakeScheduler(),
        FakeBackendClient(),
        task_id_factory=lambda: task_id,
    )
    adapter = create_http_app(simulation_service, simulation_token)
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
    cleanup_engine = create_postgres_engine(database_url)
    cleanup_factory = create_session_factory(cleanup_engine)
    run_id = None
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
                internal_run_event_service_token=internal_token,
                simulation_environment={
                    "SIMULATION_BASE_URL": "http://127.0.0.1:%d" % listener.getsockname()[1],
                    "SIMULATION_SERVICE_TOKEN": simulation_token,
                    "SIMULATION_TIMEOUT_SECONDS": "2",
                    "SIMULATION_RECOVERY_MAX_RUNS": "20",
                },
            )
            with ASGITestClient(platform) as client:
                created = client.post(
                    "/api/v1/runs",
                    headers={"Idempotency-Key": prefix + "-create"},
                    json={
                        "project_id": prefix + "-project",
                        "execution_mode": "simulation",
                        "run_kind": "train",
                        "parameters": {"seed": 7},
                        "project_config_snapshot": {"environment_type": "s1_push"},
                        "code": "print('phase 5.5 smoke')\n",
                    },
                )
                assert created.status_code == 202, created.text
                run_id = created.json()["run_id"]
                assert created.json()["simulation_task_id"] == task_id
                auth = {"Authorization": "Bearer " + internal_token}
                occurred = datetime.now(timezone.utc)
                for sequence, previous, status in (
                    (1, "queued", "initializing"),
                    (2, "initializing", "running"),
                ):
                    event = client.post(
                        "/api/internal/v1/runs/%s/events" % run_id,
                        headers=auth,
                        json={
                            "schema_version": "1.0",
                            "event_id": prefix + "-state-%d" % sequence,
                            "sequence": sequence,
                            "type": "state",
                            "run_id": run_id,
                            "task_id": task_id,
                            "occurred_at": (occurred + timedelta(seconds=sequence)).isoformat(),
                            "payload": {"previous_status": previous, "status": status},
                        },
                    )
                    assert 200 <= event.status_code < 300, event.text
                metric = client.post(
                    "/api/internal/v1/runs/%s/events" % run_id,
                    headers=auth,
                    json={
                        "schema_version": "1.0",
                        "event_id": prefix + "-metric",
                        "sequence": 3,
                        "type": "metric",
                        "run_id": run_id,
                        "task_id": task_id,
                        "occurred_at": (occurred + timedelta(seconds=3)).isoformat(),
                        "payload": {"step": 16, "metrics": {"average_reward": 0.5}},
                    },
                )
                assert 200 <= metric.status_code < 300, metric.text
                stopping = client.post("/api/v1/runs/%s/stop" % run_id)
                assert stopping.status_code == 200, stopping.text
                assert stopping.json()["status"] == "stopping"
                callback = client.post(
                    "/api/internal/v1/runs/%s/callback" % run_id,
                    headers=auth,
                    json={
                        "schema_version": "1.0",
                        "event_id": prefix + "-callback",
                        "sequence": 4,
                        "run_id": run_id,
                        "task_id": task_id,
                        "status": "stopped",
                        "started_at": occurred.isoformat(),
                        "ended_at": (occurred + timedelta(seconds=4)).isoformat(),
                        "training_steps": 16,
                        "update_steps": 1,
                        "stop_reason": "requested_by_user",
                        "metrics": {"average_reward": 0.5},
                        "model": None,
                        "artifacts": [
                            {"type": "metrics", "path": "metrics/training.jsonl"}
                        ],
                        "error": None,
                    },
                )
                assert 200 <= callback.status_code < 300, callback.text
                detail = client.get("/api/v1/runs/%s" % run_id)
                artifacts = client.get("/api/v1/runs/%s/artifacts" % run_id)
                events = client.get("/api/v1/runs/%s/events" % run_id)
                assert detail.json()["status"] == "stopped"
                assert detail.json()["metrics_summary"]["average_reward"] == 0.5
                assert [item["path"] for item in artifacts.json()] == [
                    "metrics/training.jsonl"
                ]
                assert [item["sequence"] for item in events.json()] == [1, 2, 3, 4]
        print("PHASE_5_5_INTEGRATION_SMOKE=success")
    finally:
        if run_id is not None:
            with cleanup_factory() as session:
                session.execute(delete(RunArtifactRow).where(RunArtifactRow.run_id == run_id))
                session.execute(delete(RunEventRow).where(RunEventRow.run_id == run_id))
                session.execute(
                    delete(RunIdempotencyKeyRow).where(
                        RunIdempotencyKeyRow.run_id == run_id
                    )
                )
                session.execute(delete(RunRow).where(RunRow.run_id == run_id))
                session.commit()
        cleanup_engine.dispose()
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()


if __name__ == "__main__":
    main()
