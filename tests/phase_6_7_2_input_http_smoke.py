"""Lightweight real-handler smoke for Browser WS -> Simulation input HTTP."""

import asyncio
import os
from pathlib import Path
import tempfile

import httpx

os.environ.setdefault("RUN_REPOSITORY_MODE", "memory")

from apps.api.main import create_app
from robotics.runs import ExecutionMode, InMemoryRunRepository, RunKind, RunStatus
from robotics.simulation import SimulationClient
from simulation_core import (
    FakeBackendClient,
    FakeScheduler,
    InMemoryTaskRepository,
    RecordingInputEventSink,
    SimulationService,
    TaskStatus,
    create_http_app,
)
from tests.auth import StaticCurrentUserProvider
from tests.test_v1_run_websocket import ASGIWebSocketDriver


class ASGIForwardTransport(httpx.BaseTransport):
    def __init__(self, app):
        self.app = app
        self.closed = False

    def handle_request(self, request):
        async def forward():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=self.app, raise_app_exceptions=False
                ),
                base_url="http://simulation.test",
            ) as client:
                response = await client.request(
                    request.method,
                    request.url.path,
                    headers=dict(request.headers),
                    content=request.content,
                )
                return response.status_code, response.headers, await response.aread()

        status, headers, content = asyncio.run(forward())
        return httpx.Response(
            status, headers=headers, content=content, request=request
        )

    def close(self):
        self.closed = True


async def smoke():
    token = "phase-6-7-2-smoke-service-token"
    simulation_repository = InMemoryTaskRepository()
    scheduler = FakeScheduler()
    simulation_service = SimulationService(
        simulation_repository,
        scheduler,
        FakeBackendClient(),
        task_id_factory=lambda: "task-input-http-smoke",
    )
    sink = RecordingInputEventSink()
    simulation_app = create_http_app(
        simulation_service, token, input_sink=sink
    )
    transport = ASGIForwardTransport(simulation_app)
    simulation_client = SimulationClient(
        "http://simulation.test",
        service_token=token,
        transport=transport,
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        platform = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider("smoke-user"),
            simulation_client=simulation_client,
        )
        run_service = platform.state.run_service
        run = run_service.create_run(
            user_id="smoke-user",
            project_id="project-smoke",
            execution_mode=ExecutionMode.SIMULATION,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        task = simulation_service.create_task(run.run_id, "project-smoke", {})
        simulation_service.transition_task_status(task.task_id, TaskStatus.INITIALIZING)
        simulation_service.transition_task_status(task.task_id, TaskStatus.RUNNING)
        run_service.bind_simulation_task(run.run_id, task.task_id)
        run_service.transition_status(run.run_id, RunStatus.INITIALIZING)
        run = run_service.transition_status(run.run_id, RunStatus.RUNNING)

        socket = ASGIWebSocketDriver(
            platform, "/api/v1/runs/%s/input" % run.run_id
        )
        assert (await socket.connect())["type"] == "websocket.accept"
        payloads = [
            {
                "event_id": "smoke-keyboard",
                "run_id": run.run_id,
                "task_id": task.task_id,
                "type": "KEYBOARD",
                "occurred_at": "2026-08-28T00:00:00+00:00",
                "key": "W",
                "action": "pressed",
            },
            {
                "event_id": "smoke-move",
                "run_id": run.run_id,
                "task_id": task.task_id,
                "type": "MOUSE_MOVE",
                "occurred_at": "2026-08-28T00:00:01+00:00",
                "x": 0.4,
                "y": 0.6,
            },
            {
                "event_id": "smoke-button",
                "run_id": run.run_id,
                "task_id": task.task_id,
                "type": "MOUSE_BUTTON",
                "occurred_at": "2026-08-28T00:00:02+00:00",
                "button": "left",
                "action": "released",
                "x": 0.4,
                "y": 0.6,
            },
        ]
        for payload in payloads:
            await socket.send_json(payload)
            acknowledgement = await socket.receive_json(timeout=2)
            assert acknowledgement["type"] == "ACK"
            assert acknowledgement["duplicate"] is False
        await socket.send_json(payloads[0])
        assert (await socket.receive_json(timeout=2))["duplicate"] is True
        assert len(sink.events_for_task(task.task_id)) == 3
        assert run_service.get_run(run.run_id).status == RunStatus.RUNNING
        assert scheduler.cancelled_task_ids == []
        await socket.disconnect()
        await platform.router.shutdown()
        assert transport.closed is True


if __name__ == "__main__":
    asyncio.run(smoke())
    print("phase 6.7.2 input HTTP smoke: PASS")
