import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

from sqlalchemy import delete, select

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from tests.postgres_users import delete_test_user, seed_test_user
from tests.postgres_safety import safe_test_database_url
from robotics.runs import (
    ExecutionMode,
    InMemoryRunRepository,
    RepositoryError,
    RunEvent,
    RunEventType,
    RunKind,
    RunStatus,
    TerminalCallbackCommand,
)
from robotics.runs.db_models import RunEventRow, RunIdempotencyKeyRow, RunRow


EVENT_FIELDS = {
    "event_id",
    "sequence",
    "type",
    "run_id",
    "task_id",
    "occurred_at",
    "schema_version",
    "payload",
}


class ASGIWebSocketDriver:
    def __init__(self, app, url, headers=None):
        parsed = urlsplit(url)
        self._incoming = asyncio.Queue()
        self._outgoing = asyncio.Queue()
        self._app = app
        if headers is None:
            provider = getattr(app.state, "current_user_provider", None)
            authorization = getattr(
                provider, "test_authorization_header", None
            )
            headers = (
                [(b"authorization", authorization.encode("ascii"))]
                if authorization
                else []
            )
        self._scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "scheme": "ws",
            "path": parsed.path,
            "raw_path": parsed.path.encode("ascii"),
            "query_string": parsed.query.encode("ascii"),
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "subprotocols": [],
            "state": {},
        }
        self._task = None

    async def connect(self):
        self._task = asyncio.create_task(
            self._app(self._scope, self._incoming.get, self._outgoing.put)
        )
        await self._incoming.put({"type": "websocket.connect"})
        outgoing = asyncio.create_task(self._outgoing.get())
        done, _ = await asyncio.wait(
            {outgoing, self._task},
            timeout=1,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if outgoing in done:
            return outgoing.result()
        outgoing.cancel()
        if self._task in done:
            await self._task
            if not self._outgoing.empty():
                return self._outgoing.get_nowait()
            raise AssertionError("WebSocket app returned without a response")
        raise asyncio.TimeoutError()

    async def receive(self, timeout=1):
        outgoing = asyncio.create_task(self._outgoing.get())
        watched = {outgoing}
        if self._task is not None:
            watched.add(self._task)
        done, _ = await asyncio.wait(
            watched, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if outgoing in done:
            return outgoing.result()
        outgoing.cancel()
        if self._task is not None and self._task in done:
            await self._task
            if not self._outgoing.empty():
                return self._outgoing.get_nowait()
            raise AssertionError("WebSocket app returned without a response")
        raise asyncio.TimeoutError()

    async def receive_json(self, timeout=1):
        message = await self.receive(timeout)
        if message["type"] != "websocket.send":
            raise AssertionError("expected websocket.send, got %r" % message)
        return json.loads(message["text"])

    async def send_text(self, value):
        await self._incoming.put({"type": "websocket.receive", "text": value})

    async def send_json(self, value):
        await self.send_text(json.dumps(value, separators=(",", ":")))

    async def send_bytes(self, value):
        await self._incoming.put({"type": "websocket.receive", "bytes": value})

    async def assert_no_message(self, timeout=0.05):
        with self.assert_timeout():
            await asyncio.wait_for(self._outgoing.get(), timeout)

    async def disconnect(self):
        if self._task is None:
            return
        if not self._task.done():
            await self._incoming.put(
                {"type": "websocket.disconnect", "code": 1001}
            )
        await asyncio.wait_for(self._task, 1)

    async def finish(self):
        if self._task is not None:
            await asyncio.wait_for(self._task, 1)

    @staticmethod
    def assert_timeout():
        return _ExpectedTimeout()


class _ExpectedTimeout:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is asyncio.TimeoutError:
            return True
        if exc_type is None:
            raise AssertionError("an unexpected WebSocket message was received")
        return False


class FailingEventRepository(InMemoryRunRepository):
    def __init__(self):
        super().__init__()
        self.poll_count = 0

    def list_events(self, run_id, after_sequence=None, limit=None):
        self.poll_count += 1
        raise RepositoryError("database password must not reach logs")


class V1RunWebSocketTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repository = InMemoryRunRepository()
        root = Path(self.temp_dir.name)
        self.app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=self.repository,
            current_user_provider=StaticCurrentUserProvider("trusted-user"),
            run_event_poll_interval_seconds=0.01,
            run_event_batch_size=2,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_run(self, user_id="trusted-user"):
        return self.app.state.run_service.create_run(
            user_id=user_id,
            project_id="project-1",
            execution_mode=ExecutionMode.MOCK,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )

    def append_event(self, run_id, sequence, event_type, payload, suffix=None):
        event = RunEvent(
            event_id="event-%s-%s" % (sequence, suffix or uuid4().hex),
            sequence=sequence,
            type=RunEventType(event_type),
            run_id=run_id,
            occurred_at=datetime.now(timezone.utc),
            payload=payload,
        )
        self.repository.append_event(event)
        return event

    async def test_full_history_after_sequence_gap_and_canonical_schema(self):
        run = self.create_run()
        self.append_event(
            run.run_id,
            1,
            "state",
            {"previous_status": "queued", "status": "initializing"},
        )
        self.append_event(
            run.run_id,
            3,
            "metric",
            {
                "step": 20,
                "metrics": {"update_index": 1, "average_reward": 0.4},
            },
        )
        self.append_event(
            run.run_id,
            6,
            "error",
            {"code": "TEST", "message": "safe", "retryable": False},
        )

        full = ASGIWebSocketDriver(
            self.app, "/api/v1/runs/%s/events" % run.run_id
        )
        self.assertEqual((await full.connect())["type"], "websocket.accept")
        first = await full.receive_json()
        third = await full.receive_json()
        sixth = await full.receive_json()
        self.assertEqual(
            [first["sequence"], third["sequence"], sixth["sequence"]],
            [1, 3, 6],
        )
        self.assertEqual(set(first), EVENT_FIELDS)
        self.assertEqual(first["schema_version"], "1.0")
        self.assertEqual(third["payload"]["metrics"]["update_index"], 1)
        await full.disconnect()

        resumed = ASGIWebSocketDriver(
            self.app,
            "/api/v1/runs/%s/events?after_sequence=1" % run.run_id,
        )
        self.assertEqual((await resumed.connect())["type"], "websocket.accept")
        self.assertEqual((await resumed.receive_json())["sequence"], 3)
        self.assertEqual((await resumed.receive_json())["sequence"], 6)
        await resumed.assert_no_message()
        await resumed.disconnect()

    async def test_active_connection_reads_new_event_without_duplicates(self):
        run = self.create_run()
        socket = ASGIWebSocketDriver(
            self.app, "/api/v1/runs/%s/events" % run.run_id
        )
        await socket.connect()
        self.append_event(
            run.run_id,
            5,
            "heartbeat",
            {"status": "running", "training_steps": 20},
        )
        self.assertEqual((await socket.receive_json())["sequence"], 5)
        await socket.assert_no_message(timeout=0.04)
        await socket.disconnect()

    async def test_browser_first_message_auth_precedes_event_history(self):
        run = self.create_run()
        self.append_event(
            run.run_id,
            2,
            "heartbeat",
            {"status": "running", "training_steps": 4},
        )
        socket = ASGIWebSocketDriver(
            self.app,
            "/api/v1/runs/%s/events" % run.run_id,
            headers=[],
        )
        self.assertEqual("websocket.accept", (await socket.connect())["type"])
        await socket.assert_no_message()
        await socket.send_json(
            {
                "type": "AUTH",
                "access_token": "phase-7-1-test-credential",
            }
        )
        self.assertEqual(
            {"type": "AUTH_ACK", "authenticated": True},
            await socket.receive_json(),
        )
        event = await socket.receive_json()
        self.assertEqual(2, event["sequence"])
        self.assertEqual(EVENT_FIELDS, set(event))
        await socket.disconnect()

    async def test_callback_terminal_event_is_ordered_then_closes_normally(self):
        run = self.create_run()
        self.append_event(
            run.run_id,
            4,
            "metric",
            {"step": 40, "metrics": {"average_reward": 0.8}},
        )
        self.append_event(
            run.run_id,
            7,
            "state",
            {"status": "completed", "training_steps": 40},
        )
        socket = ASGIWebSocketDriver(
            self.app, "/api/v1/runs/%s/events" % run.run_id
        )
        await socket.connect()
        self.assertEqual((await socket.receive_json())["sequence"], 4)
        terminal = await socket.receive_json()
        self.assertEqual(terminal["sequence"], 7)
        self.assertEqual(terminal["payload"]["status"], "completed")
        close = await socket.receive()
        self.assertEqual(
            close,
            {"type": "websocket.close", "code": 1000, "reason": ""},
        )
        await socket.finish()

    async def test_terminal_run_reconnect_catches_up_then_closes(self):
        run = self.create_run()
        stopped = self.app.state.run_service.request_stop(run.run_id)
        self.assertEqual(stopped.status.value, "stopped")
        self.append_event(
            run.run_id,
            2,
            "metric",
            {"step": 1, "metrics": {"average_reward": 0.1}},
        )
        socket = ASGIWebSocketDriver(
            self.app, "/api/v1/runs/%s/events?after_sequence=1" % run.run_id
        )
        await socket.connect()
        self.assertEqual((await socket.receive_json())["sequence"], 2)
        self.assertEqual((await socket.receive())["code"], 1000)
        await socket.finish()

    async def test_disconnect_never_stops_run_and_cleans_polling_task(self):
        run = self.create_run()
        socket = ASGIWebSocketDriver(
            self.app, "/api/v1/runs/%s/events" % run.run_id
        )
        await socket.connect()
        await socket.disconnect()
        self.assertEqual(
            self.app.state.run_service.get_run(run.run_id).status.value,
            "queued",
        )

    async def test_invalid_unknown_and_unauthorized_connections(self):
        owned = self.create_run()
        invalid_header = ASGIWebSocketDriver(
            self.app,
            "/api/v1/runs/%s/events" % owned.run_id,
            headers=[(b"authorization", b"Bearer invalid")],
        )
        self.assertEqual((await invalid_header.connect())["code"], 4401)
        await invalid_header.finish()
        missing_header = ASGIWebSocketDriver(
            self.app,
            "/api/v1/runs/%s/events" % owned.run_id,
            headers=[],
        )
        self.assertEqual(
            (await missing_header.connect())["type"], "websocket.accept"
        )
        await missing_header.send_json(
            {"type": "AUTH", "access_token": "invalid"}
        )
        self.assertEqual((await missing_header.receive())["code"], 4401)
        await missing_header.finish()
        run = self.create_run(user_id="another-user")
        for url, code in (
            ("/api/v1/runs/missing/events", 4404),
            ("/api/v1/runs/%s/events" % run.run_id, 4404),
            ("/api/v1/runs/%s/events?after_sequence=-1" % run.run_id, 4400),
            ("/api/v1/runs/%s/events?after_sequence=no" % run.run_id, 4400),
            (
                "/api/v1/runs/%s/events?after_sequence=1&after_sequence=2"
                % run.run_id,
                4400,
            ),
        ):
            with self.subTest(url=url):
                socket = ASGIWebSocketDriver(self.app, url)
                close = await socket.connect()
                self.assertEqual(close["type"], "websocket.close")
                self.assertEqual(close["code"], code)
                await socket.finish()

    async def test_repository_failure_closes_once_without_leaking_error(self):
        root = Path(self.temp_dir.name)
        repository = FailingEventRepository()
        app = create_app(
            root / "failing-runs",
            tick_seconds=0,
            project_root=root / "failing-projects",
            run_repository=repository,
            current_user_provider=StaticCurrentUserProvider("trusted-user"),
            run_event_poll_interval_seconds=0.01,
        )
        run = app.state.run_service.create_run(
            user_id="trusted-user",
            project_id="project-1",
            execution_mode=ExecutionMode.MOCK,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        socket = ASGIWebSocketDriver(
            app, "/api/v1/runs/%s/events" % run.run_id
        )
        with patch("apps.api.routers.runs.logger.error") as error_log:
            await socket.connect()
            close = await socket.receive()
        self.assertEqual(close["code"], 1011)
        error_log.assert_called_once_with(
            "Canonical Run Event stream repository poll failed"
        )
        self.assertNotIn("password", str(error_log.call_args))
        await socket.finish()
        polls = repository.poll_count
        await asyncio.sleep(0.03)
        self.assertEqual(repository.poll_count, polls)


SAFE_TEST_DATABASE_URL = safe_test_database_url()


@unittest.skipUnless(
    SAFE_TEST_DATABASE_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class V1RunWebSocketPostgreSQLTest(unittest.IsolatedAsyncioTestCase):
    async def test_two_apps_share_postgresql_event_stream(self):
        temp_dir = tempfile.TemporaryDirectory()
        prefix = "p61-%s-" % uuid4().hex
        user_id = prefix + "user"
        root = Path(temp_dir.name)
        cleanup_engine = create_postgres_engine(SAFE_TEST_DATABASE_URL)
        cleanup_factory = create_session_factory(cleanup_engine)
        seed_test_user(cleanup_factory, user_id)
        first = create_app(
            root / "first-runs",
            project_root=root / "first-projects",
            run_repository_mode="postgresql",
            database_url=SAFE_TEST_DATABASE_URL,
            current_user_provider=StaticCurrentUserProvider(user_id),
            run_event_poll_interval_seconds=0.01,
        )
        second = create_app(
            root / "second-runs",
            project_root=root / "second-projects",
            run_repository_mode="postgresql",
            database_url=SAFE_TEST_DATABASE_URL,
            current_user_provider=StaticCurrentUserProvider(user_id),
            run_event_poll_interval_seconds=0.01,
        )
        try:
            run = first.state.run_service.create_run(
                user_id=user_id,
                project_id=prefix + "project",
                execution_mode=ExecutionMode.SIMULATION,
                parameters={},
                project_config_snapshot={},
                code="",
            )
            first.state.run_service.bind_simulation_task(
                run.run_id, prefix + "task"
            )
            socket = ASGIWebSocketDriver(
                second, "/api/v1/runs/%s/events" % run.run_id
            )
            self.assertEqual(
                (await socket.connect())["type"], "websocket.accept"
            )
            occurred_at = datetime.now(timezone.utc)
            first.state.repository.append_event(
                RunEvent(
                    event_id=prefix + "event",
                    sequence=8,
                    type=RunEventType.HEARTBEAT,
                    run_id=run.run_id,
                    task_id=prefix + "task",
                    occurred_at=occurred_at,
                    payload={"status": "running", "training_steps": 8},
                )
            )
            self.assertEqual((await socket.receive_json())["sequence"], 8)
            terminal_event = RunEvent(
                event_id=prefix + "terminal",
                sequence=10,
                type=RunEventType.STATE,
                run_id=run.run_id,
                task_id=prefix + "task",
                occurred_at=occurred_at,
                payload={"status": "completed", "training_steps": 8},
            )
            first.state.terminal_callback_service.apply(
                TerminalCallbackCommand(
                    event=terminal_event,
                    status=RunStatus.COMPLETED,
                    started_at=occurred_at,
                    ended_at=occurred_at,
                    training_steps=8,
                    update_steps=1,
                    stop_reason=None,
                    metrics={},
                    error=None,
                    artifacts=(),
                )
            )
            terminal = await socket.receive_json()
            self.assertEqual(terminal["sequence"], 10)
            self.assertEqual(terminal["payload"]["status"], "completed")
            self.assertEqual((await socket.receive())["code"], 1000)
            await socket.finish()
        finally:
            with cleanup_factory() as session:
                run_ids = select(RunRow.run_id).where(RunRow.user_id == user_id)
                session.execute(
                    delete(RunIdempotencyKeyRow).where(
                        RunIdempotencyKeyRow.run_id.in_(run_ids)
                    )
                )
                session.execute(
                    delete(RunEventRow).where(RunEventRow.run_id.in_(run_ids))
                )
                session.execute(delete(RunRow).where(RunRow.user_id == user_id))
                session.commit()
            delete_test_user(cleanup_factory, user_id)
            first.state.engine.dispose()
            second.state.engine.dispose()
            cleanup_engine.dispose()
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
