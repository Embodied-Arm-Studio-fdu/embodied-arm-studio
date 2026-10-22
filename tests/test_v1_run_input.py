import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from threading import Lock
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import jwt

from apps.api.dependencies.auth import CurrentUserProvider
from apps.api.main import create_app
from robotics.auth import AuthenticationFailed, JWTConfiguration, User
from robotics.auth.tokens import JWTAccessTokenService
from robotics.runs import ExecutionMode, InMemoryRunRepository, RunKind, RunStatus
from robotics.simulation import (
    InputEventAck,
    SimulationNetworkError,
    SimulationRejectedError,
    SimulationResponseError,
    SimulationServerError,
    SimulationTimeoutError,
)
from tests.auth import StaticCurrentUserProvider
from tests.test_v1_run_websocket import ASGIWebSocketDriver


class FakeSimulationClient:
    def __init__(self):
        self.calls = []
        self.stop_calls = []
        self.closed_count = 0
        self.error = None
        self.delay = 0.0
        self.max_active = 0
        self._active = 0
        self._events = {}
        self._lock = Lock()

    def create_task(self, run_id, project_id, config):
        raise AssertionError("input tests must not dispatch tasks")

    def stop_task(self, task_id):
        self.stop_calls.append(task_id)
        raise AssertionError("input gateway must not stop tasks")

    def send_input_event(self, event):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls.append(event)
            if self.error is not None:
                raise self.error
            key = (event.task_id, event.event_id)
            existing = self._events.get(key)
            if existing is not None and existing != event:
                raise SimulationRejectedError(409, "event_conflict", "private")
            duplicate = existing is not None
            self._events[key] = event
            return InputEventAck(True, event.event_id, duplicate)
        finally:
            with self._lock:
                self._active -= 1

    def close(self):
        self.closed_count += 1


class _TokenResolvingService:
    def __init__(self, tokens, user):
        self.tokens = tokens
        self.user = user

    def resolve_access_token(self, token):
        claims = self.tokens.decode(token)
        if claims["sub"] != self.user.user_id or not self.user.enabled:
            raise AuthenticationFailed("invalid access token")
        return self.user


def _user(enabled=True):
    now = datetime.now(timezone.utc)
    return User(
        user_id="trusted-user",
        username="trusted-user",
        display_name="Trusted User",
        password_hash="not-exposed",
        enabled=enabled,
        created_at=now,
        updated_at=now,
    )


class V1RunInputWebSocketTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.simulation = FakeSimulationClient()
        self.app = self.make_app()

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_app(self, provider=None, **config):
        root = Path(self.temp_dir.name) / uuid4().hex
        return create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=provider
            or StaticCurrentUserProvider("trusted-user"),
            simulation_client=self.simulation,
            **config
        )

    def create_run(
        self,
        app=None,
        user_id="trusted-user",
        mode=ExecutionMode.SIMULATION,
        bind=True,
        status=RunStatus.RUNNING,
    ):
        app = app or self.app
        service = app.state.run_service
        run = service.create_run(
            user_id=user_id,
            project_id="project-1",
            execution_mode=mode,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        if mode == ExecutionMode.SIMULATION and bind:
            run = service.bind_simulation_task(run.run_id, "task-" + run.run_id)
        if status != RunStatus.QUEUED:
            run = service.transition_status(run.run_id, RunStatus.INITIALIZING)
        if status not in {RunStatus.QUEUED, RunStatus.INITIALIZING}:
            run = service.transition_status(run.run_id, RunStatus.RUNNING)
        if status in {RunStatus.FINALIZING, RunStatus.COMPLETED}:
            run = service.transition_status(run.run_id, RunStatus.FINALIZING)
        if status == RunStatus.COMPLETED:
            run = service.transition_status(run.run_id, RunStatus.COMPLETED)
        return run

    @staticmethod
    def payload(run, kind="KEYBOARD", **changes):
        value = {
            "event_id": "input-1",
            "run_id": run.run_id,
            "task_id": run.simulation_task_id or "task-unbound",
            "type": kind,
            "occurred_at": "2026-08-28T00:00:00+00:00",
        }
        if kind == "KEYBOARD":
            value.update({"key": "W", "action": "pressed"})
        elif kind == "MOUSE_MOVE":
            value.update({"x": 0.4, "y": 0.6})
        else:
            value.update(
                {"button": "left", "action": "released", "x": 0.4, "y": 0.6}
            )
        value.update(changes)
        return value

    def socket(self, run, app=None, headers=None):
        return ASGIWebSocketDriver(
            app or self.app,
            "/api/v1/runs/%s/input" % run.run_id,
            headers=headers,
        )

    async def test_header_auth_three_types_and_duplicate_ack(self):
        run = self.create_run()
        socket = self.socket(run)
        self.assertEqual("websocket.accept", (await socket.connect())["type"])
        for index, kind in enumerate(("KEYBOARD", "MOUSE_MOVE", "MOUSE_BUTTON")):
            body = self.payload(run, kind, event_id="input-%d" % index)
            await socket.send_json(body)
            self.assertEqual(
                {
                    "type": "ACK",
                    "event_id": body["event_id"],
                    "accepted": True,
                    "duplicate": False,
                },
                await socket.receive_json(),
            )
        repeated = self.payload(run, event_id="input-0")
        await socket.send_json(repeated)
        self.assertTrue((await socket.receive_json())["duplicate"])
        self.assertEqual(4, len(self.simulation.calls))
        self.assertEqual(3, len(self.simulation._events))
        await socket.disconnect()

    async def test_browser_auth_ack_and_owner_isolation(self):
        run = self.create_run()
        socket = self.socket(run, headers=[])
        self.assertEqual("websocket.accept", (await socket.connect())["type"])
        await socket.send_json(
            {"type": "AUTH", "access_token": "phase-7-1-test-credential"}
        )
        self.assertEqual(
            {"type": "AUTH_ACK", "authenticated": True},
            await socket.receive_json(),
        )
        await socket.send_json(self.payload(run))
        self.assertEqual("ACK", (await socket.receive_json())["type"])
        await socket.disconnect()

        other = self.create_run(user_id="other-user")
        hidden = self.socket(other)
        self.assertEqual(4404, (await hidden.connect())["code"])
        await hidden.finish()

    async def test_auth_timeout_invalid_expired_disabled_and_no_token_leak(self):
        run = self.create_run()
        timeout_app = self.make_app(websocket_auth_timeout_seconds=0.01)
        timeout_run = self.create_run(timeout_app)
        timeout_socket = self.socket(timeout_run, timeout_app, headers=[])
        await timeout_socket.connect()
        self.assertEqual(4401, (await timeout_socket.receive(timeout=0.2))["code"])
        await timeout_socket.finish()

        secret = "never-render-this-browser-jwt"
        for auth_message in (
            {"type": "AUTH", "access_token": secret},
            {"type": "AUTH", "access_token": secret, "extra": True},
            {"type": "AUTH", "access_token": "x" * 4097},
        ):
            socket = self.socket(run, headers=[])
            await socket.connect()
            with patch("apps.api.routers.runs.logger.error") as error_log:
                await socket.send_json(auth_message)
                close = await socket.receive()
            self.assertEqual(4401, close["code"])
            self.assertNotIn(secret, json.dumps(close))
            self.assertNotIn(secret, str(error_log.call_args_list))
            await socket.finish()

        config = JWTConfiguration(
            secret="phase-7-1-test-secret-at-least-32-bytes",
            issuer="test-issuer",
            audience="test-audience",
        )
        tokens = JWTAccessTokenService(config)
        expired_claims = {
            "sub": "trusted-user",
            "iss": config.issuer,
            "aud": config.audience,
            "iat": datetime.now(timezone.utc) - timedelta(minutes=2),
            "nbf": datetime.now(timezone.utc) - timedelta(minutes=2),
            "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
            "jti": "expired-token-id",
        }
        expired = jwt.encode(expired_claims, config.secret, algorithm="HS256")
        for enabled, token in ((True, expired), (False, tokens.issue("trusted-user"))):
            provider = CurrentUserProvider(_TokenResolvingService(tokens, _user(enabled)))
            app = self.make_app(provider=provider)
            owned = self.create_run(app)
            socket = self.socket(owned, app, headers=[])
            await socket.connect()
            await socket.send_json({"type": "AUTH", "access_token": token})
            self.assertEqual(4401, (await socket.receive())["code"])
            await socket.finish()

    async def test_run_and_task_mismatch_and_strict_validation(self):
        run = self.create_run()
        socket = self.socket(run)
        await socket.connect()
        cases = [
            self.payload(run, run_id="another-run"),
            self.payload(run, task_id="another-task"),
            dict(self.payload(run), unknown=True),
            self.payload(run, kind="MOUSE_MOVE", x=True),
            self.payload(run, occurred_at="2026-08-28T08:00:00+08:00"),
        ]
        for body in cases:
            await socket.send_json(body)
            self.assertEqual("invalid_request", (await socket.receive_json())["code"])
        self.assertEqual([], self.simulation.calls)
        await socket.disconnect()

    async def test_mock_queued_terminal_and_unbound_runs_close_4409(self):
        runs = [
            self.create_run(mode=ExecutionMode.MOCK, bind=False, status=RunStatus.QUEUED),
            self.create_run(status=RunStatus.QUEUED),
            self.create_run(status=RunStatus.FINALIZING),
            self.create_run(status=RunStatus.COMPLETED),
            self.create_run(bind=False, status=RunStatus.RUNNING),
        ]
        for run in runs:
            socket = self.socket(run)
            await socket.connect()
            await socket.send_json(self.payload(run))
            error = await socket.receive_json()
            self.assertEqual("input_not_allowed", error["code"])
            self.assertEqual(4409, (await socket.receive())["code"])
            await socket.finish()
        self.assertEqual([], self.simulation.calls)

    async def test_each_message_rechecks_canonical_run_status(self):
        run = self.create_run()
        socket = self.socket(run)
        await socket.connect()
        await socket.send_json(self.payload(run))
        self.assertEqual("ACK", (await socket.receive_json())["type"])
        self.app.state.run_service.transition_status(run.run_id, RunStatus.FINALIZING)
        await socket.send_json(self.payload(run, event_id="input-2"))
        self.assertEqual("input_not_allowed", (await socket.receive_json())["code"])
        self.assertEqual(4409, (await socket.receive())["code"])
        await socket.finish()
        self.assertEqual(1, len(self.simulation.calls))

    async def test_event_conflict_and_upstream_error_mapping_is_redacted(self):
        run = self.create_run()
        socket = self.socket(run)
        await socket.connect()
        first = self.payload(run)
        await socket.send_json(first)
        await socket.receive_json()
        await socket.send_json(dict(first, key="A"))
        conflict = await socket.receive_json()
        self.assertEqual("event_conflict", conflict["code"])

        cases = [
            (SimulationTimeoutError("private"), "simulation_unavailable", True),
            (SimulationNetworkError("private"), "simulation_unavailable", True),
            (SimulationServerError(503, "private", "private"), "simulation_unavailable", True),
            (SimulationRejectedError(401, "unauthorized", "private"), "simulation_unavailable", True),
            (SimulationRejectedError(404, "task_not_found", "private"), "simulation_unavailable", True),
            (SimulationRejectedError(400, "invalid_request", "private"), "upstream_invalid_response", False),
            (SimulationResponseError("private URL token body"), "upstream_invalid_response", False),
        ]
        for index, (error, code, retryable) in enumerate(cases):
            self.simulation.error = error
            await socket.send_json(self.payload(run, event_id="error-%d" % index))
            response = await socket.receive_json()
            self.assertEqual((code, retryable), (response["code"], response["retryable"]))
            self.assertNotIn("private", json.dumps(response))
        self.simulation.error = None
        self.assertEqual(RunStatus.RUNNING, self.app.state.run_service.get_run(run.run_id).status)
        self.assertEqual([], self.simulation.stop_calls)
        await socket.disconnect()

    async def test_upstream_input_not_allowed_and_missing_client(self):
        run = self.create_run()
        socket = self.socket(run)
        await socket.connect()
        self.simulation.error = SimulationRejectedError(
            409, "input_not_allowed", "private"
        )
        await socket.send_json(self.payload(run))
        self.assertEqual("input_not_allowed", (await socket.receive_json())["code"])
        self.assertEqual(4409, (await socket.receive())["code"])
        await socket.finish()
        self.assertEqual(RunStatus.RUNNING, self.app.state.run_service.get_run(run.run_id).status)
        self.assertEqual([], self.simulation.stop_calls)

        root = Path(self.temp_dir.name) / "unconfigured"
        app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider("trusted-user"),
            simulation_environment={},
        )
        unconfigured = self.create_run(app)
        missing = self.socket(unconfigured, app)
        await missing.connect()
        await missing.send_json(self.payload(unconfigured))
        response = await missing.receive_json()
        self.assertEqual(("simulation_unavailable", True), (response["code"], response["retryable"]))
        await missing.disconnect()

    async def test_message_size_binary_invalid_json_and_rate_limit(self):
        run = self.create_run()
        socket = self.socket(run)
        await socket.connect()
        await socket.send_bytes(b"not-json")
        self.assertEqual("invalid_request", (await socket.receive_json())["code"])
        await socket.send_text("{")
        self.assertEqual("invalid_request", (await socket.receive_json())["code"])
        compact = json.dumps(self.payload(run), separators=(",", ":"))
        boundary = " " * (8192 - len(compact.encode("utf-8"))) + compact
        await socket.send_text(boundary)
        self.assertEqual("ACK", (await socket.receive_json())["type"])
        await socket.send_text(" " + boundary)
        self.assertEqual(1009, (await socket.receive())["code"])
        await socket.finish()

        limited_app = self.make_app(
            input_ws_messages_per_second=0.0001,
            input_ws_burst=1,
            input_ws_rate_limit_close_after=3,
        )
        limited_run = self.create_run(limited_app)
        limited = self.socket(limited_run, limited_app)
        await limited.connect()
        for index in range(4):
            await limited.send_json(self.payload(limited_run, event_id="rate-%d" % index))
        self.assertEqual("ACK", (await limited.receive_json())["type"])
        for _ in range(3):
            self.assertEqual("rate_limited", (await limited.receive_json())["code"])
        self.assertEqual(4429, (await limited.receive())["code"])
        await limited.finish()

    async def test_forwarding_is_serial_disconnect_never_stops_and_shutdown_closes_once(self):
        run = self.create_run()
        self.simulation.delay = 0.02
        socket = self.socket(run)
        await socket.connect()
        for index in range(5):
            await socket.send_json(self.payload(run, event_id="serial-%d" % index))
        for _ in range(5):
            self.assertEqual("ACK", (await socket.receive_json())["type"])
        self.assertEqual(1, self.simulation.max_active)
        await socket.disconnect()
        self.assertEqual([], self.simulation.stop_calls)
        self.assertEqual(RunStatus.RUNNING, self.app.state.run_service.get_run(run.run_id).status)
        await self.app.router.shutdown()
        self.assertEqual(1, self.simulation.closed_count)

    def test_invalid_limit_configuration_fails_fast(self):
        for values in (
            {"input_ws_max_message_bytes": 0},
            {"input_ws_messages_per_second": 0},
            {"input_ws_messages_per_second": float("inf")},
            {"input_ws_burst": 0},
            {"websocket_auth_timeout_seconds": 0},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.make_app(**values)


if __name__ == "__main__":
    unittest.main()
