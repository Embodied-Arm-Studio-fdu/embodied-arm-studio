"""Browser Input WebSocket routed to the recording-only Simulation boundary."""

from dataclasses import dataclass
import asyncio
import json
import math
import time
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from apps.api.dependencies.websocket_auth import authenticate_websocket
from robotics.runs import ExecutionMode, RepositoryError, RunNotFound, RunStatus
from robotics.run_access import (
    RunAccessForbidden,
    RunAccessNotFound,
    RunAccessUnavailable,
)
from robotics.simulation import (
    InputEventRequest,
    SimulationConfigurationError,
    SimulationNetworkError,
    SimulationRejectedError,
    SimulationResponseError,
    SimulationServerError,
    SimulationTimeoutError,
)


@dataclass(frozen=True)
class InputWebSocketConfig:
    auth_timeout_seconds: float = 5.0
    max_message_bytes: int = 8192
    messages_per_second: float = 60.0
    burst: int = 120
    rate_limit_close_after: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.auth_timeout_seconds, bool)
            or not isinstance(self.auth_timeout_seconds, (int, float))
            or not math.isfinite(self.auth_timeout_seconds)
            or self.auth_timeout_seconds <= 0
        ):
            raise ValueError("WebSocket auth timeout must be finite and positive")
        if (
            isinstance(self.max_message_bytes, bool)
            or not isinstance(self.max_message_bytes, int)
            or self.max_message_bytes <= 0
        ):
            raise ValueError("input message size must be a positive integer")
        if (
            isinstance(self.messages_per_second, bool)
            or not isinstance(self.messages_per_second, (int, float))
            or not math.isfinite(self.messages_per_second)
            or self.messages_per_second <= 0
        ):
            raise ValueError("input rate must be finite and positive")
        for name in ("burst", "rate_limit_close_after"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)


class _TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate = float(rate)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.updated_at = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(
            self.capacity,
            self.tokens + max(0.0, now - self.updated_at) * self.rate,
        )
        self.updated_at = now
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


def create_input_router(
    run_service: Any,
    simulation_client: Optional[Any],
    current_user_provider: Any,
    config: InputWebSocketConfig,
    run_access_service: Any,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/api/v1/runs/{run_id}/input")
    async def v1_run_input(websocket: WebSocket, run_id: str) -> None:
        authentication = await authenticate_websocket(
            websocket,
            current_user_provider,
            timeout_seconds=config.auth_timeout_seconds,
        )
        if authentication is None:
            return
        loop = asyncio.get_running_loop()
        try:
            run = run_access_service.control(
                run_id, authentication.user.user_id
            )
        except RunAccessNotFound:
            await websocket.close(code=4404)
            return
        except RunAccessForbidden:
            await websocket.close(code=4403)
            return
        except RunAccessUnavailable:
            await websocket.close(code=1011)
            return
        except Exception:
            await websocket.close(code=1011)
            return
        if not authentication.accepted:
            await websocket.accept()

        limiter = _TokenBucket(config.messages_per_second, config.burst)
        rate_limit_violations = 0
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                raw_size = len(message.get("text", "").encode("utf-8"))
                if message.get("bytes") is not None:
                    raw_size = len(message["bytes"])
                if raw_size > config.max_message_bytes:
                    await websocket.close(code=1009)
                    return
                if not limiter.allow():
                    rate_limit_violations += 1
                    await _send_error(websocket, "rate_limited")
                    if rate_limit_violations >= config.rate_limit_close_after:
                        await websocket.close(code=4429)
                        return
                    continue
                rate_limit_violations = 0
                if "bytes" in message and message.get("bytes") is not None:
                    await _send_error(websocket, "invalid_request")
                    continue
                text = message.get("text")
                if not isinstance(text, str):
                    await _send_error(websocket, "invalid_request")
                    continue
                try:
                    payload = json.loads(text)
                    event = InputEventRequest.from_mapping(payload)
                except (TypeError, ValueError):
                    await _send_error(websocket, "invalid_request")
                    continue
                if event.run_id != run_id:
                    await _send_error(
                        websocket, "invalid_request", event_id=event.event_id
                    )
                    continue

                try:
                    current = run_access_service.control(
                        run_id, authentication.user.user_id
                    )
                except RunAccessNotFound:
                    await _send_error(websocket, "run_not_found")
                    await websocket.close(code=4404)
                    return
                except RunAccessForbidden:
                    await _send_error(websocket, "input_not_allowed")
                    await websocket.close(code=4403)
                    return
                except RunAccessUnavailable:
                    await _send_error(
                        websocket, "simulation_unavailable", retryable=True
                    )
                    continue
                if (
                    current.execution_mode != ExecutionMode.SIMULATION
                    or current.status != RunStatus.RUNNING
                    or not current.simulation_task_id
                ):
                    await _send_error(
                        websocket,
                        "input_not_allowed",
                        event_id=event.event_id,
                    )
                    await websocket.close(code=4409)
                    return
                if event.task_id != current.simulation_task_id:
                    await _send_error(
                        websocket, "invalid_request", event_id=event.event_id
                    )
                    continue
                if simulation_client is None:
                    await _send_error(
                        websocket,
                        "simulation_unavailable",
                        event_id=event.event_id,
                        retryable=True,
                    )
                    continue
                try:
                    acknowledgement = await loop.run_in_executor(
                        None, simulation_client.send_input_event, event
                    )
                except SimulationRejectedError as exc:
                    if exc.status_code == 409 and exc.code == "event_conflict":
                        await _send_error(
                            websocket, "event_conflict", event_id=event.event_id
                        )
                    elif exc.status_code == 409 and exc.code == "input_not_allowed":
                        await _send_error(
                            websocket,
                            "input_not_allowed",
                            event_id=event.event_id,
                        )
                        await websocket.close(code=4409)
                        return
                    elif exc.status_code in {401, 404} or exc.status_code >= 500:
                        await _send_error(
                            websocket,
                            "simulation_unavailable",
                            event_id=event.event_id,
                            retryable=True,
                        )
                    else:
                        await _send_error(
                            websocket,
                            "upstream_invalid_response",
                            event_id=event.event_id,
                        )
                    continue
                except (
                    SimulationTimeoutError,
                    SimulationNetworkError,
                    SimulationServerError,
                ):
                    await _send_error(
                        websocket,
                        "simulation_unavailable",
                        event_id=event.event_id,
                        retryable=True,
                    )
                    continue
                except (SimulationResponseError, SimulationConfigurationError):
                    await _send_error(
                        websocket,
                        "upstream_invalid_response",
                        event_id=event.event_id,
                    )
                    continue
                await websocket.send_json(
                    {
                        "type": "ACK",
                        "event_id": acknowledgement.event_id,
                        "accepted": acknowledgement.accepted,
                        "duplicate": acknowledgement.duplicate,
                    }
                )
        except (WebSocketDisconnect, RuntimeError):
            return

    return router


async def _send_error(
    websocket: WebSocket,
    code: str,
    event_id: Optional[str] = None,
    retryable: bool = False,
) -> None:
    await websocket.send_json(
        {
            "type": "ERROR",
            "code": code,
            "event_id": event_id,
            "retryable": retryable,
        }
    )
