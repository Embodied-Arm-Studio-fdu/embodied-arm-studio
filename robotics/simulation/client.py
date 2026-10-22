"""Synchronous HTTPX client for the external Simulation Service."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union
from urllib.parse import urlsplit

import httpx

from .contracts import (
    CreateStreamSessionRequest,
    CreateTaskRequest,
    InputEventAck,
    InputEventRequest,
    SimulationTaskResponse,
    SimulationTaskCatalog,
    StreamDescriptor,
    validate_identifier,
)


class SimulationClientError(RuntimeError):
    pass


class SimulationConfigurationError(SimulationClientError):
    pass


class SimulationClientClosedError(SimulationConfigurationError):
    pass


class SimulationTemporaryError(SimulationClientError):
    pass


class SimulationTimeoutError(SimulationTemporaryError):
    pass


class SimulationNetworkError(SimulationTemporaryError):
    pass


class SimulationServerError(SimulationTemporaryError):
    def __init__(
        self, status_code: int, code: Optional[str] = None,
        upstream_message: Optional[str] = None,
    ) -> None:
        super().__init__("Simulation Service returned a temporary server error (%d)" % status_code)
        self.status_code = status_code
        self.code = code
        self.upstream_message = upstream_message


class SimulationRejectedError(SimulationClientError):
    def __init__(
        self, status_code: int, code: Optional[str] = None,
        upstream_message: Optional[str] = None,
    ) -> None:
        super().__init__("Simulation Service rejected the request (%d)" % status_code)
        self.status_code = status_code
        self.code = code
        self.upstream_message = upstream_message


class SimulationResponseError(SimulationClientError):
    pass


TimeoutValue = Union[float, httpx.Timeout]


class SimulationClient:
    def __init__(
        self,
        base_url: str,
        timeout: TimeoutValue = 5.0,
        service_token: Optional[str] = None,
        *,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        normalized_url = self._validate_base_url(base_url)
        self._validate_timeout(timeout)
        if not isinstance(service_token, str) or not service_token.strip():
            raise SimulationConfigurationError("SIMULATION_SERVICE_TOKEN is required")
        headers = {
            "Accept": "application/json",
            "Authorization": "Bearer %s" % service_token,
        }
        self._client = httpx.Client(
            base_url=normalized_url,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )
        self._service_token = service_token
        self._closed = False

    @staticmethod
    def _validate_base_url(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SimulationConfigurationError("SIMULATION_BASE_URL is required")
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise SimulationConfigurationError("SIMULATION_BASE_URL is invalid") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise SimulationConfigurationError(
                "SIMULATION_BASE_URL must be an HTTP(S) origin without credentials, query, or fragment"
            )
        if port is not None and not 1 <= port <= 65535:
            raise SimulationConfigurationError("SIMULATION_BASE_URL has an invalid port")
        return value.rstrip("/") + "/"

    @staticmethod
    def _validate_timeout(value: TimeoutValue) -> None:
        if isinstance(value, bool):
            raise SimulationConfigurationError("timeout must be positive")
        if isinstance(value, (int, float)) and value <= 0:
            raise SimulationConfigurationError("timeout must be positive")
        if not isinstance(value, (int, float, httpx.Timeout)):
            raise SimulationConfigurationError("timeout must be a number or httpx.Timeout")

    def create_task(
        self, run_id: str, project_id: str, config: Optional[Mapping[str, Any]] = None
    ) -> SimulationTaskResponse:
        try:
            request = CreateTaskRequest(
                run_id, project_id, {} if config is None else config
            )
        except ValueError as exc:
            raise SimulationConfigurationError(str(exc)) from exc
        return self._request(
            "POST", "/simulation/tasks", json=request.to_dict(), expected_status=202
        )

    def get_task(self, task_id: str) -> SimulationTaskResponse:
        return self._task_request("GET", task_id)

    def stop_task(self, task_id: str) -> SimulationTaskResponse:
        return self._task_request("POST", task_id, suffix="/stop")

    def get_task_catalog(self) -> SimulationTaskCatalog:
        payload = self._request_json(
            "GET", "/simulation/task-catalog", expected_status=200
        )
        try:
            return SimulationTaskCatalog.from_dict(payload)
        except ValueError as exc:
            raise SimulationResponseError(
                "Simulation Service response schema is invalid"
            ) from exc

    def create_stream_session(
        self, task_id: str, camera_id: str = "overview"
    ) -> StreamDescriptor:
        try:
            safe_task_id = validate_identifier("task_id", task_id)
            request = CreateStreamSessionRequest(camera_id)
        except ValueError as exc:
            raise SimulationConfigurationError(str(exc)) from exc
        payload = self._request_json(
            "POST",
            "/simulation/tasks/%s/stream-session" % safe_task_id,
            json=request.to_dict(),
            expected_status=200,
        )
        try:
            return StreamDescriptor.from_dict(
                payload, forbidden_values=(self._service_token,)
            )
        except ValueError as exc:
            raise SimulationResponseError(
                "Simulation Service response schema is invalid"
            ) from exc

    def send_input_event(self, event: InputEventRequest) -> InputEventAck:
        if not isinstance(event, InputEventRequest):
            raise SimulationConfigurationError(
                "input event must be an InputEventRequest"
            )
        payload = self._request_json(
            "POST",
            "/simulation/tasks/%s/input-events" % event.task_id,
            json=event.to_dict(),
            expected_status=202,
        )
        try:
            acknowledgement = InputEventAck.from_dict(payload)
        except ValueError as exc:
            raise SimulationResponseError(
                "Simulation Service response schema is invalid"
            ) from exc
        if acknowledgement.event_id != event.event_id:
            raise SimulationResponseError(
                "Simulation Service response event_id does not match request"
            )
        return acknowledgement

    def _task_request(
        self, method: str, task_id: str, suffix: str = ""
    ) -> SimulationTaskResponse:
        try:
            safe_task_id = validate_identifier("task_id", task_id)
        except ValueError as exc:
            raise SimulationConfigurationError(str(exc)) from exc
        return self._request(
            method, "/simulation/tasks/%s%s" % (safe_task_id, suffix),
            expected_status=200,
        )

    def _request(
        self, method: str, path: str, json: Optional[Mapping[str, Any]] = None,
        expected_status: int = 200,
    ) -> SimulationTaskResponse:
        payload = self._request_json(
            method, path, json=json, expected_status=expected_status
        )
        try:
            return SimulationTaskResponse.from_dict(payload)
        except ValueError as exc:
            raise SimulationResponseError(
                "Simulation Service response schema is invalid"
            ) from exc

    def _request_json(
        self,
        method: str,
        path: str,
        json: Optional[Mapping[str, Any]] = None,
        expected_status: Optional[int] = 200,
    ) -> Mapping[str, Any]:
        if self._closed:
            raise SimulationClientClosedError("SimulationClient is closed")
        try:
            response = self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise SimulationTimeoutError("Simulation Service request timed out") from exc
        except httpx.RequestError as exc:
            raise SimulationNetworkError("Simulation Service network request failed") from exc

        if 400 <= response.status_code:
            code, upstream_message = self._parse_error(response)
            if 500 <= response.status_code:
                raise SimulationServerError(
                    response.status_code, code, upstream_message
                )
            raise SimulationRejectedError(
                response.status_code, code, upstream_message
            )
        if expected_status is None and not 200 <= response.status_code < 300:
            raise SimulationResponseError(
                "Simulation Service returned an unexpected non-success status"
            )
        if expected_status is not None and response.status_code != expected_status:
            raise SimulationResponseError(
                "Simulation Service returned an unexpected success status (%d)"
                % response.status_code
            )
        try:
            payload = response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise SimulationResponseError("Simulation Service returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise SimulationResponseError("Simulation Service response must be a JSON object")
        return payload

    @staticmethod
    def _parse_error(response: httpx.Response) -> tuple[Optional[str], Optional[str]]:
        """Parse the stable error envelope without rendering it in exceptions."""
        try:
            payload = response.json()
        except (ValueError, UnicodeDecodeError):
            return None, None
        if not isinstance(payload, Mapping):
            return None, None
        code = payload.get("code")
        message = payload.get("message")
        if not isinstance(code, str) or not code:
            code = None
        if not isinstance(message, str) or not message:
            message = None
        return code, message

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def __enter__(self) -> "SimulationClient":
        if self._closed:
            raise SimulationClientClosedError("SimulationClient is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

