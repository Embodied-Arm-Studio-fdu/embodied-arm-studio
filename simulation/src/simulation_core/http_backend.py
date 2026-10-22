"""HTTP implementation of the simulation-to-backend reporting port."""

import json
import logging
import os
import socket
from typing import Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .ports import BackendClient


LOGGER = logging.getLogger(__name__)


class BackendHttpError(RuntimeError):
    def __init__(self, message, status_code=None, retryable=True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = bool(retryable)


class BackendRequestRejected(BackendHttpError):
    """The backend returned 4xx for a request it understood."""

    def __init__(self, message, status_code):
        super().__init__(message, status_code=status_code, retryable=False)


class BackendTemporarilyUnavailable(BackendHttpError):
    """Network, timeout, or 5xx failure that may succeed later."""


class HttpBackendClient(BackendClient):
    EVENTS_PATH = "/api/internal/v1/runs/{run_id}/events"
    CALLBACK_PATH = "/api/internal/v1/runs/{run_id}/callback"

    def __init__(
        self, backend_base_url: str, request_timeout: float = 5.0,
        service_token: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        opener=None,
    ) -> None:
        base_url = str(backend_base_url).rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("backend_base_url must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("backend_base_url must not contain query or fragment")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        self.backend_base_url = base_url
        self.request_timeout = float(request_timeout)
        self.headers = {str(key): str(value) for key, value in dict(headers or {}).items()}
        if service_token:
            self.headers.setdefault("Authorization", "Bearer %s" % service_token)
        # Internal service traffic must not be silently redirected through
        # workstation HTTP_PROXY settings. Custom transport remains injectable.
        self._opener = opener or build_opener(ProxyHandler({})).open

    @classmethod
    def from_env(cls, required=True, environ=None):
        values = environ if environ is not None else os.environ
        base_url = values.get("SIMULATION_BACKEND_BASE_URL")
        if not base_url:
            if required:
                raise ValueError("SIMULATION_BACKEND_BASE_URL is required")
            return None
        timeout = float(values.get("SIMULATION_BACKEND_REQUEST_TIMEOUT", "5"))
        token = values.get("SIMULATION_BACKEND_SERVICE_TOKEN")
        return cls(base_url, request_timeout=timeout, service_token=token)

    def report_status(self, task) -> None:
        # Legacy Phase 1 hook; runtime state is sent through publish_event.
        return None

    def publish_event(self, event) -> None:
        self._post(self.EVENTS_PATH, event.run_id, event.to_dict(), "event")

    def publish_terminal_callback(self, callback) -> None:
        self._post(
            self.CALLBACK_PATH, callback.run_id, callback.to_dict(),
            "terminal callback",
        )

    def _post(self, path_template, run_id, payload, operation):
        path = path_template.format(run_id=quote(str(run_id), safe=""))
        url = self.backend_base_url + path
        body = json.dumps(
            payload, separators=(",", ":"), sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(self.headers)
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with self._opener(request, timeout=self.request_timeout) as response:
                status = int(response.getcode())
                response.read(4096)
        except HTTPError as exc:
            status = int(exc.code)
            message = "%s rejected with HTTP %d" % (operation, status)
            error = (
                BackendRequestRejected(message, status)
                if 400 <= status < 500
                else BackendTemporarilyUnavailable(message, status_code=status)
            )
            LOGGER.error("backend %s POST failed: %s", operation, error)
            exc.close()
            raise error from exc
        except (URLError, socket.timeout, TimeoutError, OSError) as exc:
            error = BackendTemporarilyUnavailable(
                "%s transport failed: %s" % (operation, exc)
            )
            LOGGER.error("backend %s POST failed: %s", operation, error)
            raise error from exc
        if not 200 <= status < 300:
            error = BackendTemporarilyUnavailable(
                "%s returned unexpected HTTP %d" % (operation, status),
                status_code=status,
            )
            LOGGER.error("backend %s POST failed: %s", operation, error)
            raise error
