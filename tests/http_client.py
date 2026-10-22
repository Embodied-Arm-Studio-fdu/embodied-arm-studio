"""Reliable synchronous facade over real in-process ASGI HTTP requests."""

import asyncio
from typing import Any

import httpx


class ASGITestClient:
    """Drive HTTPX ASGITransport while keeping Python 3.8's loop responsive."""

    def __init__(self, app: Any, request_timeout: float = 5.0) -> None:
        self.app = app
        self.request_timeout = request_timeout
        self._loop = asyncio.new_event_loop()
        self._client = None
        self._closed = False
        self._run(self._open())

    async def _open(self) -> None:
        await self.app.router.startup()
        provider = getattr(self.app.state, "current_user_provider", None)
        self._test_authorization = getattr(
            provider, "test_authorization_header", None
        )
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://testserver",
        )

    async def _drive(self, awaitable: Any) -> Any:
        stop = asyncio.Event()

        async def tick() -> None:
            while not stop.is_set():
                await asyncio.sleep(0.001)

        ticker = asyncio.create_task(tick())
        try:
            return await asyncio.wait_for(awaitable, timeout=self.request_timeout)
        finally:
            stop.set()
            await ticker

    def _run(self, awaitable: Any) -> Any:
        if self._closed:
            raise RuntimeError("ASGITestClient is closed")
        return self._loop.run_until_complete(self._drive(awaitable))

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if (
            (url.startswith("/api/v1/") or url.startswith("/api/agent/"))
            and self._test_authorization
        ):
            headers = dict(kwargs.get("headers") or {})
            if not any(key.lower() == "authorization" for key in headers):
                headers["Authorization"] = self._test_authorization
            kwargs["headers"] = headers
        return self._run(self._client.request(method, url, **kwargs))

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def close(self) -> None:
        if self._closed:
            return

        async def close_resources() -> None:
            await self._client.aclose()
            await self.app.router.shutdown()

        self._run(close_resources())
        self._closed = True
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.close()

    def __enter__(self) -> "ASGITestClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
