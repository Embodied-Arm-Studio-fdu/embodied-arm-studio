"""Shared header-or-first-message authentication for browser WebSockets."""

import asyncio
from dataclasses import dataclass
import json
from typing import Any, Optional

from fastapi import HTTPException, WebSocket, WebSocketDisconnect


MAX_ACCESS_TOKEN_CHARACTERS = 4096
MAX_AUTH_MESSAGE_BYTES = 8192


@dataclass(frozen=True)
class WebSocketAuthentication:
    user: Any
    accepted: bool


async def authenticate_websocket(
    websocket: WebSocket,
    current_user_provider: Any,
    timeout_seconds: float = 5.0,
) -> Optional[WebSocketAuthentication]:
    """Authenticate without ever placing a token in a URL or response."""
    authorization = websocket.headers.get("authorization")
    if authorization is not None:
        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not token.strip()
            or len(token.strip()) > MAX_ACCESS_TOKEN_CHARACTERS
        ):
            await websocket.close(code=4401)
            return None
        try:
            user = current_user_provider.from_authorization(authorization)
        except HTTPException:
            await websocket.close(code=4401)
            return None
        return WebSocketAuthentication(user=user, accepted=False)

    await websocket.accept()
    try:
        message = await asyncio.wait_for(
            websocket.receive(), timeout=timeout_seconds
        )
    except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError):
        try:
            await websocket.close(code=4401)
        except RuntimeError:
            pass
        return None
    text = message.get("text")
    if message.get("type") == "websocket.disconnect" or not isinstance(text, str):
        try:
            await websocket.close(code=4401)
        except RuntimeError:
            pass
        return None
    if len(text.encode("utf-8")) > MAX_AUTH_MESSAGE_BYTES:
        await websocket.close(code=4401)
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        await websocket.close(code=4401)
        return None
    if not isinstance(payload, dict) or set(payload) != {"type", "access_token"}:
        await websocket.close(code=4401)
        return None
    token = payload.get("access_token")
    if (
        payload.get("type") != "AUTH"
        or not isinstance(token, str)
        or not token
        or len(token) > MAX_ACCESS_TOKEN_CHARACTERS
    ):
        await websocket.close(code=4401)
        return None
    try:
        user = current_user_provider.from_authorization("Bearer " + token)
    except HTTPException:
        await websocket.close(code=4401)
        return None
    await websocket.send_json({"type": "AUTH_ACK", "authenticated": True})
    return WebSocketAuthentication(user=user, accepted=True)
