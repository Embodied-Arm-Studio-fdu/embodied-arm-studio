"""Explicit authentication dependency used by isolated non-JWT tests."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import Header, HTTPException, WebSocket

from apps.api.dependencies.auth import unauthorized
from robotics.auth import User


class StaticCurrentUserProvider:
    test_authorization_header = "Bearer phase-7-1-test-credential"

    def __init__(self, user_id: str, enabled: bool = True) -> None:
        now = datetime.now(timezone.utc)
        self.user = User(
            user_id=user_id,
            username=user_id,
            display_name=user_id,
            password_hash="not-used-by-static-test-provider",
            enabled=enabled,
            created_at=now,
            updated_at=now,
        )

    def __call__(
        self, authorization: Optional[str] = Header(None)
    ) -> User:
        return self.from_authorization(authorization or "")

    def from_authorization(self, header: str) -> User:
        if header != self.test_authorization_header or not self.user.enabled:
            raise unauthorized()
        return self.user

    def from_websocket(self, websocket: WebSocket) -> Optional[User]:
        try:
            return self.from_authorization(
                websocket.headers.get("authorization", "")
            )
        except HTTPException:
            return None
