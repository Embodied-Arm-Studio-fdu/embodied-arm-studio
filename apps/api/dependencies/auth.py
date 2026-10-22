"""Bearer JWT authentication dependency for browser-facing canonical APIs."""

from typing import Optional

from fastapi import HTTPException, Security, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from robotics.auth import (
    AuthenticationFailed,
    AuthenticationService,
    JWTTokenError,
    User,
    UserRepositoryError,
)


_bearer = HTTPBearer(auto_error=False)


def unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Invalid or missing bearer credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


class CurrentUserProvider:
    def __init__(self, service: AuthenticationService) -> None:
        self.service = service

    def __call__(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
    ) -> User:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise unauthorized()
        return self.resolve(credentials.credentials)

    def resolve(self, token: str) -> User:
        try:
            return self.service.resolve_access_token(token)
        except (AuthenticationFailed, JWTTokenError, UserRepositoryError):
            raise unauthorized()

    def from_authorization(self, header: str) -> User:
        scheme, separator, token = header.partition(" ")
        if not separator or scheme.lower() != "bearer" or not token.strip():
            raise unauthorized()
        return self.resolve(token.strip())

    def from_websocket(self, websocket: WebSocket) -> Optional[User]:
        try:
            return self.from_authorization(
                websocket.headers.get("authorization", "")
            )
        except HTTPException:
            return None


class UnavailableCurrentUserProvider:
    """Fail closed when a non-PostgreSQL test app omitted an auth provider."""

    def __call__(self) -> User:
        raise unauthorized()

    def from_websocket(self, websocket: WebSocket) -> Optional[User]:
        return None

    def from_authorization(self, header: str) -> User:
        raise unauthorized()
