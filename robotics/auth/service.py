"""User creation and authentication application service."""

from uuid import uuid4
from typing import Optional

from .models import User, utc_now
from .passwords import PasswordHasher
from .repository import PostgreSQLUserRepository, UserNotFound
from .tokens import JWTAccessTokenService


class AuthenticationFailed(ValueError):
    pass


class UserDisabled(AuthenticationFailed):
    pass


class AuthenticationService:
    def __init__(
        self,
        repository: PostgreSQLUserRepository,
        password_hasher: PasswordHasher,
        tokens: Optional[JWTAccessTokenService] = None,
    ) -> None:
        self.repository = repository
        self.password_hasher = password_hasher
        self.tokens = tokens
        self._dummy_hash = password_hasher.hash("timing-only-password")

    def create_user(self, username: str, display_name: str, password: str) -> User:
        normalized = _normalize_username(username)
        shown_name = display_name.strip()
        if not shown_name:
            raise ValueError("display_name must not be empty")
        now = utc_now()
        return self.repository.create(
            User(
                user_id=str(uuid4()),
                username=normalized,
                display_name=shown_name,
                password_hash=self.password_hasher.hash(password),
                enabled=True,
                created_at=now,
                updated_at=now,
            )
        )

    def authenticate(self, username: str, password: str) -> User:
        normalized = _normalize_username(username)
        user = self.repository.find_by_username(normalized)
        candidate_hash = user.password_hash if user is not None else self._dummy_hash
        verified = self.password_hasher.verify(candidate_hash, password)
        if user is None or not verified:
            raise AuthenticationFailed("invalid username or password")
        if not user.enabled:
            raise UserDisabled("user is disabled")
        return user

    def change_password(
        self, username: str, current_password: str, new_password: str
    ) -> User:
        user = self.authenticate(username, current_password)
        if not isinstance(new_password, str) or len(new_password) < 12:
            raise ValueError("new password must contain at least 12 characters")
        return self.repository.update_password_hash(
            user.user_id, self.password_hasher.hash(new_password)
        )

    def issue_access_token(self, user: User) -> str:
        if self.tokens is None:
            raise RuntimeError("JWT token service is not configured")
        if not user.enabled:
            raise UserDisabled("user is disabled")
        return self.tokens.issue(user.user_id)

    def resolve_access_token(self, token: str) -> User:
        if self.tokens is None:
            raise RuntimeError("JWT token service is not configured")
        claims = self.tokens.decode(token)
        try:
            user = self.repository.get_by_id(claims["sub"])
        except UserNotFound as exc:
            raise AuthenticationFailed("invalid access token") from exc
        if not user.enabled:
            raise UserDisabled("user is disabled")
        return user


def _normalize_username(username: str) -> str:
    if not isinstance(username, str):
        raise ValueError("username must be a string")
    value = username.strip()
    if not value or len(value) > 255:
        raise ValueError("username must contain 1-255 characters")
    return value
