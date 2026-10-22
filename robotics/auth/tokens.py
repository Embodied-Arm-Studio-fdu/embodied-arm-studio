"""Short-lived signed JWT access tokens."""

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from uuid import uuid4

import jwt


class JWTConfigurationError(RuntimeError):
    pass


class JWTTokenError(ValueError):
    pass


@dataclass(frozen=True)
class JWTConfiguration:
    secret: str
    issuer: str
    audience: str
    expires_seconds: int = 900
    algorithm: str = "HS256"

    def __post_init__(self) -> None:
        if not isinstance(self.secret, str) or len(self.secret.encode("utf-8")) < 32:
            raise JWTConfigurationError("USER_JWT_SECRET must contain at least 32 bytes")
        if not self.issuer or not self.audience:
            raise JWTConfigurationError("JWT issuer and audience are required")
        if self.expires_seconds < 60 or self.expires_seconds > 3600:
            raise JWTConfigurationError("JWT access token lifetime must be 60-3600 seconds")
        if self.algorithm != "HS256":
            raise JWTConfigurationError("only HS256 is supported")


class JWTAccessTokenService:
    def __init__(self, config: JWTConfiguration) -> None:
        self.config = config

    def issue(self, subject: str) -> str:
        now = datetime.now(timezone.utc)
        claims = {
            "sub": subject,
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(seconds=self.config.expires_seconds),
            "jti": str(uuid4()),
        }
        return jwt.encode(claims, self.config.secret, algorithm=self.config.algorithm)

    def decode(self, token: str) -> Dict[str, Any]:
        _require_canonical_compact_jwt(token)
        try:
            claims = jwt.decode(
                token,
                self.config.secret,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={
                    "require": ["sub", "iss", "aud", "iat", "nbf", "exp", "jti"],
                    "strict_aud": True,
                },
            )
        except jwt.PyJWTError as exc:
            raise JWTTokenError("invalid access token") from exc
        if claims.get("iss") != self.config.issuer:
            raise JWTTokenError("invalid access token")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            raise JWTTokenError("invalid access token")
        return claims


def _require_canonical_compact_jwt(token: object) -> None:
    """Reject alternate base64url spellings of the same signed bytes."""
    if not isinstance(token, str):
        raise JWTTokenError("invalid access token")
    segments = token.split(".")
    if len(segments) != 3 or any(not segment for segment in segments):
        raise JWTTokenError("invalid access token")
    try:
        for segment in segments:
            padding = "=" * (-len(segment) % 4)
            decoded = base64.b64decode(
                segment + padding, altchars=b"-_", validate=True
            )
            canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
            if segment != canonical:
                raise JWTTokenError("invalid access token")
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise JWTTokenError("invalid access token") from exc
