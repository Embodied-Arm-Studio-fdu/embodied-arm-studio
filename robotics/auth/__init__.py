"""Canonical user authentication domain."""

from .models import User
from .passwords import PasswordHasher
from .repository import (
    DuplicateUsername,
    PostgreSQLUserRepository,
    UserNotFound,
    UserRepositoryError,
)
from .service import AuthenticationFailed, AuthenticationService, UserDisabled
from .tokens import JWTConfiguration, JWTConfigurationError, JWTTokenError

__all__ = [
    "AuthenticationFailed",
    "AuthenticationService",
    "DuplicateUsername",
    "JWTConfiguration",
    "JWTConfigurationError",
    "JWTTokenError",
    "PasswordHasher",
    "PostgreSQLUserRepository",
    "User",
    "UserDisabled",
    "UserNotFound",
    "UserRepositoryError",
]
