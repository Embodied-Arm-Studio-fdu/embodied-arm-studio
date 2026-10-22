"""Argon2id password hashing boundary."""

from argon2 import PasswordHasher as Argon2PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from argon2.low_level import Type


class PasswordHasher:
    def __init__(self) -> None:
        self._hasher = Argon2PasswordHasher(type=Type.ID)

    def hash(self, password: str) -> str:
        if not isinstance(password, str) or not password:
            raise ValueError("password must not be empty")
        return self._hasher.hash(password)

    def verify(self, password_hash: str, password: str) -> bool:
        try:
            return self._hasher.verify(password_hash, password)
        except (InvalidHashError, VerificationError, TypeError):
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        try:
            return self._hasher.check_needs_rehash(password_hash)
        except (InvalidHashError, TypeError):
            return True
