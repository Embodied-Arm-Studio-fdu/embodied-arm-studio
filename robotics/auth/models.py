"""Canonical persisted user model."""

from dataclasses import dataclass
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class User:
    user_id: str
    username: str
    display_name: str
    password_hash: str
    enabled: bool
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        for name in ("created_at", "updated_at"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("%s must be timezone-aware" % name)

    def public_dict(self):
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
