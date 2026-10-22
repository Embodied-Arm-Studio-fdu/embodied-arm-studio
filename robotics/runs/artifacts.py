"""Canonical metadata-only Run Artifact models."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import PurePosixPath
import re
from typing import Optional


@dataclass(frozen=True)
class ArtifactRegistration:
    artifact_type: str
    path: str
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    is_primary_model: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_type, str) or not self.artifact_type.strip():
            raise ValueError("artifact_type must be a non-empty string")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("artifact path must be a non-empty string")
        path = PurePosixPath(self.path)
        if (
            "\\" in self.path
            or "\x00" in self.path
            or path.is_absolute()
            or ".." in path.parts
            or self.path == "."
            or str(path) != self.path
        ):
            raise ValueError("artifact path must be normalized Run-relative POSIX")
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("artifact size_bytes must be non-negative")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("artifact sha256 must be lowercase hexadecimal")
        if not isinstance(self.is_primary_model, bool):
            raise ValueError("is_primary_model must be a boolean")


@dataclass(frozen=True)
class RunArtifact:
    artifact_id: str
    run_id: str
    artifact_type: str
    path: str
    size_bytes: Optional[int]
    sha256: Optional[str]
    is_primary_model: bool
    created_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.artifact_id, "artifact_id"),
            (self.run_id, "run_id"),
            (self.artifact_type, "artifact_type"),
            (self.path, "path"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("%s must be a non-empty string" % name)
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValueError("size_bytes must be non-negative")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or len(self.sha256) != 64
        ):
            raise ValueError("sha256 must have 64 characters")
        if not isinstance(self.is_primary_model, bool):
            raise ValueError("is_primary_model must be a boolean")
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("created_at must be timezone-aware UTC")
