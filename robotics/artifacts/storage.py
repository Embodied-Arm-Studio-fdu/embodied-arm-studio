"""Filesystem-independent contract and secure local Artifact mapping."""

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import BinaryIO, Iterator, Protocol


class ArtifactStorageError(RuntimeError):
    """Base class for redacted storage failures."""


class ArtifactStorageConfigurationError(ArtifactStorageError):
    """Raised when the local storage root is unsafe or unavailable."""


class ArtifactPathError(ArtifactStorageError):
    """Raised when a stored relative path is unsafe."""


class ArtifactStorageNotFound(ArtifactStorageError):
    """Raised when a registered path is not an available regular file."""


@dataclass
class ArtifactReadHandle:
    """An already-open regular file, consumed in bounded chunks."""

    file: BinaryIO
    size_bytes: int
    chunk_size: int = 64 * 1024

    def iter_chunks(self) -> Iterator[bytes]:
        try:
            while True:
                chunk = self.file.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            self.file.close()

    def close(self) -> None:
        self.file.close()


class ArtifactStorage(Protocol):
    """Read-only Artifact content boundary."""

    def open(
        self, project_id: str, run_id: str, artifact_path: str
    ) -> ArtifactReadHandle:
        ...


class LocalArtifactStorage:
    """Map registered POSIX paths below one explicitly configured root."""

    _IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")

    def __init__(self, root: Path, chunk_size: int = 64 * 1024) -> None:
        if not isinstance(root, Path):
            root = Path(root)
        if not root.is_absolute() or root == Path("/"):
            raise ArtifactStorageConfigurationError(
                "Artifact storage root must be an explicit safe absolute path"
            )
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ArtifactStorageConfigurationError(
                "Artifact storage chunk size must be positive"
            )
        try:
            resolved = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ArtifactStorageConfigurationError(
                "Artifact storage root is unavailable"
            ) from exc
        if not resolved.is_dir() or resolved == Path("/"):
            raise ArtifactStorageConfigurationError(
                "Artifact storage root must be a directory"
            )
        self._root = resolved
        self._chunk_size = chunk_size

    def open(
        self, project_id: str, run_id: str, artifact_path: str
    ) -> ArtifactReadHandle:
        self._validate_identifier(project_id)
        self._validate_identifier(run_id)
        parts = self._validate_artifact_path(artifact_path)
        try:
            run_directory = (self._root / project_id / run_id).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ArtifactStorageNotFound("Artifact file is unavailable") from exc
        if (
            not self._contains(self._root, run_directory)
            or not run_directory.is_dir()
        ):
            raise ArtifactPathError("Artifact path is unsafe")
        try:
            resolved = run_directory.joinpath(*parts).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ArtifactStorageNotFound("Artifact file is unavailable") from exc
        if not self._contains(run_directory, resolved):
            raise ArtifactPathError("Artifact path is unsafe")
        try:
            handle = resolved.open("rb")
            opened_stat = os.fstat(handle.fileno())
        except (OSError, RuntimeError) as exc:
            raise ArtifactStorageNotFound("Artifact file is unavailable") from exc
        if not stat.S_ISREG(opened_stat.st_mode):
            handle.close()
            raise ArtifactStorageNotFound("Artifact file is unavailable")
        return ArtifactReadHandle(
            file=handle,
            size_bytes=opened_stat.st_size,
            chunk_size=self._chunk_size,
        )

    @classmethod
    def _validate_identifier(cls, value: str) -> None:
        if (
            not isinstance(value, str)
            or value in {".", ".."}
            or cls._IDENTIFIER.fullmatch(value) is None
        ):
            raise ArtifactPathError("Artifact path is unsafe")

    @staticmethod
    def _validate_artifact_path(value: str):
        if (
            not isinstance(value, str)
            or not value
            or "\\" in value
            or "\x00" in value
        ):
            raise ArtifactPathError("Artifact path is unsafe")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ArtifactPathError("Artifact path is unsafe")
        raw_parts = value.split("/")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in raw_parts)
            or str(path) != value
        ):
            raise ArtifactPathError("Artifact path is unsafe")
        return tuple(raw_parts)

    @staticmethod
    def _contains(parent: Path, child: Path) -> bool:
        return child == parent or parent in child.parents
