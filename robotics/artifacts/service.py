"""Authorization, metadata ownership, and integrity rules for downloads."""

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterator, Optional, Protocol

from robotics.artifacts.storage import (
    ArtifactPathError,
    ArtifactStorage,
    ArtifactStorageError,
    ArtifactStorageNotFound,
)
from robotics.runs.artifacts import RunArtifact
from robotics.runs.repository import RepositoryError, RunNotFound
from robotics.run_access import RunAccessNotFound, RunAccessUnavailable


class ArtifactNotFound(LookupError):
    """Unknown, inaccessible, unregistered, or unavailable Artifact."""


class ArtifactIntegrityError(RuntimeError):
    """Registered integrity metadata does not match the stored file."""


class ArtifactServiceUnavailable(RuntimeError):
    """Artifact metadata or storage is temporarily unavailable."""


class ArtifactMetadataStore(Protocol):
    def get_artifact(self, run_id: str, artifact_id: str) -> RunArtifact:
        ...


@dataclass(frozen=True)
class ArtifactDownload:
    chunks: Iterator[bytes]
    media_type: str
    headers: Dict[str, str]


class ArtifactService:
    """Open only PostgreSQL-registered files owned by the trusted user."""

    _SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")

    def __init__(
        self,
        run_service: Any,
        metadata_store: ArtifactMetadataStore,
        storage: ArtifactStorage,
        run_access_service: Optional[Any] = None,
    ) -> None:
        self._run_service = run_service
        self._metadata_store = metadata_store
        self._storage = storage
        self._run_access_service = run_access_service

    def prepare_download(
        self, run_id: str, artifact_id: str, trusted_user_id: str
    ) -> ArtifactDownload:
        if self._run_access_service is None:
            try:
                run = self._run_service.get_run(run_id)
            except RunNotFound as exc:
                raise ArtifactNotFound("Artifact not found") from exc
            except RepositoryError as exc:
                raise ArtifactServiceUnavailable(
                    "Artifact service is temporarily unavailable"
                ) from exc
            if run.user_id != trusted_user_id:
                raise ArtifactNotFound("Artifact not found")
        else:
            try:
                run = self._run_access_service.read(run_id, trusted_user_id)
            except Exception as exc:
                if isinstance(exc, RunAccessNotFound):
                    raise ArtifactNotFound("Artifact not found") from exc
                if isinstance(exc, RunAccessUnavailable):
                    raise ArtifactServiceUnavailable(
                        "Artifact service is temporarily unavailable"
                    ) from exc
                raise
        try:
            artifact = self._metadata_store.get_artifact(run_id, artifact_id)
        except RunNotFound as exc:
            raise ArtifactNotFound("Artifact not found") from exc
        except RepositoryError as exc:
            raise ArtifactServiceUnavailable(
                "Artifact service is temporarily unavailable"
            ) from exc
        if artifact.run_id != run_id:
            raise ArtifactNotFound("Artifact not found")
        if artifact.sha256 is not None and re.fullmatch(
            r"[0-9a-f]{64}", artifact.sha256
        ) is None:
            raise ArtifactIntegrityError("Artifact integrity check failed")
        try:
            handle = self._storage.open(run.project_id, run_id, artifact.path)
        except (ArtifactPathError, ArtifactStorageNotFound) as exc:
            raise ArtifactNotFound("Artifact not found") from exc
        except ArtifactStorageError as exc:
            raise ArtifactServiceUnavailable(
                "Artifact service is temporarily unavailable"
            ) from exc
        if (
            artifact.size_bytes is not None
            and artifact.size_bytes != handle.size_bytes
        ):
            handle.close()
            raise ArtifactIntegrityError("Artifact integrity check failed")
        filename = self._download_filename(artifact.path)
        headers = {
            "Content-Disposition": 'attachment; filename="%s"' % filename,
            "Content-Length": str(handle.size_bytes),
            "X-Content-Type-Options": "nosniff",
        }
        if artifact.sha256 is not None:
            headers["ETag"] = '"%s"' % artifact.sha256
        return ArtifactDownload(
            chunks=handle.iter_chunks(),
            media_type=self._media_type(artifact),
            headers=headers,
        )

    @classmethod
    def _download_filename(cls, path: str) -> str:
        name = path.rsplit("/", 1)[-1]
        safe = cls._SAFE_FILENAME.sub("_", name)
        if not safe or safe in {".", ".."}:
            safe = "artifact.bin"
        return safe[:180]

    @staticmethod
    def _media_type(artifact: RunArtifact) -> str:
        artifact_type = artifact.artifact_type.lower()
        if artifact_type in {"training_config", "manifest", "evaluation_summary"}:
            return "application/json"
        if artifact_type in {
            "training_metrics", "training_episode_metrics", "evaluation_episodes",
        }:
            return "application/x-ndjson"
        if artifact_type in {"stdout_log", "stderr_log", "task_log"}:
            return "text/plain"
        if artifact_type in {"training_video", "evaluation_video"}:
            return "video/mp4"
        return "application/octet-stream"
