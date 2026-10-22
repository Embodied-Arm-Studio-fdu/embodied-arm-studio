"""Safe artifact path calculation without creating files or directories."""

from pathlib import Path


class ArtifactPathError(ValueError):
    pass


def _validate_component(value: str, name: str) -> None:
    if not value or value in (".", ".."):
        raise ArtifactPathError("%s must be a non-empty path component" % name)
    if "\x00" in value or "/" in value or "\\" in value:
        raise ArtifactPathError("%s must not contain path separators" % name)
    if Path(value).is_absolute():
        raise ArtifactPathError("%s must not be absolute" % name)


class ArtifactPaths:
    def __init__(self, artifacts_root: Path) -> None:
        self._root = artifacts_root.resolve()

    def run_root(self, project_id: str, run_id: str) -> Path:
        _validate_component(project_id, "project_id")
        _validate_component(run_id, "run_id")
        result = (self._root / project_id / run_id).resolve()
        try:
            result.relative_to(self._root)
        except ValueError as exc:
            raise ArtifactPathError("artifact path escapes its root") from exc
        return result
