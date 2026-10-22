"""Run-scoped planning and explicitly applied cleanup for abandoned video partials."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .artifacts import ArtifactPaths


TRAINING_VIDEO_PARTIAL = "video/overview.mp4.partial"


@dataclass(frozen=True)
class PartialCleanupPlan:
    project_id: str
    run_id: str
    targets: Tuple[str, ...]


def plan_failed_run_partials(
    artifacts_root: Path, project_id: str, run_id: str
) -> PartialCleanupPlan:
    """List only abandoned partials inside the exact validated Run root."""
    paths = ArtifactPaths(Path(artifacts_root))
    run_root = paths.run_root(project_id, run_id)
    candidate = run_root / TRAINING_VIDEO_PARTIAL
    targets = ()
    if candidate.is_file() and not candidate.is_symlink():
        resolved_root = run_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
        targets = (TRAINING_VIDEO_PARTIAL,)
    return PartialCleanupPlan(project_id, run_id, targets)


def apply_failed_run_partial_cleanup(
    artifacts_root: Path, plan: PartialCleanupPlan, *, apply: bool = False
) -> Tuple[str, ...]:
    """Delete a previously displayed exact plan only after explicit apply=True."""
    if not isinstance(plan, PartialCleanupPlan):
        raise TypeError("plan must be PartialCleanupPlan")
    if apply is not True:
        return ()
    current = plan_failed_run_partials(
        artifacts_root, plan.project_id, plan.run_id
    )
    if current.targets != plan.targets:
        raise RuntimeError("partial cleanup target set changed; inspect again")
    run_root = ArtifactPaths(Path(artifacts_root)).run_root(
        plan.project_id, plan.run_id
    )
    removed = []
    for relative in plan.targets:
        (run_root / relative).unlink()
        removed.append(relative)
    return tuple(removed)
