from __future__ import annotations

import json
import re
import shutil
from difflib import unified_diff
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class PatchProposal:
    patch_id: str
    project_id: str
    path: str
    diff: str
    risk: str
    status: str
    created_at: str
    snapshot_id: Optional[str] = None


class WorkspaceService:
    allowed_names = {"controller.json"}
    editable_extensions = {".cpp", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
    protected_names = {".studio", ".git"}
    max_file_bytes = 512 * 1024
    high_risk_terms = ("joint_limit", "termination", "reward", "action_scale", "real_robot")

    def __init__(self, root: Path, template_root: Path) -> None:
        self.root = root.resolve()
        self.template_root = template_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.proposals: Dict[str, PatchProposal] = {}
        self.audit_path = self.root / "audit.jsonl"

    def create_project(self, project_id: Optional[str] = None) -> Dict[str, str]:
        identifier = self._identifier(project_id or "project-" + uuid4().hex[:10])
        project_root = self.root / identifier
        workspace = project_root / "workspace"
        snapshots = project_root / "snapshots"
        if not workspace.exists():
            workspace.mkdir(parents=True)
            snapshots.mkdir(parents=True)
            shutil.copy2(str(self.template_root / "controller.json"), str(workspace / "controller.json"))
        self._audit("create_project", {"project_id": identifier})
        return {"project_id": identifier}

    def list_projects(self) -> List[Dict[str, Any]]:
        projects = []
        for project_root in sorted(self.root.iterdir()):
            workspace = project_root / "workspace"
            if project_root.is_dir() and not project_root.is_symlink() and workspace.is_dir():
                projects.append({
                    "project_id": project_root.name,
                    "file_count": len(self.list_files(project_root.name)),
                    "workspace_status": "clean",
                })
        return projects

    def list_files(self, project_id: str) -> List[str]:
        workspace = self._project_root(project_id) / "workspace"
        return sorted(
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*")
            if path.is_file() and not path.is_symlink()
        )

    def file_tree(self, project_id: str) -> List[Dict[str, Any]]:
        workspace = self._project_root(project_id) / "workspace"

        def children(directory: Path) -> List[Dict[str, Any]]:
            nodes = []
            for path in sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
                if path.is_symlink() or path.name in self.protected_names:
                    continue
                relative = path.relative_to(workspace).as_posix()
                node: Dict[str, Any] = {"name": path.name, "path": relative, "type": "directory" if path.is_dir() else "file"}
                if path.is_dir():
                    node["children"] = children(path)
                else:
                    node["size"] = path.stat().st_size
                    node["read_only"] = False
                nodes.append(node)
            return nodes

        return children(workspace)

    def read_file(self, project_id: str, path: str) -> str:
        target = self._workspace_path(project_id, path, require_file=True)
        self._audit("read_file", {"project_id": project_id, "path": path})
        return target.read_text(encoding="utf-8")

    def write_file(self, project_id: str, path: str, content: str) -> Dict[str, Any]:
        target = self._workspace_path(project_id, path)
        if target.suffix.lower() not in self.editable_extensions:
            raise ValueError("文件类型不在工作区 allowlist")
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ValueError("文件超过 512 KiB 限制")
        target.parent.mkdir(parents=True, exist_ok=True)
        previous = target.read_text(encoding="utf-8") if target.is_file() else None
        target.write_bytes(encoded)
        if previous != content:
            self._record_version(project_id, path, content)
        self._audit("write_file", {"project_id": project_id, "path": path, "size": len(encoded)})
        return {"path": path, "size": len(encoded)}

    def file_history(self, project_id: str, path: str) -> List[Dict[str, str]]:
        self._workspace_path(project_id, path, require_file=True)
        history = self._history_path(project_id, path)
        if not history.is_dir():
            return []
        records = [json.loads(version.read_text(encoding="utf-8")) for version in history.glob("*.json")]
        return sorted(records, key=lambda record: record["created_at"], reverse=True)

    def restore_version(self, project_id: str, path: str, version_id: str) -> Dict[str, Any]:
        version = self._history_path(project_id, path) / f"{self._identifier(version_id)}.json"
        if not version.is_file() or version.is_symlink():
            raise ValueError("未知文件版本")
        record = json.loads(version.read_text(encoding="utf-8"))
        result = self.write_file(project_id, path, record["content"])
        self._audit("restore_version", {"project_id": project_id, "path": path, "version_id": version_id})
        return result

    def starter_diff(self, project_id: str, path: str) -> Dict[str, str]:
        current = self.read_file(project_id, path)
        starter = self.template_root / path
        baseline = starter.read_text(encoding="utf-8") if starter.is_file() and not starter.is_symlink() else ""
        diff = "".join(unified_diff(baseline.splitlines(keepends=True), current.splitlines(keepends=True), fromfile=f"starter/{path}", tofile=f"workspace/{path}"))
        return {"path": path, "baseline": baseline, "current": current, "diff": diff, "status": "modified" if diff else "clean"}

    def restore_starter(self, project_id: str) -> Dict[str, str]:
        snapshot_id = self._snapshot(project_id)
        workspace = self._project_root(project_id) / "workspace"
        shutil.rmtree(workspace)
        shutil.copytree(self.template_root, workspace)
        self._audit("restore_starter", {"project_id": project_id, "snapshot_id": snapshot_id})
        return {"status": "restored", "snapshot_id": snapshot_id}

    def create_directory(self, project_id: str, path: str) -> Dict[str, str]:
        target = self._workspace_path(project_id, path)
        target.mkdir(parents=True, exist_ok=False)
        self._audit("create_directory", {"project_id": project_id, "path": path})
        return {"path": path, "type": "directory"}

    def move(self, project_id: str, source: str, destination: str) -> Dict[str, str]:
        source_path = self._workspace_path(project_id, source)
        destination_path = self._workspace_path(project_id, destination)
        if not source_path.exists() or destination_path.exists():
            raise ValueError("源路径不存在或目标路径已存在")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.rename(destination_path)
        self._audit("move", {"project_id": project_id, "source": source, "destination": destination})
        return {"path": destination}

    def delete(self, project_id: str, path: str) -> None:
        target = self._workspace_path(project_id, path)
        if not target.exists():
            raise ValueError("路径不存在")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        self._audit("delete", {"project_id": project_id, "path": path})

    def search(self, project_id: str, query: str) -> List[Dict[str, Any]]:
        if not query.strip() or len(query) > 200:
            raise ValueError("搜索词长度必须为 1 到 200")
        results = []
        for path in self.list_files(project_id):
            target = self._workspace_path(project_id, path, require_file=True)
            if target.stat().st_size > self.max_file_bytes:
                continue
            for line_number, line in enumerate(target.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if query.lower() in line.lower():
                    results.append({"path": path, "line": line_number, "preview": line[:240]})
                    if len(results) >= 100:
                        return results
        return results

    def create_snapshot(self, project_id: str) -> Dict[str, str]:
        snapshot_id = self._snapshot(project_id)
        self._audit("create_snapshot", {"project_id": project_id, "snapshot_id": snapshot_id})
        return {"snapshot_id": snapshot_id}

    def propose_patch(self, project_id: str, path: str, diff: str) -> PatchProposal:
        self._allowed_file(project_id, path)
        old_lines, new_lines = self._parse_diff(path, diff)
        changed_lines = max(len(old_lines), len(new_lines))
        if changed_lines > 80:
            raise ValueError("patch 超过 80 行限制")
        current = self.read_file(project_id, path).splitlines()
        if old_lines != current:
            raise ValueError("patch 基线与当前文件不一致")
        self._validate_content(path, new_lines)
        risk = "high" if any(term in diff for term in self.high_risk_terms) else "normal"
        proposal = PatchProposal(
            patch_id="patch-" + uuid4().hex[:12],
            project_id=project_id,
            path=path,
            diff=diff,
            risk=risk,
            status="pending",
            created_at=_now(),
        )
        self.proposals[proposal.patch_id] = proposal
        self._audit("propose_patch", asdict(proposal))
        return proposal

    def approve(self, patch_id: str, confirm_high_risk: bool = False) -> PatchProposal:
        proposal = self._proposal(patch_id)
        if proposal.status != "pending":
            raise ValueError("patch 已处理")
        if proposal.risk == "high" and not confirm_high_risk:
            raise ValueError("高风险 patch 需要二次确认")
        old_lines, new_lines = self._parse_diff(proposal.path, proposal.diff)
        target = self._allowed_file(proposal.project_id, proposal.path)
        if target.read_text(encoding="utf-8").splitlines() != old_lines:
            raise ValueError("文件在审批前已变化")
        snapshot_id = self._snapshot(proposal.project_id)
        target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        proposal.snapshot_id = snapshot_id
        proposal.status = "applied"
        self._audit("approve_patch", asdict(proposal))
        return proposal

    def reject(self, patch_id: str) -> PatchProposal:
        proposal = self._proposal(patch_id)
        if proposal.status != "pending":
            raise ValueError("patch 已处理")
        proposal.status = "rejected"
        self._audit("reject_patch", asdict(proposal))
        return proposal

    def rollback(self, project_id: str, snapshot_id: str) -> None:
        snapshot = self._project_root(project_id) / "snapshots" / self._identifier(snapshot_id)
        if not snapshot.is_dir() or snapshot.is_symlink():
            raise ValueError("未知快照")
        workspace = self._project_root(project_id) / "workspace"
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(snapshot, workspace)
        self._audit("rollback", {"project_id": project_id, "snapshot_id": snapshot_id})

    def _snapshot(self, project_id: str) -> str:
        snapshot_id = "snapshot-" + uuid4().hex[:12]
        project_root = self._project_root(project_id)
        snapshot = project_root / "snapshots" / snapshot_id
        snapshot.mkdir()
        shutil.copytree(project_root / "workspace", snapshot, dirs_exist_ok=True)
        return snapshot_id

    def _history_path(self, project_id: str, path: str) -> Path:
        safe_path = path.replace("/", "__")
        history = self._project_root(project_id) / "history" / safe_path
        history.mkdir(parents=True, exist_ok=True)
        return history

    def _record_version(self, project_id: str, path: str, content: str) -> None:
        version_id = "version-" + uuid4().hex[:12]
        record = {"version_id": version_id, "created_at": _now(), "content": content}
        (self._history_path(project_id, path) / f"{version_id}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    def _workspace_path(self, project_id: str, path: str, require_file: bool = False) -> Path:
        relative = Path(path)
        if not path or relative.is_absolute() or ".." in relative.parts or any(part in self.protected_names for part in relative.parts):
            raise ValueError("非法工作区路径")
        workspace = (self._project_root(project_id) / "workspace").resolve()
        target = (workspace / relative).resolve()
        if target != workspace and workspace not in target.parents:
            raise ValueError("非法工作区路径")
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != workspace.parent):
            raise ValueError("工作区不允许符号链接")
        if require_file and not target.is_file():
            raise ValueError("文件不存在")
        return target

    def _allowed_file(self, project_id: str, path: str) -> Path:
        if path not in self.allowed_names or Path(path).name != path:
            raise ValueError("文件不在课程 allowlist")
        return self._workspace_path(project_id, path, require_file=True)

    def _project_root(self, project_id: str) -> Path:
        target = self.root / self._identifier(project_id)
        if not target.is_dir() or target.is_symlink():
            raise ValueError("未知项目")
        return target

    @staticmethod
    def _identifier(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            raise ValueError("非法标识符")
        return value

    @staticmethod
    def _parse_diff(path: str, diff: str):
        lines = diff.splitlines()
        if len(lines) < 3 or lines[0] != "--- a/" + path or lines[1] != "+++ b/" + path or not lines[2].startswith("@@ "):
            raise ValueError("只支持目标文件的单 hunk unified diff")
        old_lines = []
        new_lines = []
        for line in lines[3:]:
            if line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith(" "):
                old_lines.append(line[1:])
                new_lines.append(line[1:])
            elif line == "\\ No newline at end of file":
                continue
            else:
                raise ValueError("patch 包含非法行")
        return old_lines, new_lines

    @staticmethod
    def _validate_content(path: str, lines: List[str]) -> None:
        if path.endswith(".json"):
            parsed = json.loads("\n".join(lines))
            if not isinstance(parsed, dict):
                raise ValueError("配置文件必须是 JSON 对象")

    def _proposal(self, patch_id: str) -> PatchProposal:
        try:
            return self.proposals[patch_id]
        except KeyError as exc:
            raise ValueError("未知 patch") from exc

    def _audit(self, action: str, payload: Dict) -> None:
        record = {"timestamp": _now(), "action": action, "payload": payload}
        with self.audit_path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()