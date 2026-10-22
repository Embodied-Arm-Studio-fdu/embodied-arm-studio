"""User-owned reward drafts and read-only training source inventory."""

import hashlib
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from simulation.src.simulation_core.reward_extension import (
    PROTECTED_TERMS, REWARD_EXTENSION_SCHEMA, TASK_TERMS, RewardExpression,
    starter_source,
)


def preflight(task_id, source):
    try:
        expression = RewardExpression(task_id, source)
        samples = []
        for magnitude in (0.0, 0.001, 0.1, 1.0, -0.1):
            terms = {name: magnitude for name in TASK_TERMS[task_id]}
            terms.update({"success_bonus": 1.0 if magnitude == 1.0 else 0.0, "joint_limit_penalty": -0.01})
            samples.append({"input": terms, "output": expression.evaluate(terms)})
        return {
            "valid": True, "sha256": expression.digest,
            "extension": {"schema": REWARD_EXTENSION_SCHEMA, "source": source, "sha256": expression.digest},
            "samples": samples, "errors": [],
            "warnings": ["静态契约与样例计算已通过；尚未执行物理仿真，也不保证训练成功率。"],
        }
    except (ValueError, TypeError, RecursionError) as exc:
        return {"valid": False, "errors": [{"code": "reward_contract_error", "message": str(exc)}], "warnings": [], "samples": []}


class ExperimentWorkbenchService:
    def __init__(self, root: Path, source_root: Path):
        self.root = root
        self.source_root = source_root.resolve()

    def _folder(self, user_id, task_id):
        if task_id not in TASK_TERMS:
            raise ValueError("当前仅支持 Push / Reach / Place 奖励实验")
        identifier = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        folder = self.root / identifier / task_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def document(self, user_id, task_id):
        folder = self._folder(user_id, task_id)
        latest = folder / "current.json"
        current = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else None
        history = sorted(
            [json.loads(path.read_text(encoding="utf-8")) for path in folder.glob("revision-*.json")],
            key=lambda item: item["created_at"], reverse=True,
        )
        return {
            "task_id": task_id, "starter": starter_source(task_id),
            "source": current["source"] if current else starter_source(task_id),
            "revision_id": current["revision_id"] if current else "starter",
            "editable_terms": TASK_TERMS[task_id], "protected_terms": PROTECTED_TERMS,
            "history": [{key: item[key] for key in ("revision_id", "sha256", "created_at")} for item in history],
        }

    def save(self, user_id, task_id, source, base_revision):
        folder = self._folder(user_id, task_id)
        with (folder / ".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return self._save_locked(user_id, task_id, source, base_revision, folder)

    def _save_locked(self, user_id, task_id, source, base_revision, folder):
        current = self.document(user_id, task_id)
        if current["revision_id"] != base_revision:
            raise RuntimeError("草稿已在其他窗口更新，请重新加载后对比。")
        if not isinstance(source, str) or len(source.encode("utf-8")) > 8192:
            raise ValueError("奖励草稿不得超过 8 KiB")
        revision = {
            "revision_id": "revision-" + uuid4().hex,
            "source": source, "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        serialized = json.dumps(revision, ensure_ascii=False, indent=2)
        (folder / (revision["revision_id"] + ".json")).write_text(serialized, encoding="utf-8")
        temporary = folder / (revision["revision_id"] + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(folder / "current.json")
        return self.document(user_id, task_id)

    def restore(self, user_id, task_id, revision_id, base_revision):
        folder = self._folder(user_id, task_id)
        if revision_id == "starter":
            source = starter_source(task_id)
        else:
            if not revision_id.startswith("revision-") or len(revision_id) != 41 or any(character not in "0123456789abcdef" for character in revision_id[9:]):
                raise ValueError("无效版本")
            path = folder / (revision_id + ".json")
            if not path.is_file():
                raise ValueError("版本不存在")
            source = json.loads(path.read_text(encoding="utf-8"))["source"]
        return self.save(user_id, task_id, source, base_revision)

    def _source_inventory(self):
        # Browse the entire shipped Python runtime, including evaluation scripts.
        roots = (("", self.source_root), ("scripts/", self.source_root.parents[1] / "scripts"))
        return {
            prefix + path.relative_to(root).as_posix(): path
            for prefix, root in roots for path in root.rglob("*.py")
            if "__pycache__" not in path.parts and not path.is_symlink()
            and root.resolve() in path.resolve().parents
        }

    def sources(self):
        return sorted(self._source_inventory())

    def source(self, path):
        target = self._source_inventory().get(path)
        if target is None:
            raise ValueError("文件不在训练源码目录")
        content = target.read_text(encoding="utf-8")
        return {"path": path, "content": content, "read_only": True, "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
