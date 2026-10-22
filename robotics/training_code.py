from __future__ import annotations

import ast
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


BLOCKED_NAMES = {"eval", "exec", "compile", "__import__", "open"}
BLOCKED_MODULES = {"os", "subprocess", "socket", "ctypes", "shutil", "requests"}


def validate_training_code(content: str) -> Dict[str, Any]:
    errors: List[str] = []
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return {"valid": False, "errors": ["语法错误: %s" % exc.msg], "warnings": [], "execution": "static-validation-only"}
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "train" for node in tree.body):
        errors.append("训练代码必须提供顶层 train(env, seed) 函数")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_MODULES:
                    errors.append("禁止导入模块: " + alias.name)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in BLOCKED_MODULES:
            errors.append("禁止导入模块: " + (node.module or ""))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
            errors.append("禁止动态或宿主机调用: " + node.func.id)
    return {"valid": not errors, "errors": errors, "warnings": ["代码只做静态校验，不会在 API 进程中执行"], "execution": "static-validation-only"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingCodeRepository:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_default()

    def _path(self, code_id: str) -> Path:
        if not code_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in code_id):
            raise ValueError("无效 code_id")
        return self.root / (code_id + ".json")

    def _ensure_default(self) -> None:
        path = self._path("training_policy_baseline")
        if not path.exists():
            path.write_text(json.dumps({"schema_version": "1.0", "code_id": "training_policy_baseline", "revision": 1, "name": "Joint policy baseline", "language": "python", "content": "def train(env, seed):\n    # Mock-safe policy entrypoint\n    return {\"strategy\": \"joint-p-control\", \"seed\": seed}\n", "execution": "static-validation-only", "scene_ref": {"scene_id": "scene_mock_reach_lab", "revision": 1}, "safety": {"real_robot_control_enabled": False, "sim2real": "blocked"}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def list(self) -> List[Dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.root.glob("*.json"))]

    def get(self, code_id: str) -> Dict[str, Any]:
        path = self._path(code_id)
        if not path.exists():
            raise KeyError(code_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, document: Dict[str, Any]) -> Dict[str, Any]:
        validation = validate_training_code(document.get("content", ""))
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        saved = deepcopy(document)
        code_id = saved.get("code_id") or "training_policy_custom"
        saved["code_id"] = code_id
        path = self._path(code_id)
        saved["revision"] = (self.get(code_id).get("revision", 0) if path.exists() else 0) + 1
        saved["updated_at"] = _now()
        saved["execution"] = "static-validation-only"
        saved["safety"] = {"real_robot_control_enabled": False, "sim2real": "blocked"}
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return saved