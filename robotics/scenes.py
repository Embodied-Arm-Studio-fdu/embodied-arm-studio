from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4


OBJECT_TYPES = {"plane", "box", "cylinder", "sphere", "target", "wall"}
S1_TABLE_BOUNDS = ((0.18, 0.72), (-0.32, 0.32))


def push_geometry_valid(box: Dict[str, Any], target: Dict[str, Any]) -> bool:
    """Geometry consumed by Worker; no collision/IK or reachability claim."""
    try:
        size, position, goal = box["scale"], box["position"], target["position"]
        return (len(size) == len(position) == len(goal) == 3
                and all(type(value) in (int, float) and math.isfinite(value)
                        for value in size + position + goal)
                and all(0 < value <= 0.2 for value in size)
                and abs(position[1] - size[1] / 2) <= 1e-6
                and position[0] - size[0] / 2 >= S1_TABLE_BOUNDS[0][0]
                and position[0] + size[0] / 2 <= S1_TABLE_BOUNDS[0][1]
                and position[2] - size[2] / 2 >= S1_TABLE_BOUNDS[1][0]
                and position[2] + size[2] / 2 <= S1_TABLE_BOUNDS[1][1]
                and S1_TABLE_BOUNDS[0][0] <= goal[0] <= S1_TABLE_BOUNDS[0][1]
                and S1_TABLE_BOUNDS[1][0] <= goal[2] <= S1_TABLE_BOUNDS[1][1])
    except (KeyError, TypeError, ValueError):
        return False


def scene_preflight(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Return conservative training gates without claiming full physics proof."""
    errors = validate_scene(scene)
    warnings: List[str] = []
    checks: Dict[str, bool] = {"schema": not errors}
    if errors:
        return {"valid": False, "trainable_push": False, "trainable_place": False,
                "checks": checks, "errors": errors, "warnings": warnings}
    if scene.get("capability") != "simulation" or scene.get("robot", {}).get("dof") != 6:
        checks["s1_identity"] = False
        errors.append("需要 simulation capability 和六自由度 S1 场景")
    else:
        checks["s1_identity"] = True
    objects = scene.get("objects", []) if isinstance(scene.get("objects"), list) else []
    boxes = [item for item in objects if item.get("type") == "box" and item.get("static") is False]
    targets = [item for item in objects if item.get("type") == "target"]
    checks["push_roles"] = len(boxes) == 1 and len(targets) == 1
    if not checks["push_roles"]:
        errors.append("训练场景需要恰好一个可移动 box 和一个 target")
    in_bounds = True
    for item in boxes + targets:
        position = item.get("position", [])
        if len(position) != 3 or not (S1_TABLE_BOUNDS[0][0] <= float(position[0]) <= S1_TABLE_BOUNDS[0][1]
                                      and S1_TABLE_BOUNDS[1][0] <= float(position[2]) <= S1_TABLE_BOUNDS[1][1]):
            in_bounds = False
    checks["workspace_bounds"] = in_bounds
    if not in_bounds:
        errors.append("box/target 超出已验证的 S1 桌面工作区")
    geometry_valid = len(boxes) == len(targets) == 1 and push_geometry_valid(boxes[0], targets[0])
    checks["push_geometry"] = geometry_valid
    if not geometry_valid:
        errors.append("Push 方块尺寸须为正且不超过 0.2 米，底面须位于桌面，整体须在工作区内")
    has_untranslated_obstacle = any(item.get("type") in {"wall", "cylinder", "sphere"} for item in objects)
    rotated_box = any(any(abs(value) > 1e-6 for value in item["rotation"]) for item in boxes)
    checks["worker_geometry_complete"] = not has_untranslated_obstacle and not rotated_box
    if rotated_box:
        warnings.append("当前 Worker 不转译方块旋转，不能用于正式训练")
    if has_untranslated_obstacle:
        warnings.append("当前 Worker 只转译 box/target；障碍物仍是视觉预览，不能用于正式训练")
    checks["place_candidate"] = checks["s1_identity"] and checks["push_roles"] and checks["workspace_bounds"] and checks["worker_geometry_complete"]
    if not checks["place_candidate"]:
        warnings.append("Place 需要 Worker 侧候选目标和抓取抬起 reset 的运行证据")
    return {
        "valid": not errors,
        "trainable_push": not errors and checks["worker_geometry_complete"],
        "trainable_place": checks["place_candidate"] and not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_scene(scene: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if scene.get("schema_version") != "1.0":
        errors.append("仅支持 scene schema 1.0")
    if not isinstance(scene.get("name"), str) or not scene["name"].strip():
        errors.append("场景需要非空 name")
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        errors.append("场景至少需要机械臂平面和一个对象")
        return errors
    object_ids = set()
    for index, item in enumerate(objects):
        path = "objects[%d]" % index
        if not isinstance(item, dict):
            errors.append(path + " 必须是对象")
            continue
        object_id = item.get("object_id")
        if not isinstance(object_id, str) or not object_id or object_id in object_ids:
            errors.append(path + ".object_id 必须唯一且非空")
        object_ids.add(object_id)
        if item.get("type") not in OBJECT_TYPES:
            errors.append(path + ".type 不是受支持的教学对象")
        for key in ("position", "rotation", "scale"):
            value = item.get(key)
            if not isinstance(value, list) or len(value) != 3 or not all(type(number) in (int, float) and math.isfinite(number) for number in value):
                errors.append(path + "." + key + " 必须是长度为 3 的数值数组")
    if not any(isinstance(item, dict) and item.get("type") == "plane" for item in objects):
        errors.append("场景必须包含一个 plane 对象")
    return errors


def default_scene() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scene_id": "scene_mock_reach_lab",
        "revision": 1,
        "name": "桌面三关节机械臂实验台",
        "capability": "mock",
        "robot": {"id": "studio-arm-m3", "dof": 3, "control_mode": "joint_delta"},
        "objects": [
            {"object_id": "plane", "type": "plane", "label": "实验平面", "position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1], "color": "#c9d1c8", "static": True},
            {"object_id": "target-01", "type": "target", "label": "目标区域", "position": [1.0, 0.02, 0.7], "rotation": [0, 0, 0], "scale": [0.5, 0.08, 0.5], "color": "#e05a3f", "static": True},
        ],
        "observation": ["joint_positions", "target"],
        "action": {"type": "joint_delta", "shape": [3], "range": [-1, 1]},
        "reward": [{"id": "distance", "weight": 1.0}, {"id": "action_penalty", "weight": 0.1}],
        "termination": ["success", "time_limit", "workspace_limit"],
        "safety": {"real_robot_control_enabled": False, "sim2real": "blocked"},
    }


def default_s1_scene() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scene_id": "scene-s1-v2-visual",
        "revision": 1,
        "name": "S1 v2 桌面推物场景",
        "capability": "simulation",
        "robot": {
            "id": "yhrg-s1-v2",
            "dof": 6,
            "gripper_dof": 2,
            "end_effector": "ee_link",
            "control_mode": "joint_position_preview",
            "asset": "yhrg_s1_v2.urdf",
        },
        "objects": [
            {"object_id": "plane", "type": "plane", "label": "实验平面", "position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1], "color": "#c9d1c8", "static": True},
            {"object_id": "push-object", "type": "box", "label": "推动方块", "position": [0.45, 0.05, 0], "rotation": [0, 0, 0], "scale": [0.1, 0.1, 0.1], "color": "#2d6cdf", "static": False},
            {"object_id": "target-01", "type": "target", "label": "目标区域", "position": [0.7, 0.01, 0], "rotation": [0, 0, 0], "scale": [0.18, 0.02, 0.18], "color": "#e05a3f", "static": True},
        ],
        "observation": [],
        "action": {"type": "unavailable", "shape": [], "range": []},
        "reward": [],
        "termination": [],
        "safety": {"real_robot_control_enabled": False, "sim2real": "blocked"},
        "evidence": {
            "visual_asset": "runtime-aligned",
            "collision": "unavailable",
            "inverse_kinematics": "unavailable",
            "reachability": "unavailable",
        },
    }


class SceneRepository:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_default()

    def _path(self, scene_id: str) -> Path:
        if not scene_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in scene_id):
            raise ValueError("无效 scene_id")
        return self.root / (scene_id + ".json")

    def _ensure_default(self) -> None:
        path = self._path("scene_mock_reach_lab")
        if not path.exists():
            path.write_text(json.dumps(default_scene(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        s1_path = self._path("scene-s1-v2-visual")
        if not s1_path.exists():
            s1_path.write_text(json.dumps(default_s1_scene(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def list_scenes(self) -> List[Dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.root.glob("scene-*.json"))] + [self.get("scene_mock_reach_lab")]

    def get(self, scene_id: str) -> Dict[str, Any]:
        path = self._path(scene_id)
        if not path.exists():
            raise KeyError(scene_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, document: Dict[str, Any]) -> Dict[str, Any]:
        errors = validate_scene(document)
        if errors:
            raise ValueError("; ".join(errors))
        saved = deepcopy(document)
        saved.setdefault("scene_id", "scene-" + uuid4().hex[:10])
        previous_revision = self.get(saved["scene_id"]).get("revision", 0) if self._path(saved["scene_id"]).exists() else 0
        saved["revision"] = previous_revision + 1
        saved["created_at"] = _utc_now()
        self._path(saved["scene_id"]).write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return saved
