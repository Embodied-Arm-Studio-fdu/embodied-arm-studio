from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PORTABLE_KINDS = {
    "environment", "observation", "policy", "action", "transition",
    "reward", "termination", "safety", "metrics", "evaluation", "export", "adapter",
}
CAPABILITIES = {"mock", "external", "planned", "unavailable"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def builtin_mock_reach_pipeline() -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "pipeline_id": "pipeline_mock_reach_baseline",
        "revision": 1,
        "name": "Mock Reach Baseline",
        "capability": "mock",
        "created_at": _utc_now(),
        "environment_ref": {"id": "mock-reach", "revision": 0},
        "nodes": [
            {
                "node_id": "environment", "kind": "environment",
                "implementation": "mock.reach.v1", "capability": "mock", "enabled": True,
                "config": {}, "inputs": [{"name": "action", "type": "joint.position[3]", "required": False}],
                "outputs": [{"name": "state", "type": "observation.map"}],
                "requirements": [], "metadata": {},
            },
            {
                "node_id": "policy", "kind": "policy",
                "implementation": "joint-p-control", "capability": "mock", "enabled": True,
                "config": {"action_scale": 0.35},
                "inputs": [{"name": "state", "type": "observation.map"}],
                "outputs": [{"name": "action", "type": "joint.position[3]"}],
                "requirements": [], "metadata": {},
            },
            {
                "node_id": "metrics", "kind": "metrics",
                "implementation": "mock.metrics.v1", "capability": "mock", "enabled": True,
                "config": {}, "inputs": [{"name": "state", "type": "observation.map"}],
                "outputs": [{"name": "sample", "type": "metric.sample"}],
                "requirements": [], "metadata": {},
            },
        ],
        "edges": [
            {
                "edge_id": "state-policy",
                "source": {"node_id": "environment", "port": "state"},
                "target": {"node_id": "policy", "port": "state"}, "type": "observation.map",
            },
            {
                "edge_id": "action-environment",
                "source": {"node_id": "policy", "port": "action"},
                "target": {"node_id": "environment", "port": "action"},
                "type": "joint.position[3]", "feedback": True,
            },
            {
                "edge_id": "state-metrics",
                "source": {"node_id": "environment", "port": "state"},
                "target": {"node_id": "metrics", "port": "state"}, "type": "observation.map",
            },
        ],
        "reproducibility": {"seed_policy": "required", "deterministic": True},
        "metadata": {"disclaimer": "Approximate teaching dynamics; not a physics simulator."},
    }


def validate_pipeline(document: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    if document.get("schema_version") != "1.0":
        errors.append({"path": "schema_version", "message": "仅支持 Pipeline schema 1.0"})
    if document.get("capability") not in CAPABILITIES:
        errors.append({"path": "capability", "message": "未知 capability"})

    nodes = document.get("nodes")
    edges = document.get("edges")
    if not isinstance(nodes, list) or not nodes:
        errors.append({"path": "nodes", "message": "Pipeline 至少需要一个节点"})
        nodes = []
    if not isinstance(edges, list):
        errors.append({"path": "edges", "message": "edges 必须是数组"})
        edges = []

    node_index: Dict[str, Dict[str, Any]] = {}
    input_ports: Dict[Tuple[str, str], Dict[str, Any]] = {}
    output_ports: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        node_id = node.get("node_id")
        path = "nodes[%d]" % index
        if not isinstance(node_id, str) or not node_id:
            errors.append({"path": path + ".node_id", "message": "节点需要非空 node_id"})
            continue
        if node_id in node_index:
            errors.append({"path": path + ".node_id", "message": "node_id 重复"})
            continue
        node_index[node_id] = node
        if node.get("kind") not in PORTABLE_KINDS:
            errors.append({"path": path + ".kind", "message": "未知节点 kind"})
        if node.get("capability") not in CAPABILITIES:
            errors.append({"path": path + ".capability", "message": "未知 capability"})
        if node.get("enabled", True) and node.get("capability") == "unavailable":
            errors.append({"path": path + ".capability", "message": "启用节点不可标记为 unavailable"})
        for direction, port_index in (("inputs", input_ports), ("outputs", output_ports)):
            for port in node.get(direction, []):
                key = (node_id, port.get("name", ""))
                if not key[1] or key in port_index:
                    errors.append({"path": path + "." + direction, "message": "端口名称为空或重复"})
                else:
                    port_index[key] = port

    connected_inputs = set()
    edge_ids = set()
    for index, edge in enumerate(edges):
        path = "edges[%d]" % index
        edge_id = edge.get("edge_id")
        if not edge_id or edge_id in edge_ids:
            errors.append({"path": path + ".edge_id", "message": "edge_id 为空或重复"})
        edge_ids.add(edge_id)
        source = edge.get("source", {})
        target = edge.get("target", {})
        source_key = (source.get("node_id", ""), source.get("port", ""))
        target_key = (target.get("node_id", ""), target.get("port", ""))
        source_port = output_ports.get(source_key)
        target_port = input_ports.get(target_key)
        if source_port is None:
            errors.append({"path": path + ".source", "message": "源输出端口不存在"})
        if target_port is None:
            errors.append({"path": path + ".target", "message": "目标输入端口不存在"})
        edge_type = edge.get("type")
        if source_port and edge_type != source_port.get("type"):
            errors.append({"path": path + ".type", "message": "边类型与源端口不匹配"})
        if target_port and edge_type != target_port.get("type"):
            errors.append({"path": path + ".type", "message": "边类型与目标端口不匹配"})
        if source.get("node_id") == target.get("node_id") and not edge.get("feedback"):
            errors.append({"path": path, "message": "自环必须显式标记 feedback"})
        connected_inputs.add(target_key)

    for key, port in input_ports.items():
        if port.get("required", True) and key not in connected_inputs:
            errors.append({"path": "nodes.%s.inputs.%s" % key, "message": "必需输入未连接"})

    if document.get("capability") == "mock":
        warnings.append({"path": "capability", "message": "Mock Pipeline 是教学近似，不代表物理精确仿真"})
    return {"valid": not errors, "errors": errors, "warnings": warnings}


class PipelineRepository:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_builtin()

    def _ensure_builtin(self) -> None:
        path = self._revision_path("pipeline_mock_reach_baseline", 1)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(builtin_mock_reach_pipeline(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _revision_path(self, pipeline_id: str, revision: int) -> Path:
        if not pipeline_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in pipeline_id):
            raise ValueError("无效 pipeline_id")
        if revision < 1:
            raise ValueError("revision 必须大于 0")
        return self.root / pipeline_id / ("rev-%d.json" % revision)

    def list_pipelines(self) -> List[Dict[str, Any]]:
        summaries = []
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            revisions = sorted(directory.glob("rev-*.json"))
            if not revisions:
                continue
            document = json.loads(revisions[-1].read_text(encoding="utf-8"))
            summaries.append({
                "pipeline_id": document["pipeline_id"], "revision": document["revision"],
                "name": document["name"], "capability": document["capability"],
                "node_count": len(document.get("nodes", [])), "edge_count": len(document.get("edges", [])),
            })
        return summaries

    def get(self, pipeline_id: str, revision: Optional[int] = None) -> Dict[str, Any]:
        if revision is None:
            directory = self.root / pipeline_id
            revisions = sorted(directory.glob("rev-*.json")) if directory.is_dir() else []
            if not revisions:
                raise KeyError(pipeline_id)
            path = revisions[-1]
        else:
            path = self._revision_path(pipeline_id, revision)
        if not path.exists():
            raise KeyError("%s@%s" % (pipeline_id, revision))
        return json.loads(path.read_text(encoding="utf-8"))

    def validate(self, document: Dict[str, Any]) -> Dict[str, Any]:
        return validate_pipeline(document)

    def save_revision(self, document: Dict[str, Any]) -> Dict[str, Any]:
        validation = validate_pipeline(document)
        if not validation["valid"]:
            raise ValueError(json.dumps(validation["errors"], ensure_ascii=False))
        pipeline_id = document.get("pipeline_id")
        if not isinstance(pipeline_id, str):
            raise ValueError("缺少 pipeline_id")
        try:
            previous = self.get(pipeline_id)
            revision = previous["revision"] + 1
        except KeyError:
            revision = 1
        saved = deepcopy(document)
        saved["revision"] = revision
        saved["created_at"] = _utc_now()
        path = self._revision_path(pipeline_id, revision)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return saved