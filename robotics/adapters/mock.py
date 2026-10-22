from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from uuid import uuid4

from robotics.contracts import EnvironmentInfo, RunEvent, RunRecord, RunRequest, RunStatus
from robotics.runs import (
    ExecutionMode,
    InMemoryRunRepository,
    RunEvent as CanonicalRunEvent,
    RunEventType,
    RunKind,
    RunNotFound,
    RunRepository,
    RunService,
    RunStatus as CanonicalRunStatus,
    legacy_request_to_create_command,
    to_legacy_run_event,
    to_legacy_run_record,
)
from robotics.strategies import STRATEGIES


class MockSimulatorAdapter:
    name = "mock"

    def __init__(
        self,
        storage_root: Path,
        tick_seconds: float = 0.015,
        run_service: Optional[RunService] = None,
        repository: Optional[RunRepository] = None,
        run_id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        if run_service is not None and repository is not None:
            raise ValueError("run_service 与 repository 不能同时注入")
        self.storage_root = storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.tick_seconds = tick_seconds
        if run_service is None:
            repository = repository or InMemoryRunRepository()
            run_service = RunService(
                repository,
                run_id_factory=run_id_factory or self._new_run_id,
            )
        self.run_service = run_service
        self.repository = run_service.repository
        self._queues: Dict[str, "queue.Queue[RunEvent]"] = {}
        self._stops: Dict[str, threading.Event] = {}
        self._finished: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def health(self) -> Dict[str, Any]:
        return {"status": "healthy", "adapter": self.name, "gpu_required": False}

    def list_environments(self) -> List[EnvironmentInfo]:
        common = {
            "simulator": self.name,
            "robot": "Studio Arm M3",
            "capabilities": ["play", "train", "metrics", "export", "joint-control"],
            "observation_space": {
                "joint_positions": {"type": "vector", "shape": [3], "unit": "rad"},
                "target": {"type": "vector", "shape": [3], "unit": "normalized"},
            },
            "action_space": {
                "joint_delta": {"type": "vector", "shape": [3], "range": [-1.0, 1.0]},
            },
        }
        return [
            EnvironmentInfo(
                id="mock-reach",
                name="桌面三关节机械臂",
                description="无需 GPU 的确定性机械臂到达任务，适合观察 action scale、reward 和 seed。",
                reward_terms=[{"id": "distance", "description": "目标距离"}, {"id": "action_penalty", "description": "动作幅度惩罚"}],
                terminations=[{"id": "time_limit", "description": "达到最大步数"}],
                evidence_channels=["metrics", "arm"],
                **common,
            ),
            EnvironmentInfo(
                id="mock-pick-place",
                name="三关节抓取放置近似任务",
                description="确定性教学状态机，用几何近似展示抓取、放置、碰撞和终止证据；不提供物理精度。",
                capabilities=common["capabilities"] + ["trajectory", "collision-approximation", "structured-reward"],
                simulator=common["simulator"],
                robot=common["robot"],
                observation_space={
                    **common["observation_space"],
                    "object_xyz": {"type": "vector", "shape": [3], "unit": "m"},
                    "goal_xyz": {"type": "vector", "shape": [3], "unit": "m"},
                    "grasped": {"type": "boolean"},
                },
                action_space=common["action_space"],
                reward_terms=[
                    {"id": "reach", "description": "接近物体"},
                    {"id": "grasp", "description": "进入抓取状态"},
                    {"id": "place", "description": "物体到达放置区"},
                    {"id": "action_penalty", "description": "动作幅度惩罚"},
                ],
                terminations=[
                    {"id": "success", "description": "完成放置"},
                    {"id": "time_limit", "description": "达到最大步数"},
                ],
                evidence_channels=["reward_terms", "termination", "trajectory", "collision"],
            ),
        ]

    def get_environment_schema(self, environment_id: str) -> Dict[str, Any]:
        self._require_environment(environment_id)
        return {
            "type": "object",
            "properties": {
                "action_scale": {"type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.35, "title": "动作缩放"},
                "reward_distance_weight": {"type": "number", "minimum": 0.1, "maximum": 5.0, "default": 1.0, "title": "距离奖励权重"},
                "learning_rate": {"type": "number", "minimum": 0.00001, "maximum": 0.01, "default": 0.0003, "title": "学习率"},
            },
        }

    def validate_config(self, environment_id: str, config: Dict[str, Any]) -> List[str]:
        schema = self.get_environment_schema(environment_id)
        errors = []
        for key, value in config.items():
            if key == "scene_id":
                if not isinstance(value, str) or not value:
                    errors.append("scene_id 必须是非空字符串")
                continue
            field_schema = schema["properties"].get(key)
            if field_schema is None:
                errors.append("未知参数: %s" % key)
                continue
            if not isinstance(value, (int, float)):
                errors.append("%s 必须是数值" % key)
                continue
            if value < field_schema["minimum"] or value > field_schema["maximum"]:
                errors.append("%s 超出范围" % key)
        return errors

    def launch_play(self, request: RunRequest) -> RunRecord:
        return self._launch("play", request)

    def launch_training(self, request: RunRequest) -> RunRecord:
        return self._launch("train", request)

    def stop_run(self, run_id: str) -> RunRecord:
        return to_legacy_run_record(self.stop_canonical_run(run_id))

    def stop_canonical_run(self, run_id: str):
        with self._lock:
            run = self._get_canonical_run(run_id)
            stop_event = self._stops.get(run_id)
            if stop_event is not None and run.status in {
                CanonicalRunStatus.QUEUED,
                CanonicalRunStatus.INITIALIZING,
                CanonicalRunStatus.RUNNING,
            }:
                stop_event.set()
            if run.status == CanonicalRunStatus.QUEUED:
                self._change_status_locked(run_id, CanonicalRunStatus.STOPPED)
            elif run.status == CanonicalRunStatus.RUNNING:
                self._change_status_locked(run_id, CanonicalRunStatus.STOPPING)
            return self._get_canonical_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        run = self._get_canonical_run(run_id)
        if run.status in {
            CanonicalRunStatus.COMPLETED,
            CanonicalRunStatus.FAILED,
            CanonicalRunStatus.STOPPED,
        }:
            finished = self._finished.get(run_id)
            if finished is not None and threading.current_thread().name != run_id:
                finished.wait()
                run = self._get_canonical_run(run_id)
        return to_legacy_run_record(run)

    def list_runs(self) -> List[RunRecord]:
        return [self.get_run(run.run_id) for run in self.run_service.list_runs()]

    def stream_events(self, run_id: str, after_event_id: str = "") -> Iterable[RunEvent]:
        events = self._canonical_events(run_id)
        seen_requested = not after_event_id
        for canonical_event in events:
            event = to_legacy_run_event(canonical_event)
            if event is None:
                continue
            if seen_requested:
                yield event
            elif event.event_id == after_event_id:
                seen_requested = True

    def wait_for_event(self, run_id: str, timeout: float = 1.0) -> RunEvent:
        self._get_canonical_run(run_id)
        return self._queues[run_id].get(timeout=timeout)

    def list_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        run_dir = self.storage_root / run_id
        if not run_dir.exists():
            return []
        return [{"name": path.name, "size": path.stat().st_size} for path in sorted(run_dir.iterdir()) if path.is_file()]

    def export_policy(self, run_id: str, export_format: str, export_gate: Dict[str, Any] | None = None) -> Dict[str, Any]:
        record = self.get_run(run_id)
        if record.status != RunStatus.COMPLETED:
            raise ValueError("只有已完成运行可以导出")
        if export_format not in ("manifest", "onnx", "torchscript"):
            raise ValueError("不支持的导出格式")
        payload = {
            "run_id": run_id,
            "format": export_format,
            "strategy_id": record.strategy_id,
            "joint_order": ["base", "shoulder", "elbow"],
            "control_frequency_hz": 20,
            "joint_limits_rad": [[-2.8, 2.8], [-1.8, 1.8], [-2.4, 2.4]],
            "real_robot_control_enabled": False,
            "safety_checks": ["joint_order", "normalization", "watchdog", "e_stop", "operator_approval"],
            "export_gate": export_gate or {"eligible": False, "report_id": None, "reasons": ["Evaluation report not provided."]},
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        payload["checksum_sha256"] = hashlib.sha256(encoded).hexdigest()
        path = self.storage_root / run_id / "policy-manifest.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def _launch(self, mode: str, request: RunRequest) -> RunRecord:
        self._require_environment(request.environment_id)
        if request.strategy_id not in STRATEGIES:
            raise ValueError("未知策略: %s" % request.strategy_id)
        errors = self.validate_config(request.environment_id, request.config)
        if errors:
            raise ValueError("; ".join(errors))
        if request.max_steps < 1 or request.max_steps > 2000:
            raise ValueError("max_steps 必须在 1 到 2000 之间")

        command = legacy_request_to_create_command(
            request,
            user_id="mock-adapter",
            project_id="mock-adapter",
            execution_mode=ExecutionMode.MOCK,
            run_kind=RunKind(mode),
        )
        run = self.run_service.create_run(**command)
        with self._lock:
            self._queues[run.run_id] = queue.Queue()
            self._stops[run.run_id] = threading.Event()
            self._finished[run.run_id] = threading.Event()
        thread = threading.Thread(target=self._execute, args=(run.run_id,), daemon=True, name=run.run_id)
        thread.start()
        return self.get_run(run.run_id)

    def _execute(self, run_id: str) -> None:
        run_dir = self.storage_root / run_id
        try:
            with self._lock:
                if self._get_canonical_run(run_id).status == CanonicalRunStatus.STOPPED:
                    self._persist_run(run_id)
                    return
                self._change_status_locked(run_id, CanonicalRunStatus.INITIALIZING)
            run = self._get_canonical_run(run_id)
            snapshot = run.project_config_snapshot
            parameters = run.parameters
            strategy = STRATEGIES[snapshot["strategy_id"]]
            strategy.reset(parameters["seed"])
            positions = [-0.9, 0.45, -0.55]
            target = [0.72, -0.35, 0.62]
            config = parameters["config"]
            action_scale = float(config.get("action_scale", 0.35))
            reward_weight = float(config.get("reward_distance_weight", 1.0))
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = run_dir / "metrics.jsonl"
            with self._lock:
                self._change_status_locked(run_id, CanonicalRunStatus.RUNNING)
                if self._stops[run_id].is_set():
                    self._change_status_locked(run_id, CanonicalRunStatus.STOPPING)

            with metrics_path.open("w", encoding="utf-8") as metrics_file:
                for step in range(parameters["max_steps"]):
                    if self._stops[run_id].is_set():
                        break
                    observation = positions + target
                    action = strategy.act(observation, step)
                    positions = [
                        max(-2.8, min(2.8, position + action_scale * delta))
                        for position, delta in zip(positions, action)
                    ]
                    distance = math.sqrt(sum((goal - position) ** 2 for position, goal in zip(positions, target)))
                    success_rate = max(0.0, min(1.0, 1.0 - distance / 2.5))
                    reward = reward_weight * (1.0 - distance) - 0.03 * sum(abs(value) for value in action)
                    metric_payload = {
                        "step": step,
                        "wall_time_s": round(step * self.tick_seconds, 4),
                        "metrics": {
                            "mean_reward": round(reward, 6),
                            "success_rate": round(success_rate, 6),
                            "distance": round(distance, 6),
                            "policy_loss": round(0.45 * math.exp(-step / 35.0), 6),
                        },
                        "arm": {"joint_positions": [round(value, 6) for value in positions], "target": target},
                    }
                    if snapshot["environment_id"] == "mock-pick-place":
                        metric_payload.update(self._pick_place_evidence(
                            positions=positions,
                            action=action,
                            distance=distance,
                            step=step,
                            max_steps=parameters["max_steps"],
                        ))
                    with self._lock:
                        event = self._record_event_locked(run_id, RunEventType.METRIC, metric_payload)
                    metrics_file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                    if self.tick_seconds:
                        time.sleep(self.tick_seconds)
            with self._lock:
                current = self._get_canonical_run(run_id).status
                if current == CanonicalRunStatus.STOPPING:
                    self._change_status_locked(run_id, CanonicalRunStatus.STOPPED)
                elif current == CanonicalRunStatus.RUNNING:
                    self._change_status_locked(run_id, CanonicalRunStatus.FINALIZING)
                    self._change_status_locked(run_id, CanonicalRunStatus.COMPLETED)
        except Exception as exc:
            with self._lock:
                current = self._get_canonical_run(run_id).status
                if current not in {
                    CanonicalRunStatus.COMPLETED,
                    CanonicalRunStatus.FAILED,
                    CanonicalRunStatus.STOPPED,
                }:
                    self._record_event_locked(
                        run_id,
                        RunEventType.ERROR,
                        {"code": "MOCK_EXECUTION_FAILED", "message": str(exc), "retryable": False},
                    )
                    target = (
                        CanonicalRunStatus.STOPPED
                        if current == CanonicalRunStatus.STOPPING
                        else CanonicalRunStatus.FAILED
                    )
                    self._change_status_locked(run_id, target)
        finally:
            try:
                self._persist_run(run_id)
            finally:
                self._finished[run_id].set()

    def _change_status_locked(self, run_id: str, target: CanonicalRunStatus) -> None:
        current = self._get_canonical_run(run_id).status
        if current == target:
            return
        self._record_event_locked(
            run_id,
            RunEventType.STATE,
            {"previous_status": current.value, "status": target.value},
        )

    def _record_event_locked(
        self, run_id: str, event_type: RunEventType, payload: Dict[str, Any]
    ) -> RunEvent:
        run = self._get_canonical_run(run_id)
        canonical_event = CanonicalRunEvent(
            event_id=str(uuid4()),
            sequence=run.last_sequence + 1,
            type=event_type,
            run_id=run_id,
            occurred_at=datetime.now(timezone.utc),
            payload=payload,
        )
        self.run_service.apply_event(canonical_event)
        legacy_event = to_legacy_run_event(canonical_event)
        if legacy_event is not None:
            legacy_queue = self._queues.get(run_id)
            if legacy_queue is not None:
                legacy_queue.put(legacy_event)
            return legacy_event
        raise RuntimeError("Mock Adapter 不生成 heartbeat")

    def _canonical_events(self, run_id: str) -> List[CanonicalRunEvent]:
        try:
            return self.run_service.list_events(run_id)
        except RunNotFound as exc:
            raise KeyError("未知运行: %s" % run_id) from exc

    def _get_canonical_run(self, run_id: str):
        try:
            return self.run_service.get_run(run_id)
        except RunNotFound as exc:
            raise KeyError("未知运行: %s" % run_id) from exc

    def _persist_run(self, run_id: str) -> None:
        run_dir = self.storage_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record = to_legacy_run_record(self._get_canonical_run(run_id))
        run_path = run_dir / "run.json"
        temporary_path = run_dir / "run.json.tmp"
        temporary_path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(run_path)

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        return "run-" + hashlib.sha1((timestamp + str(uuid4())).encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _require_environment(environment_id: str) -> None:
        if environment_id not in ("mock-reach", "mock-pick-place"):
            raise ValueError("未知环境: %s" % environment_id)

    @staticmethod
    def _pick_place_evidence(
        positions: List[float],
        action: List[float],
        distance: float,
        step: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        end_effector = [
            round(0.35 + 0.12 * math.cos(positions[0]) + 0.08 * math.cos(positions[0] + positions[1]), 6),
            round(0.12 * math.sin(positions[0]) + 0.08 * math.sin(positions[0] + positions[1]), 6),
            round(0.18 + 0.07 * math.sin(positions[1]) + 0.05 * math.sin(positions[1] + positions[2]), 6),
        ]
        progress = max(0.0, min(1.0, 1.0 - distance / 2.5))
        grasped = progress >= 0.55
        placed = progress >= 0.82
        action_penalty = -0.03 * sum(abs(value) for value in action)
        reason = "success" if placed else "time_limit" if step == max_steps - 1 else "running"
        joint_limit_margin = min(2.8 - abs(value) for value in positions)
        collision_detected = end_effector[2] < 0.1 or joint_limit_margin < 0.08
        return {
            "reward_terms": {
                "reach": round(-distance, 6),
                "grasp": 0.5 if grasped else 0.0,
                "place": 1.0 if placed else 0.0,
                "action_penalty": round(action_penalty, 6),
            },
            "termination": {"terminated": placed, "truncated": reason == "time_limit", "reason": reason},
            "trajectory": {
                "end_effector_xyz": end_effector,
                "object_xyz": [0.62, -0.18, 0.06] if grasped else [0.45, 0.1, 0.05],
                "goal_xyz": [0.62, -0.18, 0.06],
                "grasped": grasped,
            },
            "collision": {
                "model": "geometric-approximation",
                "detected": collision_detected,
                "minimum_margin": round(min(end_effector[2] - 0.08, joint_limit_margin), 6),
            },
        }
