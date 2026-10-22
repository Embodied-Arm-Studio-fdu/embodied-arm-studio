from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class EnvironmentInfo:
    id: str
    name: str
    description: str
    simulator: str
    robot: str
    capabilities: List[str]
    available: bool = True
    revision: int = 1
    capability: str = "mock"
    observation_space: Dict[str, Any] = field(default_factory=dict)
    action_space: Dict[str, Any] = field(default_factory=dict)
    reward_terms: List[Dict[str, Any]] = field(default_factory=list)
    terminations: List[Dict[str, Any]] = field(default_factory=list)
    evidence_channels: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StrategyInfo:
    id: str
    name: str
    family: str
    description: str
    compatible_environments: List[str]
    requires_gpu: bool = False
    external: bool = False


@dataclass(frozen=True)
class RunRequest:
    environment_id: str
    strategy_id: str
    seed: int = 42
    max_steps: int = 120
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    run_id: str
    mode: str
    environment_id: str
    strategy_id: str
    seed: int
    max_steps: int
    config: Dict[str, Any]
    status: RunStatus = RunStatus.QUEUED
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()