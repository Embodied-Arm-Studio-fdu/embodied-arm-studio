from __future__ import annotations

import math
import random
from typing import Dict, List, Protocol, Sequence

from robotics.contracts import StrategyInfo


class StrategyAdapter(Protocol):
    info: StrategyInfo

    def reset(self, seed: int) -> None: ...

    def act(self, observation: Sequence[float], step: int) -> List[float]: ...


class JointPStrategy:
    info = StrategyInfo(
        id="joint-p-control",
        name="关节 P 控制",
        family="classic-control",
        description="观察目标误差如何通过比例增益转化为关节动作。",
        compatible_environments=["mock-reach", "mock-pick-place"],
    )

    def __init__(self, gain: float = 0.18) -> None:
        self.gain = gain

    def reset(self, seed: int) -> None:
        return None

    def act(self, observation: Sequence[float], step: int) -> List[float]:
        joint_positions = observation[:3]
        target = observation[3:6]
        return [self.gain * (goal - position) for position, goal in zip(joint_positions, target)]


class ExplorationStrategy:
    info = StrategyInfo(
        id="policy-gradient-demo",
        name="策略梯度入门",
        family="reinforcement-learning",
        description="用带衰减的探索噪声模拟策略从探索到稳定的训练过程。",
        compatible_environments=["mock-reach", "mock-pick-place"],
    )

    def __init__(self) -> None:
        self.random = random.Random()

    def reset(self, seed: int) -> None:
        self.random.seed(seed)

    def act(self, observation: Sequence[float], step: int) -> List[float]:
        noise_scale = 0.12 * math.exp(-step / 45.0)
        joint_positions = observation[:3]
        target = observation[3:6]
        return [
            0.16 * (goal - position) + self.random.uniform(-noise_scale, noise_scale)
            for position, goal in zip(joint_positions, target)
        ]


STRATEGIES: Dict[str, StrategyAdapter] = {
    JointPStrategy.info.id: JointPStrategy(),
    ExplorationStrategy.info.id: ExplorationStrategy(),
}