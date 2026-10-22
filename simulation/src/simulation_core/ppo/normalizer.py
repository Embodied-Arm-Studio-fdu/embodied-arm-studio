"""Versioned running normalization for task-aligned Push observations."""

import math
from typing import Mapping, Optional

import torch

from .policy import OBSERVATION_DIM, OBSERVATION_SLICES


class ObservationNormalizer:
    """Numerically stable Welford statistics with a frozen evaluation mode."""

    def __init__(
        self,
        dimension: int = OBSERVATION_DIM,
        epsilon: float = 1.0e-8,
        variance_floor: float = 1.0e-6,
        clip: float = 10.0,
        frozen: bool = False,
        previous_action_slice: Optional[slice] = None,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("normalizer dimension must be a positive integer")
        if epsilon <= 0 or variance_floor <= 0 or clip <= 0:
            raise ValueError("epsilon, variance_floor, and clip must be positive")
        self.dimension = dimension
        self.epsilon = float(epsilon)
        self.variance_floor = float(variance_floor)
        self.clip = float(clip)
        self.frozen = bool(frozen)
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.m2 = torch.zeros(dimension, dtype=torch.float64)
        # previous_action_xy is already deterministically scaled to [-1, 1].
        self.running_mask = torch.ones(dimension, dtype=torch.bool)
        action_slice = previous_action_slice or (
            OBSERVATION_SLICES["previous_action_xy"]
            if dimension == OBSERVATION_DIM else slice(dimension - 2, dimension)
        )
        self.running_mask[action_slice] = False

    @property
    def variance(self) -> torch.Tensor:
        if self.count < 2:
            return torch.full_like(self.mean, self.variance_floor)
        return (self.m2 / self.count).clamp_min(self.variance_floor)

    def update(self, observation: torch.Tensor) -> None:
        if self.frozen:
            raise RuntimeError("frozen observation normalizer cannot update")
        value = torch.as_tensor(observation).detach().to(device="cpu", dtype=torch.float64)
        if tuple(value.shape) != (self.dimension,) or not torch.isfinite(value).all():
            raise ValueError("observation must be one finite task vector")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def normalize(self, observation: torch.Tensor, update: bool = False) -> torch.Tensor:
        value = torch.as_tensor(observation)
        if tuple(value.shape) != (self.dimension,) or not torch.isfinite(value).all():
            raise ValueError("observation must be one finite task vector")
        if update:
            self.update(value)
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        variance = self.variance.to(device=value.device, dtype=value.dtype)
        mask = self.running_mask.to(value.device)
        normalized = (value - mean) / torch.sqrt(variance + self.epsilon)
        normalized = normalized.clamp(-self.clip, self.clip)
        result = torch.where(mask, normalized, value)
        if not torch.isfinite(result).all():
            raise RuntimeError("normalized observation contains non-finite values")
        return result

    def freeze(self) -> "ObservationNormalizer":
        self.frozen = True
        return self

    def state_dict(self):
        return {
            "count": self.count,
            "mean": self.mean.clone(),
            "m2": self.m2.clone(),
            "variance": self.variance.clone(),
            "epsilon": self.epsilon,
            "variance_floor": self.variance_floor,
            "clip": self.clip,
            "dimension": self.dimension,
            "running_mask": self.running_mask.clone(),
        }

    def load_state_dict(self, state: Mapping, frozen: Optional[bool] = None) -> None:
        required = {
            "count", "mean", "m2", "epsilon", "variance_floor", "clip",
            "dimension", "running_mask",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError("normalizer state is missing: %s" % ", ".join(sorted(missing)))
        if int(state["dimension"]) != self.dimension:
            raise ValueError("normalizer observation dimension mismatch")
        mean = torch.as_tensor(state["mean"], dtype=torch.float64, device="cpu")
        m2 = torch.as_tensor(state["m2"], dtype=torch.float64, device="cpu")
        mask = torch.as_tensor(state["running_mask"], dtype=torch.bool, device="cpu")
        if tuple(mean.shape) != (self.dimension,) or tuple(m2.shape) != (self.dimension,):
            raise ValueError("normalizer mean/m2 shape mismatch")
        if not torch.equal(mask, self.running_mask):
            raise ValueError("normalizer field scaling mask mismatch")
        count = int(state["count"])
        if count < 0 or not torch.isfinite(mean).all() or not torch.isfinite(m2).all():
            raise ValueError("normalizer state is invalid")
        for name in ("epsilon", "variance_floor", "clip"):
            if not math.isclose(float(state[name]), getattr(self, name), rel_tol=0.0, abs_tol=0.0):
                raise ValueError("normalizer %s mismatch" % name)
        self.count = count
        self.mean.copy_(mean)
        self.m2.copy_(m2)
        if frozen is not None:
            self.frozen = bool(frozen)
