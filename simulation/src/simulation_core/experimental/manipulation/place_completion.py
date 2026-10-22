"""P4 completion curriculum environments and stage checkpoint support."""

import hashlib
from pathlib import Path

import torch

from ...ppo.trainer import PPOTrainer
from .lower_release_env import (
    LOWER_RELEASE_BOOTSTRAP_VERSION, LOWER_RELEASE_TARGET,
    LowerReleaseEnvironment, lower_release_config_digest,
)
from .lower_release_ppo import (
    ExperimentalLowerReleasePPOTrainer, reward_config_digest,
)
from .lower_release_schema import (
    LOWER_RELEASE_REWARD_VERSION, LOWER_RELEASE_SUCCESS_CRITERIA_VERSION,
)
from .place_schema import PLACE_ACTION_SCHEMA_ID, PLACE_OBSERVATION_SCHEMA_ID


RELEASE_ONLY_ACTIVE_ACTION_MASK = (0, 0, 0, 1)
FULL_PLACE_ACTIVE_ACTION_MASK = (1, 1, 1, 1)
P4_2_20K_SHA256 = "58ddf5e91a29ac087d37dac9094dad77a3596f5e32693fe5ad3f8e78b9dcba33"


class ReleaseOnlyEnvironment(LowerReleaseEnvironment):
    """Reset at legal height and expose only the release trigger."""

    active_action_mask = RELEASE_ONLY_ACTIVE_ACTION_MASK

    def _scripted_lower_to_curriculum_height(self):
        stable = 0; snapshot = self.scene.snapshot()
        for _ in range(120):
            clearance = self._bottom_clearance(snapshot)
            dz = max(-.003, min(.003, .008 - clearance))
            requested = (self._target[0], self._target[1], self._target[2] + dz)
            snapshot, command = self._control(requested, self.config.stable_grasp_jaw)
            self._target = self.config.workspace.clamp(requested)[0]
            metrics = self._metrics(snapshot, command.jaw_target)
            clearance = self._bottom_clearance(snapshot)
            valid = (.006 <= clearance <= .010 and metrics["grasp_valid"]
                     and self._finite(snapshot)
                     and max(metrics["forces"].values()) <= self.config.force_abort
                     and not command.workspace_clamped and not command.joint_limit_clamped)
            stable = stable + 1 if valid else 0
            if not self._finite(snapshot): raise RuntimeError("RELEASE_RESET_NON_FINITE")
            if command.workspace_clamped: raise RuntimeError("RELEASE_RESET_WORKSPACE")
            if command.joint_limit_clamped: raise RuntimeError("RELEASE_RESET_JOINT_LIMIT")
            if max(metrics["forces"].values()) > self.config.force_abort:
                raise RuntimeError("RELEASE_RESET_UNSAFE_FORCE")
            if stable >= 5: return snapshot
        raise RuntimeError("RELEASE_RESET_HEIGHT_TIMEOUT")

    def reset(self):
        super().reset()
        snapshot = self._scripted_lower_to_curriculum_height()
        metrics = self._metrics(snapshot); clearance = self._bottom_clearance(snapshot)
        if not (.006 <= clearance <= .010 and metrics["grasp_valid"]
                and self._finite(snapshot)
                and max(metrics["forces"].values()) <= self.config.force_abort):
            raise RuntimeError("RELEASE_ONLY_RESET_VALIDATION_FAILED")
        self._policy_xy = tuple(snapshot["ee_state"][:2])
        self._target = tuple(snapshot["ee_state"][:3])
        self._place_previous_action = (0.0,) * 4
        self._steps = self._settle_steps = self._maximum_settle_steps = 0
        self._start_clearance = self._minimum_clearance = clearance
        self._minimum_bottom_clearance = clearance
        self._maximum_open_fraction = 0.0
        self._release_started = self._release_latched = False
        self._release_trigger_streak = 0; self._actual_jaw_opened = False
        self._detached = self._table_contact = False
        self._inside_target = False; self._descended_to_release_height = True
        self._transport_reached = True
        self._settled_half = False; self._release_height = self._detach_height = None
        self.phase = "LOWER_RELEASE_POLICY"
        self._bootstrap_last = dict(self._bootstrap_last)
        self._bootstrap_last.update({"release_only_reset": True,
                                     "release_only_bottom_clearance": clearance})
        return self._place_observation(snapshot)

    def step(self, action):
        values = tuple(float(value) for value in action)
        if len(values) != 4:
            raise ValueError("ACTION_NON_FINITE_OR_DIMENSION")
        observation, reward, done, info = super().step((0.0, 0.0, 0.0, values[3]))
        info["active_action_mask"] = RELEASE_ONLY_ACTIVE_ACTION_MASK
        info["applied_action"] = (0.0, 0.0, 0.0, max(-1.0, min(1.0, values[3])))
        return observation, reward, done, info


class StagePlaceTrainer(ExperimentalLowerReleasePPOTrainer):
    """Same network/contract with explicit curriculum-stage mask metadata."""

    def __init__(self, environment, *, active_action_mask, stage_name,
                 release_semantics="policy_trigger_latched_open", **kwargs):
        super().__init__(environment, **kwargs)
        self.stage_active_action_mask = tuple(active_action_mask)
        self.stage_name = str(stage_name)
        self.release_semantics = str(release_semantics)
        self.policy.set_active_action_mask(self.stage_active_action_mask)
        self.source_checkpoint = None

    def initialize_from_checkpoint(self, checkpoint_path, expected_sha256=None,
                                   reinitialize_release_row=False):
        path = Path(checkpoint_path); digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("source checkpoint SHA-256 mismatch")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if (payload.get("task_id") != "s1_place" or payload.get("observation_dim") != 32
                or payload.get("action_dim") != 4
                or payload.get("observation_schema_id") != PLACE_OBSERVATION_SCHEMA_ID
                or payload.get("action_schema_id") != PLACE_ACTION_SCHEMA_ID):
            raise ValueError("source checkpoint is not compatible with Place v1")
        self.policy.load_state_dict(payload["policy"])
        self.normalizer.load_state_dict(payload["normalizer_state"], frozen=False)
        if reinitialize_release_row:
            output = self.policy.actor[-1]
            with torch.no_grad():
                output.weight[3].zero_()
                output.bias[3] = -0.25
                self.policy.log_std[3] = -0.50
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.training_steps = self.update_steps = 0
        self.source_checkpoint = {"path": str(path), "sha256": digest,
                                  "source_training_steps": int(payload["training_steps"]),
                                  "release_row_reinitialized": bool(reinitialize_release_row),
                                  "transfer_strategy": (
                                      "preserve normalizer, actor trunk, critic, and dx/dy/dz rows; "
                                      "reinitialize only release-trigger output row"
                                      if reinitialize_release_row else
                                      "preserve all policy rows across curriculum stages")}
        self.parent_checkpoint = payload.get("parent_checkpoint")
        return payload

    def save_checkpoint(self, artifacts_root, project_id, run_id,
                        filename="model_latest.pt"):
        path = PPOTrainer.save_checkpoint(self, artifacts_root, project_id, run_id, filename)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update({
            "observation_schema_id": PLACE_OBSERVATION_SCHEMA_ID,
            "action_schema_id": PLACE_ACTION_SCHEMA_ID,
            "active_action_mask": self.stage_active_action_mask,
            "reward_version": LOWER_RELEASE_REWARD_VERSION,
            "reward_config_digest": reward_config_digest(self.environment.config),
            "success_criteria_version": LOWER_RELEASE_SUCCESS_CRITERIA_VERSION,
            "physics_config_digest": lower_release_config_digest(self.environment.config),
            "bootstrap_version": LOWER_RELEASE_BOOTSTRAP_VERSION,
            "target_center": LOWER_RELEASE_TARGET,
            "network_config": {"type": "mlp_actor_critic", "hidden_sizes": [64, 64],
                               "activation": "tanh", "recurrent": False},
            "fixed_observation_normalization": True,
            "stage_name": self.stage_name,
            "stage_training_transitions": self.training_steps,
            "release_semantics": self.release_semantics,
            "source_checkpoint": self.source_checkpoint,
            "parent_checkpoint": self.parent_checkpoint,
        })
        temporary = path.with_name(".%s.augment.tmp" % path.name)
        torch.save(payload, temporary); temporary.replace(path)
        return path

    def load_checkpoint(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if tuple(payload.get("active_action_mask") or ()) != self.stage_active_action_mask:
            raise ValueError("stage checkpoint active mask incompatible")
        if payload.get("stage_name") != self.stage_name:
            raise ValueError("stage checkpoint name incompatible")
        if payload.get("release_semantics") != self.release_semantics:
            raise ValueError("stage checkpoint release semantics incompatible")
        self.source_checkpoint = payload.get("source_checkpoint")
        self.parent_checkpoint = payload.get("parent_checkpoint")
        return PPOTrainer.load_checkpoint(self, Path(checkpoint_path))


class FullPlaceEnvironment(LowerReleaseEnvironment):
    """Original grasped/lifted Place reset with all XYZ plus release active."""

    active_action_mask = FULL_PLACE_ACTIVE_ACTION_MASK

    def reset(self):
        # Deliberately bypass LowerReleaseEnvironment.reset: Full Place starts
        # at the original grasped + lifted (~60 mm) P4-1 reset, with no scripted
        # transport, descent, or release timing.
        observation = super(LowerReleaseEnvironment, self).reset()
        snapshot = self.scene.snapshot()
        metrics = self._metrics(snapshot)
        clearance = self._bottom_clearance(snapshot)
        self._policy_xy = tuple(snapshot["ee_state"][:2])
        self._target = tuple(snapshot["ee_state"][:3])
        self._place_previous_action = (0.0,) * 4
        self._steps = self._settle_steps = self._maximum_settle_steps = 0
        self._start_clearance = self._minimum_clearance = clearance
        self._minimum_bottom_clearance = clearance
        self._maximum_open_fraction = 0.0
        self._release_started = self._release_latched = False
        self._release_trigger_streak = 0; self._actual_jaw_opened = False
        self._detached = self._table_contact = self._inside_target = False
        self._descended_to_release_height = self._settled_half = False
        self._transport_reached = self._xy_error(snapshot) <= self.config.target_xy_tolerance
        self._release_height = self._detach_height = None
        self.phase = "LOWER_RELEASE_POLICY"
        self._bootstrap_last = dict(self._bootstrap_last)
        self._bootstrap_last.update({
            "full_place_reset": True,
            "full_place_bottom_clearance": clearance,
            "full_place_grasp_valid": bool(metrics["grasp_valid"]),
        })
        return self._place_observation(snapshot)
