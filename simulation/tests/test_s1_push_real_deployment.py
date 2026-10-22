import importlib
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "real_robot" / "s1_push"
if str(REAL) not in sys.path:
    sys.path.insert(0, str(REAL))

config = importlib.import_module("config")
robot_module = importlib.import_module("s1_robot_adapter")
safety_module = importlib.import_module("safety_filter")


class FakeArm:
    def __init__(self):
        self.enabled = False
        self.commands = []
        self.disabled = 0

    def refresh(self): pass
    def get_pos(self): return [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    def get_vel(self): return [0.0] * 6
    def get_fk_quat(self, q): return [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    def enable(self): self.enabled = True
    def joint_control(self, q): self.commands.append(tuple(q)); return True
    def disable(self): self.disabled += 1


class RealDeploymentSafetyTest(unittest.TestCase):
    def test_dry_run_never_enables_or_sends(self):
        arm = FakeArm()
        adapter = robot_module.S1RobotAdapter(
            "fake", dry_run=True, enable_real_motion=True, arm_factory=lambda: arm
        ).connect()
        self.assertFalse(adapter.send_command([0.0] * 6))
        self.assertFalse(arm.enabled)
        self.assertEqual([], arm.commands)

    def test_both_motion_keys_are_required(self):
        arm = FakeArm()
        adapter = robot_module.S1RobotAdapter(
            "fake", dry_run=False, enable_real_motion=False, arm_factory=lambda: arm
        ).connect()
        self.assertFalse(adapter.send_command([0.0] * 6))
        self.assertFalse(arm.enabled)

    def test_stop_uses_disable(self):
        arm = FakeArm()
        adapter = robot_module.S1RobotAdapter("fake", arm_factory=lambda: arm).connect()
        adapter.stop()
        self.assertEqual(1, arm.disabled)

    def test_motion_limits_fail_closed_when_unapproved(self):
        filt = safety_module.S1SafetyFilter(config.MotionLimits())
        with self.assertRaises(safety_module.SafetyViolation):
            filt.validate_workspace((0.4, 0.0, 0.4))
        with self.assertRaises(safety_module.SafetyViolation):
            filt.filter_joint_command([0.0] * 6, [0.0] * 6, 0.1)
        with self.assertRaises(safety_module.SafetyViolation):
            filt.filter_policy_action((0.006, 0.0))

    def test_joint_position_step_and_velocity_limits(self):
        limits = config.MotionLimits(
            workspace_min_policy_m=(0.2, -0.2, 0.2),
            workspace_max_policy_m=(0.7, 0.2, 0.6),
            max_joint_step_rad=(0.01,) * 6,
            max_joint_velocity_rad_s=(0.1,) * 6,
        )
        filt = safety_module.S1SafetyFilter(limits)
        with self.assertRaises(safety_module.SafetyViolation):
            filt.filter_joint_command([0.0] * 6, [0.02] + [0.0] * 5, 0.2)
        self.assertEqual(
            (0.005,) + (0.0,) * 5,
            filt.filter_joint_command([0.0] * 6, [0.005] + [0.0] * 5, 0.2),
        )

    def test_selected_checkpoint_contract_and_deterministic_action(self):
        sys.path.insert(0, str(ROOT / "src"))
        from simulation_core.ppo.normalizer import ObservationNormalizer
        from simulation_core.ppo.policy import MLPActorCritic
        from simulation_core.ppo.trainer import (
            validate_checkpoint_action_schema,
            validate_checkpoint_observation_schema,
        )
        payload = torch.load(config.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        validate_checkpoint_action_schema(payload)
        validate_checkpoint_observation_schema(payload)
        policy = MLPActorCritic(); policy.load_state_dict(payload["policy"]); policy.eval()
        normalizer = ObservationNormalizer(frozen=True)
        normalizer.load_state_dict(payload["normalizer_state"], frozen=True)
        raw = payload["normalizer_state"]["mean"].float()
        with torch.no_grad():
            first = policy.act(normalizer.normalize(raw), deterministic=True)[0]
            second = policy.act(normalizer.normalize(raw), deterministic=True)[0]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first.abs() <= 0.005000001))


if __name__ == "__main__":
    unittest.main()

