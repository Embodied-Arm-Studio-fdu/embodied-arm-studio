import unittest

import torch

from simulation_core.cartesian_controller import (
    CartesianControllerConfig,
    ControllerAdapter,
)
from simulation_core.push_adapter import EETarget
from simulation_core.s1_isaac_gym_adapter import S1State


class CartesianControllerTestCase(unittest.TestCase):
    def make_state(self, jacobian=None):
        return S1State(
            dof_positions=torch.tensor((0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.01, 0.02)),
            dof_velocities=torch.zeros(8),
            ee_link_state=torch.tensor((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0) + (0.0,) * 6),
            arm_ee_jacobian=torch.eye(6) if jacobian is None else jacobian,
        )

    def test_dls_maps_position_error_to_arm_joint_delta(self):
        controller = ControllerAdapter(
            CartesianControllerConfig(damping=0.1, max_joint_delta=0.2)
        )
        target = EETarget((0.5, -0.1, 0.5), (0.0, 0.0, 0.0, 1.0))
        command = controller.compute(target, self.make_state())
        self.assertEqual((8,), tuple(command.positions.shape))
        self.assertGreater(float(command.joint_delta[0]), 0.0)
        self.assertLess(float(command.joint_delta[1]), 0.0)
        self.assertAlmostEqual(0.01, float(command.positions[6]), places=6)
        self.assertAlmostEqual(0.02, float(command.positions[7]), places=6)

    def test_singular_jacobian_is_safe_and_bounded(self):
        controller = ControllerAdapter(CartesianControllerConfig(max_joint_delta=0.03))
        target = EETarget((1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0))
        command = controller.compute(target, self.make_state(torch.zeros((6, 6))))
        self.assertTrue(torch.isfinite(command.positions).all())
        self.assertTrue(torch.all(command.joint_delta.abs() <= 0.03))

    def test_orientation_target_contributes_to_command(self):
        controller = ControllerAdapter()
        target = EETarget((0.4, 0.0, 0.5), (0.0, 0.0, 0.1, 0.994987))
        command = controller.compute(target, self.make_state())
        self.assertGreater(abs(float(command.joint_delta[5])), 0.0)


if __name__ == "__main__":
    unittest.main()
