import unittest
from types import SimpleNamespace

import torch

from simulation_core.experimental.manipulation.target_compositor import (
    ExperimentalTargetCompositor, Workspace3D,
)


class FakeController:
    config = SimpleNamespace(
        max_joint_delta=0.08,
        lower_limits=(-3.,0.,0.,-2.,-2.,-2.,0.,0.),
        upper_limits=(3.,3.2,3.2,2.,2.,2.,.05,.05),
    )
    def compute(self, target, state):
        positions=torch.as_tensor(state.dof_positions).clone()
        positions[:6]+=0.01
        return SimpleNamespace(positions=positions,
            cartesian_error=torch.zeros(6), joint_delta=torch.ones(6)*0.01)


class TargetCompositorTests(unittest.TestCase):
    def test_composes_arm_and_symmetric_rate_limited_jaw_once(self):
        state=SimpleNamespace(dof_positions=torch.tensor([0.,1.,1.,0.,0.,0.,.05,.05]))
        item=ExperimentalTargetCompositor(FakeController(), Workspace3D(), .002)
        item.reset(state.dof_positions)
        command=item.compose((.4,0.,.5),(0.,0.,0.,1.),.01,state)
        self.assertTrue(torch.allclose(command.positions[:6], torch.tensor([.01,1.01,1.01,.01,.01,.01])))
        self.assertAlmostEqual(.048, float(command.positions[6]), places=6)
        self.assertAlmostEqual(float(command.positions[6]), float(command.positions[7]), places=7)
        self.assertFalse(command.workspace_clamped)

    def test_workspace_clamps_and_reports(self):
        state=SimpleNamespace(dof_positions=torch.tensor([0.,1.,1.,0.,0.,0.,.05,.05]))
        item=ExperimentalTargetCompositor(FakeController(), Workspace3D(), .002)
        command=item.compose((99.,0.,.5),(0.,0.,0.,1.),.05,state)
        self.assertTrue(command.workspace_clamped)

    def test_rejects_invalid_jaw(self):
        state=SimpleNamespace(dof_positions=torch.tensor([0.,1.,1.,0.,0.,0.,.05,.05]))
        item=ExperimentalTargetCompositor(FakeController())
        with self.assertRaises(ValueError): item.compose((.4,0.,.5),(0.,0.,0.,1.),-.1,state)


if __name__ == "__main__": unittest.main()

