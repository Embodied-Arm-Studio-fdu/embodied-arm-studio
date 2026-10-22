import unittest

from simulation_core.experimental.manipulation.scene import ManipulationSceneConfig
from simulation_core.experimental.manipulation.scripted_runtime import ScriptedRuntimeConfig


class SceneConfigTests(unittest.TestCase):
    def test_explicit_cube_physics_and_pose(self):
        config=ManipulationSceneConfig()
        self.assertEqual(.045,config.cube_size)
        self.assertEqual(300.,config.cube_density)
        self.assertEqual(1.,config.cube_friction)
        self.assertEqual(0.,config.cube_restitution)
        self.assertEqual(300.,config.gripper_stiffness)
        self.assertEqual(20.,config.gripper_damping)
        self.assertAlmostEqual(.398,config.cube_start[2])

    def test_rejects_implicit_or_invalid_physics(self):
        with self.assertRaises(ValueError): ManipulationSceneConfig(cube_density=0)
        with self.assertRaises(ValueError): ManipulationSceneConfig(cube_friction=0)

    def test_stage_limits_and_jaw_order(self):
        config=ScriptedRuntimeConfig()
        self.assertLess(config.close_safe,config.open_safe)
        self.assertEqual(5.,config.hold_seconds)
        self.assertEqual(2.,config.settle_seconds)


if __name__ == "__main__": unittest.main()
