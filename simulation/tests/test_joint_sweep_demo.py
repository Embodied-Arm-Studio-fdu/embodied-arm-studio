import unittest

from simulation_core.demos.config import SAFE_JOINT_POSES, parse_joint_sweep
from simulation_core.demos.joint_sweep import (
    ARM_JOINT_INDICES, SAFE_SWEEP_RANGES, build_sweep_targets,
    selected_joint_indices, smoothstep,
)
from simulation_core.cartesian_controller import S1_LOWER_LIMITS, S1_UPPER_LIMITS
from simulation_core.task_catalog import task_catalog
from simulation_core.task_registry import PRODUCTION_TASK_REGISTRY


def values(**changes):
    result={"task_type":"demo","environment_type":"demo_s1_joint_sweep","video_enabled":False}
    result.update(changes); return result


class JointSweepDemoTests(unittest.TestCase):
    def test_config_defaults_and_validation(self):
        config=parse_joint_sweep(values()); self.assertEqual("all_arm",config.joint_index); self.assertEqual(.2,config.speed_scale)
        for invalid in (-1,6,True,"0","gripper"):
            with self.assertRaises(ValueError): parse_joint_sweep(values(joint_index=invalid))
        for invalid in (.049,.501):
            with self.assertRaises(ValueError): parse_joint_sweep(values(speed_scale=invalid))

    def test_all_arm_order_and_single_joint(self):
        self.assertEqual(ARM_JOINT_INDICES,selected_joint_indices("all_arm"))
        self.assertEqual((4,),selected_joint_indices(4))

    def test_safe_ranges_are_inside_physical_limits(self):
        self.assertEqual(set(ARM_JOINT_INDICES),set(SAFE_SWEEP_RANGES))
        for index,(lower,upper) in SAFE_SWEEP_RANGES.items():
            self.assertLess(S1_LOWER_LIMITS[index],lower);self.assertLess(lower,upper);self.assertLess(upper,S1_UPPER_LIMITS[index])

    def test_plan_never_scans_gripper_and_returns_home(self):
        plan=build_sweep_targets("all_arm")
        self.assertEqual(1+5*6,len(plan));self.assertEqual(tuple(SAFE_JOINT_POSES["home"]),plan[-1][2])
        self.assertEqual(list(ARM_JOINT_INDICES),[item[1] for item in plan if item[0]=="lower"])
        for _,_,target in plan:
            self.assertEqual(SAFE_JOINT_POSES["home"][6:],target[6:])

    def test_smooth_interpolation(self):
        self.assertEqual(0,smoothstep(-1));self.assertEqual(0,smoothstep(0));self.assertAlmostEqual(.5,smoothstep(.5));self.assertEqual(1,smoothstep(1));self.assertEqual(1,smoothstep(2))

    def test_visible_after_gpu_gates_pass(self):
        definition=PRODUCTION_TASK_REGISTRY.get("demo_s1_joint_sweep")
        self.assertTrue(definition.catalog_visible)
        self.assertIn("demo_s1_joint_sweep",[item["id"] for item in task_catalog(PRODUCTION_TASK_REGISTRY)])


if __name__=="__main__": unittest.main()
