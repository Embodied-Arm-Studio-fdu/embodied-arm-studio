import math
import unittest

from simulation_core.demos.circle_tracking import circle_target
from simulation_core.demos.config import (
    SAFE_JOINT_POSES, parse_circle_tracking, parse_joint_position,
    parse_line_tracking, parse_target_pose,
)
from simulation_core.demos.line_tracking import line_target
from simulation_core.task_registry import DEMO_RUNNER, PRODUCTION_TASK_REGISTRY


def base(task_id, **extra):
    values={"task_type":"demo","environment_type":task_id,"video_enabled":False}
    values.update(extra); return values


class S1ControlDemoTests(unittest.TestCase):
    def test_four_strict_parsers_and_defaults(self):
        self.assertEqual(parse_target_pose(base("demo_s1_target_pose")).target_offset_xy_m,(.08,.05))
        self.assertEqual(parse_line_tracking(base("demo_s1_line_tracking")).line_length_m,.15)
        self.assertEqual(parse_circle_tracking(base("demo_s1_circle_tracking")).radius_m,.05)
        self.assertEqual(parse_joint_position(base("demo_s1_joint_position")).pose,"ready")

    def test_duration_range_and_unknown_fields(self):
        for value in (4.9,300.1):
            with self.assertRaises(ValueError): parse_target_pose(base("demo_s1_target_pose",duration_seconds=value))
        with self.assertRaises(ValueError): parse_target_pose(base("demo_s1_target_pose",controller_gain=1))

    def test_target_offset_validation(self):
        with self.assertRaises(ValueError): parse_target_pose(base("demo_s1_target_pose",target_offset_xy_m=(.2,0)))

    def test_line_math(self):
        center=(.4,0); self.assertAlmostEqual(line_target(center,.15,"x",6,1.5)[0],.475)
        self.assertAlmostEqual(line_target(center,.15,"x",6,4.5)[0],.325)

    def test_circle_math(self):
        self.assertEqual(circle_target((.39,0),.05,8,0),(.44,0.0,circle_target((.39,0),.05,8,0)[2]))
        target=circle_target((.39,0),.05,8,2); self.assertAlmostEqual(target[0],.39); self.assertAlmostEqual(target[1],.05)

    def test_joint_pose_allowlist(self):
        self.assertEqual(set(SAFE_JOINT_POSES),{"home","ready","side"})
        self.assertEqual(SAFE_JOINT_POSES["home"][:6], (0.0, 0.24, 0.16, -0.33, 0.0, 0.0))
        with self.assertRaises(ValueError): parse_joint_position(base("demo_s1_joint_position",pose="arbitrary"))

    def test_registry_catalog_contract(self):
        for task_id in ("demo_s1_target_pose","demo_s1_line_tracking","demo_s1_circle_tracking","demo_s1_joint_position"):
            definition=PRODUCTION_TASK_REGISTRY.get(task_id); self.assertEqual(definition.runner_kind,DEMO_RUNNER); self.assertFalse(definition.supports_training)
        self.assertTrue(PRODUCTION_TASK_REGISTRY.get("demo_s1_joint_sweep").catalog_visible)

    def test_video_alias_properties(self):
        config=parse_circle_tracking(base("demo_s1_circle_tracking",width=320,height=240,fps=10))
        self.assertEqual((config.video_width,config.video_height,config.video_fps),(320,240,10))


if __name__ == "__main__": unittest.main()
