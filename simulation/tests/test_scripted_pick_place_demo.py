import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simulation_core.demos.config import parse_scripted_pick_place
from simulation_core.demos.scripted_pick_place import ScriptedPickPlaceRuntime


class ScriptedPickPlaceDemoTests(unittest.TestCase):
    def config(self, **changes):
        values = {
            "task_type": "demo",
            "environment_type": "demo_s1_scripted_pick_place",
            "physics_steps_per_command": 4,
            "video_enabled": False,
        }
        values.update(changes)
        return parse_scripted_pick_place(values)

    def test_parser_locks_proven_physics_and_video_shape(self):
        config = self.config()
        self.assertEqual(4, config.physics_steps_per_command)
        self.assertEqual((640, 480), (config.width, config.height))
        with self.assertRaisesRegex(ValueError, "physics_steps_per_command=4"):
            self.config(physics_steps_per_command=1)
        with self.assertRaisesRegex(ValueError, "640x480"):
            self.config(width=320)

    def test_wrapper_runs_existing_state_machine_once_and_reports_evidence(self):
        scene = Mock()
        scene.initialize.return_value = scene
        program = Mock()
        program.run_trial.return_value = {
            "success": True,
            "stage_reached": "SETTLE",
            "resource_cleanup": True,
            "trace": [{"large": "omitted"}],
            "video_path": None,
        }
        with patch(
            "simulation_core.demos.scripted_pick_place.ManipulationScene",
            return_value=scene,
        ), patch(
            "simulation_core.demos.scripted_pick_place.ScriptedManipulationRuntime",
            return_value=program,
        ):
            runtime = ScriptedPickPlaceRuntime(
                self.config(), SimpleNamespace(run_id="run", task_id="task")
            )
            runtime.initialize()
            first = runtime.step()
            second = runtime.step()
            runtime.close()
        program.run_trial.assert_called_once_with(7)
        self.assertEqual(first, second)
        self.assertTrue(first["success"])
        self.assertEqual("deterministic_rule_state_machine", first["controller"])
        self.assertNotIn("trace", first)
        scene.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
