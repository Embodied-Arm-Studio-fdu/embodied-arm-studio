import unittest

from robotics.scenes import default_s1_scene, scene_preflight
from robotics.simulation.dispatch import _scene_push_geometry, SimulationSnapshotMismatch


class ScenePreflightTests(unittest.TestCase):
    def test_default_s1_scene_is_trainable_without_place_claim(self):
        result = scene_preflight(default_s1_scene())
        self.assertTrue(result["valid"])
        self.assertTrue(result["trainable_push"])
        self.assertTrue(result["trainable_place"])

    def test_untranslated_obstacle_blocks_formal_training(self):
        scene = default_s1_scene()
        scene["objects"].append({"object_id": "wall-1", "type": "wall", "label": "wall", "position": [0.4, 0, 0], "rotation": [0, 0, 0], "scale": [0.1, 0.1, 0.1], "static": True})
        result = scene_preflight(scene)
        self.assertTrue(result["valid"])
        self.assertFalse(result["trainable_push"])
        self.assertFalse(result["trainable_place"])
        self.assertTrue(result["warnings"])

    def test_out_of_bounds_target_is_invalid(self):
        scene = default_s1_scene()
        scene["objects"][-1]["position"] = [2.0, 0, 0]
        result = scene_preflight(scene)
        self.assertFalse(result["valid"])
        self.assertFalse(result["trainable_push"])

    def test_saved_geometry_reaches_worker_with_correct_axes_and_table_height(self):
        scene = default_s1_scene()
        box = scene["objects"][1]
        box["position"] = [0.48, 0.05, -0.12]
        box["scale"] = [0.06, 0.10, 0.08]
        scene["objects"][2]["position"] = [0.62, 0.01, 0.15]
        self.assertTrue(scene_preflight(scene)["trainable_push"])
        result = _scene_push_geometry({"scene": {"document": scene}})
        self.assertEqual(result["object_size"], [0.06, 0.08, 0.10])
        self.assertEqual(result["object_start"], [0.48, -0.12, 0.425])
        self.assertEqual(result["goal_position"], [0.62, 0.15, 0.425])

    def test_invalid_or_untranslated_geometry_cannot_reach_worker(self):
        for field, value in [("scale", [0, 0.1, 0.1]),
                             ("scale", [0.1, float("nan"), 0.1]),
                             ("position", [0.19, 0.05, 0]),
                             ("position", [0.45, 0.2, 0]),
                             ("rotation", [0, 0.3, 0])]:
            with self.subTest(field=field, value=value):
                scene = default_s1_scene()
                scene["objects"][1][field] = value
                self.assertFalse(scene_preflight(scene)["trainable_push"])
                with self.assertRaises(SimulationSnapshotMismatch):
                    _scene_push_geometry({"scene": {"document": scene}})


if __name__ == "__main__":
    unittest.main()
