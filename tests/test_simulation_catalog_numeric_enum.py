import json
import unittest

from robotics.simulation.contracts import (
    SimulationTaskCatalog,
    resolve_catalog_task_parameters,
)


class SimulationCatalogNumericEnumTests(unittest.TestCase):
    @staticmethod
    def _catalog(default=0.0):
        return SimulationTaskCatalog.from_dict({
            "tasks": [{
                "id": "s1_lift",
                "display_name": "S1 Lift",
                "task_type": "training",
                "robot": "s1",
                "supports_training": True,
                "supports_video": True,
                "parameters": [{
                    "name": "randomize_xy_m",
                    "type": "enum",
                    "default": default,
                    "enum": [0.0],
                }],
            }],
        })

    def test_integral_float_enum_accepts_browser_json_integer(self):
        supplied = json.loads(json.dumps({"randomize_xy_m": 0.0}))
        supplied["randomize_xy_m"] = int(supplied["randomize_xy_m"])

        _, resolved = resolve_catalog_task_parameters(
            self._catalog(), "s1_lift", "training", supplied
        )

        self.assertEqual(0, resolved["randomize_xy_m"])

    def test_numeric_enum_still_rejects_boolean(self):
        with self.assertRaisesRegex(
            ValueError, "randomize_xy_m does not match catalog type"
        ):
            resolve_catalog_task_parameters(
                self._catalog(),
                "s1_lift",
                "training",
                {"randomize_xy_m": False},
            )


if __name__ == "__main__":
    unittest.main()
