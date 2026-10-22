import unittest

from simulation_core import (
    DEMO_ARTIFACT_POLICY, DemoTaskConfig, ManipulationTrainingTaskConfig,
    PRODUCTION_TASK_REGISTRY,
    S1_PUSH_DEFINITION, S1_REACH_DEFINITION, TaskDefinition, TaskRegistry, TrainingTaskConfig,
    UnknownTaskDefinition,
)
from simulation_core.reward_extension import RewardExpression, starter_source


def push_values(**changes):
    values = TrainingTaskConfig(
        task_type="training", environment_type="s1_push", seed=7,
        rollout_steps=4, num_updates=1, checkpoint_every=1,
        episode_steps=8, physics_steps_per_action=1, batch_size=4,
    ).to_dict()
    values.update(changes)
    return values


class TaskRegistryTests(unittest.TestCase):
    def test_registry_contains_existing_and_manipulation_tasks(self):
        self.assertIs(S1_PUSH_DEFINITION, PRODUCTION_TASK_REGISTRY.get("s1_push"))
        self.assertIs(S1_REACH_DEFINITION, PRODUCTION_TASK_REGISTRY.get("s1_reach"))
        self.assertEqual(
            ["demo_s1_circle_tracking", "demo_s1_joint_position",
             "demo_s1_joint_sweep", "demo_s1_line_tracking",
             "demo_s1_scripted_pick_place",
             "demo_s1_target_pose", "s1_grasp", "s1_lift",
               "s1_lower_release", "s1_place", "s1_push", "s1_reach"],
            [x.id for x in PRODUCTION_TASK_REGISTRY.definitions()],
        )
        self.assertTrue(S1_REACH_DEFINITION.catalog_visible)
        place = PRODUCTION_TASK_REGISTRY.get("s1_place")
        self.assertTrue(place.supports_training)
        self.assertIn("validated_target_candidate", place.semantics.preconditions)
        scripted = PRODUCTION_TASK_REGISTRY.get("demo_s1_scripted_pick_place")
        self.assertTrue(scripted.catalog_visible)
        config = scripted.parse_config({
            item["name"]: item["default"] for item in scripted.parameters
        })
        self.assertEqual(4, config.physics_steps_per_command)

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(UnknownTaskDefinition):
            PRODUCTION_TASK_REGISTRY.get("not_implemented")

    def test_push_parser_is_strict_and_canonical(self):
        parsed = PRODUCTION_TASK_REGISTRY.parse_config(push_values())
        self.assertIsInstance(parsed, TrainingTaskConfig)
        self.assertEqual(push_values(), parsed.to_dict())
        with self.assertRaises(ValueError):
            PRODUCTION_TASK_REGISTRY.parse_config(push_values(unknown=True))

    def test_place_parser_preserves_validated_reward_extension(self):
        definition = PRODUCTION_TASK_REGISTRY.get("s1_place")
        values = {
            parameter["name"]: parameter["default"]
            for parameter in definition.parameters
        }
        source = starter_source("s1_place")
        extension = {
            "schema": "reward-expression/v1",
            "source": source,
            "sha256": RewardExpression("s1_place", source).digest,
        }
        parsed = PRODUCTION_TASK_REGISTRY.parse_config(
            dict(values, reward_extension=extension)
        )
        self.assertIsInstance(parsed, ManipulationTrainingTaskConfig)
        self.assertEqual(extension, parsed.to_dict()["reward_extension"])

        grasp_values = dict(values, environment_type="s1_grasp")
        with self.assertRaises(ValueError):
            PRODUCTION_TASK_REGISTRY.parse_config(
                dict(grasp_values, reward_extension=extension)
            )

    def test_definition_rejects_mismatched_runner_metadata(self):
        with self.assertRaises(ValueError):
            TaskDefinition(
                id="bad", display_name="bad", task_type="demo", robot="s1",
                supports_training=True, supports_video=False,
                runner_kind="demo", config_parser=lambda values: values,
                artifact_policy=DEMO_ARTIFACT_POLICY,
            )

    def test_registry_can_hold_test_only_demo_without_production_exposure(self):
        definition = TaskDefinition(
            id="fake_demo", display_name="Fake", task_type="demo", robot="s1",
            supports_training=False, supports_video=False, runner_kind="demo",
            config_parser=lambda values: DemoTaskConfig(**values),
            runtime_factory=lambda config: object(),
            artifact_policy=DEMO_ARTIFACT_POLICY,
        )
        registry = TaskRegistry((definition,))
        parsed = registry.parse_config({
            "task_type": "demo", "environment_type": "fake_demo",
            "duration_seconds": 1,
        })
        self.assertEqual("fake_demo", parsed.environment_type)
        with self.assertRaises(UnknownTaskDefinition):
            PRODUCTION_TASK_REGISTRY.get("fake_demo")


if __name__ == "__main__":
    unittest.main()
