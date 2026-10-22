import asyncio
import unittest

import httpx

from simulation_core import (
    CATALOG_VERSION, FakeBackendClient, FakeScheduler, InMemoryTaskRepository,
    SimulationService, TrainingTaskConfig, create_http_app,
    sha256_digest,
)


TOKEN = "catalog-token"
AUTH = {"Authorization": "Bearer " + TOKEN}


def push_config(**changes):
    values = TrainingTaskConfig(
        task_type="training", environment_type="s1_push", seed=7,
        rollout_steps=4, num_updates=1, checkpoint_every=1,
        episode_steps=8, physics_steps_per_action=1, batch_size=4,
    ).to_dict()
    values.update(changes)
    return values


class TaskCatalogTests(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        service = SimulationService(
            InMemoryTaskRepository(), FakeScheduler(), FakeBackendClient(),
            task_id_factory=lambda: "catalog-task",
        )
        self.scheduler = service._scheduler
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_http_app(service, TOKEN), raise_app_exceptions=False
            ), base_url="http://testserver",
        )

    def tearDown(self):
        self.loop.run_until_complete(self.client.aclose())
        self.loop.close()

    def request(self, method, path, **kwargs):
        return self.loop.run_until_complete(
            self.client.request(method, path, **kwargs)
        )

    def test_catalog_exposes_only_implemented_tasks(self):
        response = self.request("GET", "/simulation/task-catalog", headers=AUTH)
        self.assertEqual(200, response.status_code)
        expected = [
            {"id": "demo_s1_circle_tracking", "display_name": "S1 圆轨迹跟踪",
             "task_type": "demo", "robot": "s1", "supports_training": False, "supports_video": True},
            {"id": "demo_s1_joint_position", "display_name": "S1 关节位置控制",
             "task_type": "demo", "robot": "s1", "supports_training": False, "supports_video": True},
            {"id": "demo_s1_joint_sweep", "display_name": "S1 关节范围扫描",
             "task_type": "demo", "robot": "s1", "supports_training": False, "supports_video": True},
            {"id": "demo_s1_line_tracking", "display_name": "S1 直线轨迹跟踪",
             "task_type": "demo", "robot": "s1", "supports_training": False, "supports_video": True},
            {"id": "demo_s1_scripted_pick_place", "display_name": "S1 规则抓取放下",
             "task_type": "demo", "robot": "s1", "supports_training": False, "supports_video": True},
            {"id": "demo_s1_target_pose", "display_name": "S1 目标位姿控制",
             "task_type": "demo", "robot": "s1", "supports_training": False, "supports_video": True},
            {"id": "s1_grasp", "display_name": "S1 抓取",
             "task_type": "training", "robot": "s1",
             "supports_training": True, "supports_video": True},
            {"id": "s1_lift", "display_name": "S1 抓取抬起",
             "task_type": "training", "robot": "s1",
             "supports_training": True, "supports_video": True},
            {"id": "s1_lower_release", "display_name": "S1 抓取放下",
             "task_type": "training", "robot": "s1",
             "supports_training": True, "supports_video": True},
            {"id": "s1_place", "display_name": "S1 搬运放置",
             "task_type": "training", "robot": "s1",
             "supports_training": True, "supports_video": True},
            {"id": "s1_push", "display_name": "S1 推动任务",
             "task_type": "training", "robot": "s1",
             "supports_training": True, "supports_video": True},
            {"id": "s1_reach", "display_name": "S1 平面到达任务",
             "task_type": "training", "robot": "s1",
             "supports_training": True, "supports_video": True},
        ]
        payload = response.json()
        self.assertEqual(CATALOG_VERSION, payload["catalog_version"])
        self.assertEqual(payload["catalog_digest"], sha256_digest({
            "catalog_version": payload["catalog_version"],
            "tasks": payload["tasks"],
        }))
        self.assertEqual(expected, [
            {key: task[key] for key in expected[0]}
            for task in payload["tasks"]
        ])
        for task in payload["tasks"]:
            self.assertTrue(task["parameters"])
            self.assertEqual(len(task["parameters"]), len({
                parameter["name"] for parameter in task["parameters"]
            }))
        by_id = {task["id"]: task for task in payload["tasks"]}
        self.assertEqual(
            ["all_arm", 0, 1, 2, 3, 4, 5],
            next(parameter for parameter in by_id["demo_s1_joint_sweep"]["parameters"]
                 if parameter["name"] == "joint_index")["enum"],
        )

    def test_unknown_and_invalid_config_are_rejected_before_enqueue(self):
        unknown = push_config(environment_type="not_implemented")
        invalid = push_config(rollout_steps=0)
        for index, config in enumerate((unknown, invalid)):
            response = self.request(
                "POST", "/simulation/tasks", headers=AUTH,
                json={"run_id": "invalid-%d" % index, "project_id": "p", "config": config},
            )
            self.assertEqual(422, response.status_code)
            self.assertEqual("invalid_task_config", response.json()["code"])
        self.assertEqual([], self.scheduler.submitted_task_ids)

    def test_old_push_post_shape_is_accepted_and_canonicalized(self):
        config = push_config()
        response = self.request(
            "POST", "/simulation/tasks", headers=AUTH,
            json={"run_id": "push-compatible", "project_id": "p", "config": config},
        )
        self.assertEqual(202, response.status_code)
        self.assertEqual(config, response.json()["config"])


if __name__ == "__main__":
    unittest.main()
