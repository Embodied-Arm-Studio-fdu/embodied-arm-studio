import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from robotics.runs import InMemoryRunRepository
from tests.http_client import ASGITestClient


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.client = ASGITestClient(
            create_app(
                root / "runs",
                tick_seconds=0,
                project_root=root / "projects",
                run_repository=InMemoryRunRepository(),
                current_user_provider=StaticCurrentUserProvider("test-user"),
            )
        )

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def launch_and_wait(self, strategy_id):
        response = self.client.post("/api/runs/train", json={
            "environment_id": "mock-reach",
            "strategy_id": strategy_id,
            "seed": 11,
            "max_steps": 8,
            "config": {"action_scale": 0.4},
        })
        self.assertEqual(response.status_code, 202)
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            record = self.client.get("/api/runs/%s" % run_id).json()
            if record["status"] not in ("queued", "running"):
                return run_id, record
            time.sleep(0.005)
        self.fail("API run did not finish")

    def test_discovery_training_events_comparison_and_export(self):
        environments = self.client.get("/api/environments")
        self.assertEqual(environments.status_code, 200)
        self.assertEqual(environments.json()[0]["id"], "mock-reach")
        self.assertEqual(len(self.client.get("/api/strategies").json()), 2)

        first_id, first = self.launch_and_wait("joint-p-control")
        second_id, second = self.launch_and_wait("policy-gradient-demo")
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "completed")
        events = self.client.get("/api/runs/%s/events" % first_id).json()
        self.assertEqual(len([event for event in events if event["type"] == "metric"]), 8)

        compared = self.client.post("/api/runs/compare", json={"run_ids": [first_id, second_id]})
        self.assertEqual(compared.status_code, 200)
        self.assertEqual(len(compared.json()["runs"]), 2)
        exported = self.client.post("/api/runs/%s/export" % first_id, json={"format": "manifest"})
        self.assertEqual(exported.status_code, 200)
        self.assertFalse(exported.json()["real_robot_control_enabled"])
        self.assertFalse(exported.json()["export_gate"]["eligible"])

        evaluation = self.client.post("/api/evaluations", json={"run_id": first_id})
        blocked = self.client.post("/api/runs/%s/export" % first_id, json={
            "format": "onnx",
            "evaluation_report_id": evaluation.json()["report_id"],
        })
        self.assertEqual(blocked.status_code, 409)
        self.assertIn("导出门禁未满足", blocked.json()["detail"])

    def test_invalid_config_is_a_validation_error(self):
        response = self.client.post("/api/runs/train", json={
            "environment_id": "mock-reach",
            "strategy_id": "joint-p-control",
            "config": {"action_scale": 9},
        })
        self.assertEqual(response.status_code, 422)

    def test_unknown_run_is_not_found(self):
        response = self.client.get("/api/runs/not-a-run")
        self.assertEqual(response.status_code, 404)

    def test_pipeline_discovery_and_validation(self):
        pipelines = self.client.get("/api/pipelines")
        self.assertEqual(pipelines.status_code, 200)
        self.assertEqual(pipelines.json()[0]["pipeline_id"], "pipeline_mock_reach_baseline")

        document = self.client.get("/api/pipelines/pipeline_mock_reach_baseline").json()
        validation = self.client.post("/api/pipelines/validate", json={"document": document})
        self.assertEqual(validation.status_code, 200)
        self.assertTrue(validation.json()["valid"])

    def test_v1_workspace_crud_search_and_snapshot(self):
        created = self.client.post("/api/v1/projects", json={"project_id": "workspace-course"})
        self.assertEqual(created.status_code, 201)
        project_id = created.json()["project_id"]
        project_ids = [item["project_id"] for item in self.client.get("/api/v1/projects").json()]
        self.assertIn(project_id, project_ids)

        directory = self.client.post("/api/v1/workspaces/%s/directories" % project_id, json={"path": "src"})
        self.assertEqual(directory.status_code, 201)
        written = self.client.put("/api/v1/workspaces/%s/files/src/task.py" % project_id, json={"content": "reward = distance\n"})
        self.assertEqual(written.status_code, 200)
        workspace = self.client.get("/api/v1/workspaces/%s" % project_id).json()
        self.assertEqual(workspace["tree"][0]["name"], "src")
        search_results = self.client.get("/api/v1/workspaces/%s/search?q=distance" % project_id).json()
        self.assertIn({"path": "src/task.py", "line": 1, "preview": "reward = distance"}, search_results)

        snapshot = self.client.post("/api/v1/workspaces/%s/snapshots" % project_id).json()["snapshot_id"]
        self.client.put("/api/v1/workspaces/%s/files/src/task.py" % project_id, json={"content": "reward = 0\n"})
        restored = self.client.post("/api/v1/workspaces/%s/snapshots/%s/restore" % (project_id, snapshot))
        self.assertEqual(restored.status_code, 200)
        content = self.client.get("/api/v1/workspaces/%s/files/src/task.py" % project_id).json()["content"]
        self.assertIn("distance", content)

    def test_v1_workspace_blocks_traversal_and_unsafe_types(self):
        project_id = self.client.post("/api/v1/projects", json={"project_id": "safe-workspace"}).json()["project_id"]
        unsafe_type = self.client.put("/api/v1/workspaces/%s/files/run.sh" % project_id, json={"content": "exit 0"})
        self.assertEqual(unsafe_type.status_code, 422)
        traversal = self.client.put("/api/v1/workspaces/%s/files/../escape.py" % project_id, json={"content": "pass"})
        self.assertIn(traversal.status_code, (404, 422))

    def test_v1_workspace_history_diff_and_starter_restore(self):
        project_id = self.client.post("/api/v1/projects", json={"project_id": "history-workspace"}).json()["project_id"]
        path = "/api/v1/workspaces/%s/files/controller.json" % project_id
        self.client.put(path, json={"content": '{\n  "action_scale": 0.45\n}\n'})
        self.client.put(path, json={"content": '{\n  "action_scale": 0.55\n}\n'})

        history = self.client.get("/api/v1/workspaces/%s/history/controller.json" % project_id)
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()), 2)
        diff = self.client.get("/api/v1/workspaces/%s/diff?path=controller.json" % project_id)
        self.assertEqual(diff.json()["status"], "modified")
        self.assertIn("+  \"action_scale\": 0.55", diff.json()["diff"])

        version_id = history.json()[1]["version_id"]
        restored_version = self.client.post("/api/v1/workspaces/%s/history/controller.json/versions/%s/restore" % (project_id, version_id))
        self.assertEqual(restored_version.status_code, 200)
        self.assertIn("0.45", self.client.get(path).json()["content"])
        restored_starter = self.client.post("/api/v1/workspaces/%s/starter/restore" % project_id)
        self.assertEqual(restored_starter.status_code, 200)
        self.assertTrue(restored_starter.json()["snapshot_id"].startswith("snapshot-"))
        self.assertIn("0.35", self.client.get(path).json()["content"])

    def test_agent_policy_requires_level_and_student_design_for_patch(self):
        project_id = self.client.post("/api/projects", json={"project_id": "policy-course"}).json()["project_id"]
        level_four = self.client.post("/api/agent/sessions", json={
            "project_id": project_id,
            "provider": "mock",
            "assistance_level": 4,
        }).json()["session_id"]
        blocked = self.client.post("/api/agent/sessions/%s/messages" % level_four, json={
            "content": "请生成 patch",
        })
        self.assertEqual(blocked.status_code, 403)

        level_five = self.client.post("/api/agent/sessions", json={
            "project_id": project_id,
            "provider": "mock",
            "assistance_level": 5,
        }).json()["session_id"]
        patch_payload = {
            "session_id": level_five,
            "project_id": project_id,
            "path": "controller.json",
            "diff": "--- a/controller.json\n+++ b/controller.json\n@@ -1,1 +1,1 @@\n-{}\n+{}",
        }
        self.assertEqual(self.client.post("/api/agent/patches", json=patch_payload).status_code, 403)

        design = self.client.put("/api/agent/sessions/%s/student-design" % level_five, json={
            "observation": "joint positions",
            "action": "joint targets",
            "reward": "distance reduction",
            "termination": "success or timeout",
            "expected_result": "distance decreases",
        })
        self.assertEqual(design.status_code, 200)
        self.assertTrue(design.json()["complete"])
        participation = self.client.get("/api/agent/sessions/%s/participation" % level_five).json()
        self.assertEqual(participation[0]["event"], "student_design_submitted")

    def test_agent_and_recent_http_contracts_are_exposed(self):
        providers = self.client.get("/api/agent/providers").json()
        self.assertEqual(
            {item["id"] for item in providers},
            {"mock", "openai-compatible", "deepseek", "anthropic"},
        )

        levels = self.client.get("/api/agent/assistance-levels").json()
        self.assertEqual([item["level"] for item in levels], [1, 2, 3, 4, 5])
        self.assertIn("propose_patch", levels[-1]["capabilities"])

        paths = self.client.get("/openapi.json").json()["paths"]
        expected_operations = {
            ("/api/agent/providers/{provider_id}/configuration", "put"),
            ("/api/agent/providers/{provider_id}/configuration", "delete"),
            ("/api/ready", "get"),
            ("/api/v1/classes/{class_id}/runs", "get"),
        }
        self.assertTrue(all(method in paths[path] for path, method in expected_operations))

        level_six = self.client.post("/api/agent/sessions", json={
            "project_id": "any-project",
            "provider": "mock",
            "assistance_level": 6,
        })
        self.assertEqual(level_six.status_code, 422)

    def test_agent_provider_configuration_and_page_context_are_server_controlled(self):
        project_id = self.client.post(
            "/api/projects", json={"project_id": "context-course"}
        ).json()["project_id"]
        with patch.dict("os.environ", {}, clear=True):
            browser_override = self.client.post("/api/agent/sessions", json={
                "project_id": project_id,
                "provider": "openai-compatible",
                "model": "browser-controlled-model",
                "base_url": "https://browser.invalid",
            })
            unavailable = self.client.post("/api/agent/sessions", json={
                "project_id": project_id,
                "provider": "openai-compatible",
            })
        self.assertEqual(browser_override.status_code, 422)
        self.assertEqual(unavailable.status_code, 503)

        session_id = self.client.post("/api/agent/sessions", json={
            "project_id": project_id,
            "provider": "mock",
        }).json()["session_id"]
        reply = self.client.post("/api/agent/sessions/%s/messages" % session_id, json={
            "content": "我现在应该检查什么？",
            "context": {
                "path": "/runs/run-01",
                "project_id": project_id,
                "run_id": "run-01",
                "scene_id": "scene-s1-v2-visual",
            },
        })
        self.assertEqual(reply.status_code, 200)
        self.assertIn("当前问题", reply.json()["content"])

    def test_agent_runtime_provider_config_never_returns_secret_and_can_be_cleared(self):
        project_id = self.client.post(
            "/api/projects", json={"project_id": "provider-revocation-course"}
        ).json()["project_id"]
        with patch("apps.api.main.validate_runtime_provider_config", return_value="https://api.deepseek.com"):
            configured = self.client.put("/api/agent/providers/deepseek/configuration", json={
                "api_key": "sk-test-secret",
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com",
            })
        self.assertEqual(configured.status_code, 200)
        self.assertTrue(configured.json()["configured"])
        self.assertNotIn("sk-test-secret", configured.text)
        self.assertNotIn("api_key", configured.json())
        status = self.client.get("/api/agent/providers").json()
        deepseek = next(item for item in status if item["id"] == "deepseek")
        self.assertTrue(deepseek["configured"])
        session_id = self.client.post("/api/agent/sessions", json={
            "project_id": project_id,
            "provider": "deepseek",
        }).json()["session_id"]
        cleared = self.client.delete("/api/agent/providers/deepseek/configuration")
        self.assertEqual(cleared.status_code, 200)
        self.assertFalse(cleared.json()["configured"])
        revoked = self.client.post("/api/agent/sessions/%s/messages" % session_id, json={
            "content": "旧凭据不应继续可用",
        })
        self.assertEqual(revoked.status_code, 404)

    def test_agent_context_tools_follow_assistance_capabilities(self):
        project_id = self.client.post("/api/projects", json={"project_id": "capability-course"}).json()["project_id"]
        context = {
            "path": "/scenes/scene-s1-v2-visual",
            "project_id": project_id,
            "scene_id": "scene-s1-v2-visual",
        }
        level_three = self.client.post("/api/agent/sessions", json={
            "project_id": project_id, "provider": "mock", "assistance_level": 3,
        }).json()["session_id"]
        level_three_reply = self.client.post("/api/agent/sessions/%s/messages" % level_three, json={
            "content": "请解释当前场景。", "context": context,
        })
        self.assertEqual(level_three_reply.status_code, 200)
        self.assertEqual(level_three_reply.json()["tools_used"], [])

        level_four = self.client.post("/api/agent/sessions", json={
            "project_id": project_id, "provider": "mock", "assistance_level": 4,
        }).json()["session_id"]
        level_four_reply = self.client.post("/api/agent/sessions/%s/messages" % level_four, json={
            "content": "请解释当前页面。", "context": context,
        })
        self.assertEqual(level_four_reply.status_code, 200)
        self.assertEqual(level_four_reply.json()["tools_used"], ["page_context"])

        level_five = self.client.post("/api/agent/sessions", json={
            "project_id": project_id, "provider": "mock", "assistance_level": 5,
        }).json()["session_id"]
        level_five_reply = self.client.post("/api/agent/sessions/%s/messages" % level_five, json={
            "content": "请检查当前场景。", "context": context,
        })
        self.assertEqual(level_five_reply.status_code, 200)
        self.assertEqual(level_five_reply.json()["tools_used"], ["page_context", "workspace_read", "scene_read"])

    def test_agent_context_rejects_page_from_another_project(self):
        project_id = self.client.post("/api/projects", json={"project_id": "owned-course"}).json()["project_id"]
        other_project_id = self.client.post("/api/projects", json={"project_id": "other-course"}).json()["project_id"]
        session_id = self.client.post("/api/agent/sessions", json={
            "project_id": project_id, "provider": "mock", "assistance_level": 5,
        }).json()["session_id"]
        response = self.client.post("/api/agent/sessions/%s/messages" % session_id, json={
            "content": "请检查页面。",
            "context": {"path": "/projects/other", "project_id": other_project_id},
        })
        self.assertEqual(response.status_code, 422)

    def test_evaluation_withholds_score_when_evidence_is_incomplete(self):
        run_id, _ = self.launch_and_wait("joint-p-control")
        response = self.client.post("/api/evaluations", json={"run_id": run_id})
        self.assertEqual(response.status_code, 201)
        report = response.json()
        self.assertEqual(report["overall"]["status"], "not_assessed")
        self.assertIsNone(report["overall"]["score"])
        self.assertLess(report["evidence_completeness"], 0.8)
        self.assertEqual(report["sim2real"]["status"], "blocked")
        persisted = self.client.get("/api/evaluations/%s" % report["report_id"])
        self.assertEqual(persisted.status_code, 200)
        self.assertEqual(persisted.json(), report)

    def test_cohort_evaluation_persists_multiple_run_evidence(self):
        first_id, _ = self.launch_and_wait("joint-p-control")
        second_id, _ = self.launch_and_wait("joint-p-control")
        response = self.client.post("/api/evaluations", json={"run_ids": [first_id, second_id]})
        self.assertEqual(response.status_code, 201)
        report = response.json()
        self.assertEqual(report["run_ids"], [first_id, second_id])
        reproducibility = next(item for item in report["criteria"] if item["criterion_id"] == "reproducibility")
        self.assertEqual(reproducibility["status"], "assessed")
        persisted = self.client.get("/api/evaluations/%s" % report["report_id"])
        self.assertEqual(persisted.json()["run_ids"], [first_id, second_id])
        listed = self.client.get("/api/evaluations")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()[0]["report_id"], report["report_id"])

    def test_evaluation_reports_can_be_compared(self):
        run_id, _ = self.launch_and_wait("joint-p-control")
        before = self.client.post("/api/evaluations", json={"run_id": run_id}).json()
        after = self.client.post("/api/evaluations", json={"run_id": run_id}).json()

        response = self.client.post("/api/evaluations/compare", json={
            "before_report_id": before["report_id"],
            "after_report_id": after["report_id"],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["before_report_id"], before["report_id"])
        self.assertEqual(response.json()["after_report_id"], after["report_id"])
        self.assertTrue(all(item["change"] == "unchanged" for item in response.json()["criteria"]))

    def test_scene_can_be_saved_loaded_and_exported(self):
        scene = self.client.get("/api/scenes/scene_mock_reach_lab").json()
        scene["name"] = "教学台改版"
        scene["objects"].append({"object_id": "box-01", "type": "box", "label": "障碍箱", "position": [0.4, 0.15, 0.2], "rotation": [0, 0, 0], "scale": [0.3, 0.3, 0.3], "color": "#2d6cdf", "static": True})
        saved = self.client.post("/api/scenes", json={"document": scene})
        self.assertEqual(saved.status_code, 201)
        scene_id = saved.json()["scene_id"]
        self.assertEqual(self.client.get("/api/scenes/" + scene_id).status_code, 200)
        exported = self.client.get("/api/scenes/" + scene_id + "/export")
        self.assertEqual(exported.status_code, 200)
        self.assertFalse(exported.json()["real_robot_control_enabled"])

    def test_default_s1_scene_is_visual_only_and_runtime_aligned(self):
        scene = self.client.get("/api/scenes/scene-s1-v2-visual")

        self.assertEqual(scene.status_code, 200)
        document = scene.json()
        self.assertEqual(document["capability"], "simulation")
        self.assertEqual(document["robot"]["id"], "yhrg-s1-v2")
        self.assertEqual(document["robot"]["dof"], 6)
        self.assertEqual(document["robot"]["gripper_dof"], 2)
        self.assertEqual(document["robot"]["end_effector"], "ee_link")
        self.assertEqual(document["robot"]["asset"], "yhrg_s1_v2.urdf")
        self.assertEqual(document["evidence"]["collision"], "unavailable")
        self.assertEqual(document["action"]["type"], "unavailable")
        self.assertFalse(document["safety"]["real_robot_control_enabled"])

    def test_training_code_is_static_checked_and_versioned(self):
        baseline = self.client.get("/api/training-code/training_policy_baseline")
        self.assertEqual(baseline.status_code, 200)
        blocked = self.client.post("/api/training-code/validate", json={"content": "import subprocess\ndef train(env, seed): pass"})
        self.assertFalse(blocked.json()["valid"])
        saved = self.client.post("/api/training-code", json={"document": {"code_id": "student-policy", "content": "def train(env, seed):\n    return seed\n", "scene_ref": {"scene_id": "scene_mock_reach_lab", "revision": 1}}})
        self.assertEqual(saved.status_code, 201)
        self.assertEqual(saved.json()["revision"], 1)


if __name__ == "__main__":
    unittest.main()
