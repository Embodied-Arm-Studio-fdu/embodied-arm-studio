import base64
import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from apps.api.main import create_app
from robotics.sandbox import ContainerResult, SandboxRunner


class RecordingBackend:
    def __init__(self):
        self.arguments = []
        self.stopped = []

    def run(self, job_id, template, arguments, timeout_seconds):
        self.arguments = list(arguments)
        if template.template_id == "gpu-pytorch-train":
            checkpoint = json.dumps({"framework": "pytorch", "final_loss": 0.001}).encode("utf-8")
            encoded = base64.b64encode(checkpoint).decode("ascii")
            return ContainerResult(0, "STUDIO_METRIC {\"epoch\":80,\"train_loss\":0.001}\nSTUDIO_ARTIFACT checkpoint.json %s\n" % encoded, "")
        return ContainerResult(0, "NVIDIA GeForce RTX 4080, 16376 MiB\n", "")

    def stop(self, container_name):
        self.stopped.append(container_name)


class SandboxRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.backend = RecordingBackend()
        self.runner = SandboxRunner(Path(self.temp_dir.name), self.backend)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_gpu_template_has_fixed_isolation_and_resource_limits(self):
        job = self.runner.submit("gpu-smoke")
        deadline = time.monotonic() + 1
        while self.runner.get(job.job_id).status.value in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.005)

        record = self.runner.get(job.job_id)
        self.assertEqual(record.status.value, "completed")
        arguments = self.backend.arguments
        self.assertIn("none", arguments)
        self.assertIn("--read-only", arguments)
        self.assertIn("no-new-privileges", arguments)
        self.assertIn("--cap-drop", arguments)
        self.assertIn("--pids-limit", arguments)
        self.assertIn("--memory", arguments)
        self.assertEqual(arguments[arguments.index("--gpus") + 1], "device=0")
        self.assertNotIn("--privileged", arguments)
        self.assertNotIn("/var/run/docker.sock", arguments)
        self.assertNotIn("-v", arguments)
        self.assertIn("RTX 4080", self.runner.logs(job.job_id)["stdout"])
        audit = self.runner.audit(job.job_id)
        self.assertEqual(audit["policy"]["network"], "none")
        self.assertFalse(audit["policy"]["docker_socket"])
        self.assertTrue((Path(self.temp_dir.name) / job.job_id / "job.json").exists())

    def test_unknown_template_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知 Sandbox 模板"):
            self.runner.submit("arbitrary-image-and-command")

    def test_gpu_training_template_persists_verified_checkpoint(self):
        job = self.runner.submit("gpu-pytorch-train")
        deadline = time.monotonic() + 1
        while self.runner.get(job.job_id).status.value in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(self.runner.get(job.job_id).status.value, "completed")
        self.assertIn("pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime", self.backend.arguments)
        artifacts = self.runner.artifacts(job.job_id)
        self.assertEqual(artifacts[0]["name"], "checkpoint.json")
        self.assertEqual(len(artifacts[0]["sha256"]), 64)
        checkpoint_path = Path(self.temp_dir.name) / job.job_id / "checkpoint.json"
        self.assertEqual(json.loads(checkpoint_path.read_text(encoding="utf-8"))["framework"], "pytorch")

    def test_api_only_accepts_templates_and_exposes_restricted_logs(self):
        client = TestClient(create_app(
            Path(self.temp_dir.name) / "runs",
            tick_seconds=0,
            project_root=Path(self.temp_dir.name) / "projects",
            sandbox_backend=self.backend,
        ))
        templates = client.get("/api/sandbox/templates")
        self.assertEqual(templates.status_code, 200)
        self.assertEqual({item["template_id"] for item in templates.json()}, {"gpu-smoke", "gpu-pytorch-train"})

        rejected = client.post("/api/sandbox/jobs", json={
            "template_id": "gpu-smoke",
            "image": "ubuntu:latest",
            "command": ["sh"],
        })
        self.assertEqual(rejected.status_code, 422)

        response = client.post("/api/sandbox/jobs", json={"template_id": "gpu-smoke"})
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 1
        while client.get("/api/sandbox/jobs/" + job_id).json()["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.005)
        logs = client.get("/api/sandbox/jobs/%s/logs" % job_id)
        self.assertEqual(logs.status_code, 200)
        self.assertIn("RTX 4080", logs.json()["stdout"])
        self.assertEqual(client.get("/api/sandbox/jobs").json()[0]["job_id"], job_id)
        audit = client.get("/api/sandbox/jobs/%s/audit" % job_id)
        self.assertEqual(audit.status_code, 200)
        self.assertEqual(audit.json()["policy"]["capabilities"], "drop-all")
        self.assertEqual(client.get("/api/sandbox/jobs/missing").status_code, 404)

        training = client.post("/api/sandbox/jobs", json={"template_id": "gpu-pytorch-train"})
        training_job_id = training.json()["job_id"]
        deadline = time.monotonic() + 1
        while client.get("/api/sandbox/jobs/" + training_job_id).json()["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.005)
        artifact = client.get("/api/sandbox/jobs/%s/artifacts/checkpoint.json" % training_job_id)
        self.assertEqual(artifact.status_code, 200)
        self.assertEqual(artifact.json()["framework"], "pytorch")
        self.assertEqual(client.get("/api/sandbox/jobs/%s/artifacts/other.json" % training_job_id).status_code, 422)


if __name__ == "__main__":
    unittest.main()