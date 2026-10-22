import json
import os
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from robotics.sandbox import ContainerResult, GPU_SMOKE_TEMPLATE, RemoteWorkerBackend
from scripts.sandbox_worker import WorkerHandler


class RecordingBackend:
    def __init__(self):
        self.arguments = []

    def run(self, job_id, template, arguments, timeout_seconds):
        self.arguments = list(arguments)
        return ContainerResult(0, "NVIDIA GeForce RTX 4080\n", "")

    def stop(self, container_name):
        pass


class SandboxWorkerTest(unittest.TestCase):
    def setUp(self):
        self.backend = RecordingBackend()
        WorkerHandler.backend = self.backend
        WorkerHandler.token = "test-worker-token"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), WorkerHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = "http://127.0.0.1:%s" % self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    def request(self, payload, token="test-worker-token"):
        return urllib.request.urlopen(urllib.request.Request(
            self.base_url + "/v1/run",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
            method="POST",
        ))

    def test_requires_authorization(self):
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request({"job_id": "sandbox-0123456789ab", "template_id": "gpu-smoke"}, token="wrong")
        self.assertEqual(context.exception.code, 401)

    def test_rejects_client_supplied_container_fields(self):
        with self.assertRaises(urllib.error.HTTPError) as context:
            self.request({
                "job_id": "sandbox-0123456789ab",
                "template_id": "gpu-smoke",
                "image": "ubuntu:latest",
            })
        self.assertEqual(context.exception.code, 422)
        self.assertEqual(self.backend.arguments, [])

    def test_valid_request_builds_fixed_restricted_arguments(self):
        with self.request({"job_id": "sandbox-0123456789ab", "template_id": "gpu-smoke"}) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("--read-only", self.backend.arguments)
        self.assertIn("none", self.backend.arguments)
        self.assertEqual(self.backend.arguments[self.backend.arguments.index("--gpus") + 1], "device=0")
        self.assertNotIn("-v", self.backend.arguments)
        self.assertNotIn("/var/run/docker.sock", self.backend.arguments)

    def test_training_template_is_worker_allowlisted(self):
        with self.request({"job_id": "sandbox-0123456789ab", "template_id": "gpu-pytorch-train"}) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime", self.backend.arguments)
        self.assertNotIn("-v", self.backend.arguments)

    def test_remote_backend_round_trip_uses_worker_protocol(self):
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1"}, clear=True):
            backend = RemoteWorkerBackend(self.base_url, "test-worker-token")
            result = backend.run(
                "sandbox-0123456789ab",
                GPU_SMOKE_TEMPLATE,
                ["client-arguments-are-not-forwarded"],
                30,
            )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("RTX 4080", result.stdout)
        self.assertNotIn("client-arguments-are-not-forwarded", self.backend.arguments)

    def test_remote_backend_loads_token_from_restricted_file(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as token_file:
            token_file.write("test-worker-token")
            token_file.flush()
            os.chmod(token_file.name, 0o600)
            environment = {
                "STUDIO_SANDBOX_WORKER_URL": self.base_url,
                "STUDIO_SANDBOX_WORKER_TOKEN_FILE": token_file.name,
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                backend = RemoteWorkerBackend.from_environment()
        self.assertIsNotNone(backend)
        self.assertEqual(backend.token, "test-worker-token")


if __name__ == "__main__":
    unittest.main()