from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import socket
import unittest
from urllib.error import HTTPError

from simulation_core import (
    BackendMessageFactory, BackendRequestRejected,
    BackendTemporarilyUnavailable, CallbackDispatcher, DeliveryStatus,
    FakeScheduler, HttpBackendClient, InMemoryCallbackOutbox,
    InMemoryTaskRepository, MockBackendServer, MockBackendState,
    RedisCallbackOutbox, SimulationService, TaskStatus, TrainingTaskResult,
    run_polling_loop,
)


STARTED = "2026-08-23T10:00:00+00:00"
ENDED = "2026-08-23T10:01:00+00:00"


def event(factory=None, event_type="state"):
    factory = factory or BackendMessageFactory()
    payloads = {
        "state": {"previous_status": "initializing", "status": "running"},
        "metric": {"step": 16, "metrics": {
            "update_index": 1, "average_reward": 0.2,
        }},
        "heartbeat": {"status": "running", "training_steps": 16},
        "error": {"code": "TRAINING_FAILED", "message": "failed", "retryable": False},
    }
    return factory.event(event_type, "run http", "task-http", payloads[event_type])


def callback(factory=None):
    factory = factory or BackendMessageFactory()
    return factory.terminal_callback(
        "run-http", "task-http", status="completed",
        started_at=STARTED, ended_at=ENDED, training_steps=16,
        update_steps=1, stop_reason=None, metrics={"average_reward": 0.2},
        model={"type": "checkpoint_best", "path": "checkpoints/model_best.pt"},
        artifacts=[{"type": "manifest", "path": "manifest.json"}], error=None,
    )


class FakeResponse:
    def __init__(self, status=202):
        self.status = status

    def getcode(self):
        return self.status

    def read(self, size=-1):
        return b'{"accepted":true}'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class CapturingOpener:
    def __init__(self, action=None):
        self.action = action
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        if isinstance(self.action, BaseException):
            raise self.action
        return self.action or FakeResponse()


class HttpBackendClientTests(unittest.TestCase):
    def test_event_url_headers_json_and_identity_are_unchanged(self):
        opener = CapturingOpener()
        client = HttpBackendClient(
            "http://backend.example/root/", 3.5, service_token="secret",
            headers={"X-Service": "simulation"}, opener=opener,
        )
        original = event(event_type="metric")
        client.publish_event(original)
        request, timeout = opener.calls[0]
        self.assertEqual(
            "http://backend.example/root/api/internal/v1/runs/run%20http/events",
            request.full_url,
        )
        self.assertEqual(3.5, timeout)
        self.assertEqual("application/json", request.get_header("Content-type"))
        self.assertEqual("Bearer secret", request.get_header("Authorization"))
        self.assertEqual("simulation", request.get_header("X-service"))
        self.assertEqual(original.to_dict(), json.loads(request.data))
        body = json.loads(request.data)
        self.assertEqual(1, body["payload"]["metrics"]["update_index"])
        self.assertEqual(16, body["payload"]["step"])
        self.assertNotIn("update_index", body["payload"])

    def test_callback_posts_existing_model_directly(self):
        opener = CapturingOpener()
        client = HttpBackendClient("https://backend.example", opener=opener)
        original = callback()
        client.publish_terminal_callback(original)
        request, _ = opener.calls[0]
        self.assertEqual(
            "https://backend.example/api/internal/v1/runs/run-http/callback",
            request.full_url,
        )
        self.assertEqual(original.to_dict(), json.loads(request.data))

    def test_4xx_is_explicit_rejection(self):
        error = HTTPError("http://backend", 422, "bad", {}, BytesIO(b"{}"))
        client = HttpBackendClient("http://backend", opener=CapturingOpener(error))
        with self.assertRaises(BackendRequestRejected) as raised:
            client.publish_event(event())
        self.assertEqual(422, raised.exception.status_code)
        self.assertFalse(raised.exception.retryable)

    def test_5xx_and_timeout_are_temporary_failures(self):
        server_error = HTTPError("http://backend", 503, "down", {}, BytesIO(b"{}"))
        with self.assertRaises(BackendTemporarilyUnavailable) as raised:
            HttpBackendClient("http://backend", opener=CapturingOpener(server_error)).publish_event(event())
        self.assertEqual(503, raised.exception.status_code)
        self.assertTrue(raised.exception.retryable)
        with self.assertRaises(BackendTemporarilyUnavailable):
            HttpBackendClient("http://backend", opener=CapturingOpener(socket.timeout("late"))).publish_event(event())

    def test_environment_configuration_is_optional_and_validated(self):
        self.assertIsNone(HttpBackendClient.from_env(False, {}))
        client = HttpBackendClient.from_env(True, {
            "SIMULATION_BACKEND_BASE_URL": "http://backend.example",
            "SIMULATION_BACKEND_REQUEST_TIMEOUT": "2.5",
            "SIMULATION_BACKEND_SERVICE_TOKEN": "token",
        })
        self.assertEqual(2.5, client.request_timeout)
        self.assertEqual("Bearer token", client.headers["Authorization"])
        with self.assertRaises(ValueError):
            HttpBackendClient.from_env(True, {})


class RedisFake:
    def __init__(self):
        self.values, self.sets = {}, {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value.encode() if isinstance(value, str) else value

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value.encode() if isinstance(value, str) else value)

    def smembers(self, key):
        return self.sets.get(key, set())

    def eval(self, script, count, task_key, record_key, index_key, event_id, payload):
        if self.get(task_key) is not None:
            return self.get(task_key)
        self.set(task_key, event_id)
        self.set(record_key, payload)
        self.sadd(index_key, event_id)
        return event_id


class HttpIntegrationTests(unittest.TestCase):
    def test_all_event_types_reach_mock_with_exact_json(self):
        with MockBackendServer() as server:
            client = HttpBackendClient(server.base_url, 1)
            factory = BackendMessageFactory()
            sent = [event(factory, name) for name in ("state", "metric", "heartbeat", "error")]
            for item in sent:
                client.publish_event(item)
            self.assertEqual([item.to_dict() for item in sent], server.state.events)
            for original, received in zip(sent, server.state.events):
                self.assertEqual(original.event_id, received["event_id"])
                self.assertEqual(original.sequence, received["sequence"])
                self.assertEqual(original.task_id, received["task_id"])

    def test_redis_outbox_http_500_retry_success_preserves_callback(self):
        state = MockBackendState(callback_failures=1)
        with MockBackendServer(state) as server:
            outbox = RedisCallbackOutbox(RedisFake(), "http-test")
            original = callback()
            now = datetime.now(timezone.utc)
            record = outbox.put_if_absent(original, now)
            dispatcher = CallbackDispatcher(
                outbox, HttpBackendClient(server.base_url, 1),
                max_attempts=3, base_delay_seconds=0,
            )
            first = dispatcher.dispatch_event(record.event_id, now)
            second = dispatcher.dispatch_event(record.event_id, now + timedelta(microseconds=1))
            self.assertEqual(DeliveryStatus.PENDING, first.delivery_status)
            self.assertEqual(1, first.attempt_count)
            self.assertEqual(DeliveryStatus.DELIVERED, second.delivery_status)
            self.assertEqual(2, second.attempt_count)
            self.assertEqual([original.to_dict(), original.to_dict()], state.callback_attempts)

    def test_continuous_http_failure_exhausts_without_changing_completed_task(self):
        state = MockBackendState(callback_failures=10)
        with MockBackendServer(state) as server:
            backend = HttpBackendClient(server.base_url, 1)
            outbox = InMemoryCallbackOutbox()
            dispatcher = CallbackDispatcher(outbox, backend, 2, 0)
            repository = InMemoryTaskRepository()
            service = SimulationService(
                repository, FakeScheduler(), backend,
                task_id_factory=lambda: "task-http", callback_outbox=outbox,
                callback_dispatcher=dispatcher,
            )
            task = service.create_task("run-http", "project-http", {})
            for status in (TaskStatus.INITIALIZING, TaskStatus.RUNNING, TaskStatus.FINALIZING, TaskStatus.COMPLETED):
                service.transition_task_status(task.task_id, status)
            training_result = TrainingTaskResult(
                task_id=task.task_id, run_id=task.run_id,
                status=TaskStatus.COMPLETED, training_steps=1, update_steps=1,
                metrics_path="/local/metrics", checkpoint_path="/local/latest",
                final_metrics_summary={"average_reward": 0.1},
                manifest_path="/local/manifest", best_checkpoint_path="/local/best",
                artifacts=(
                    {"type": "checkpoint_best", "path": "checkpoints/model_best.pt"},
                    {"type": "manifest", "path": "manifest.json",
                     "size_bytes": 123, "sha256": "a" * 64},
                ), started_at=STARTED, ended_at=ENDED,
            )
            first = service.enqueue_terminal_callback(task.task_id, training_result)
            second = dispatcher.dispatch_event(first.event_id)
            self.assertEqual(DeliveryStatus.EXHAUSTED, second.delivery_status)
            self.assertEqual(TaskStatus.COMPLETED, service.get_task(task.task_id).status)

    def test_event_500_does_not_change_task_or_stop_metric_reporting(self):
        state = MockBackendState(event_failures=2)
        with MockBackendServer(state) as server:
            repository = InMemoryTaskRepository()
            service = SimulationService(
                repository, FakeScheduler(), HttpBackendClient(server.base_url, 1),
                task_id_factory=lambda: "task-http",
            )
            task = service.create_task("run-http", "project-http", {})
            updated = service.transition_task_status(task.task_id, TaskStatus.INITIALIZING)
            self.assertEqual(TaskStatus.INITIALIZING, updated.status)
            self.assertEqual(TaskStatus.INITIALIZING, service.get_task(task.task_id).status)
            service.publish_metric_record(task.task_id, {
                "update_index": 1,
                "training_steps": 16,
                "average_reward": 0.2,
                "episode_length": 16.0,
                "episodes": 0.0,
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 7.4,
            })
            self.assertEqual(TaskStatus.INITIALIZING, service.get_task(task.task_id).status)
            self.assertEqual(2, len(state.event_attempts))


class DispatcherRuntimeTests(unittest.TestCase):
    def test_polling_loop_dispatches_and_stops_cleanly(self):
        calls, sleeps = [], []

        class Dispatcher:
            def dispatch_once(self):
                calls.append("dispatch")

        run_polling_loop(
            Dispatcher(), polling_interval=0.25,
            stop_requested=lambda: len(calls) >= 2,
            sleep=sleeps.append,
        )
        self.assertEqual(["dispatch", "dispatch"], calls)
        self.assertEqual([0.25, 0.25], sleeps)


if __name__ == "__main__":
    unittest.main()
