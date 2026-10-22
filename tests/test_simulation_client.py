import json
import unittest
from datetime import datetime, timezone

import httpx

from robotics.simulation import (
    InputEventRequest,
    SimulationClient,
    SimulationClientClosedError,
    SimulationConfigurationError,
    SimulationNetworkError,
    SimulationRejectedError,
    SimulationResponseError,
    SimulationServerError,
    SimulationTaskStatus,
    SimulationTimeoutError,
    StreamSessionStatus,
)


def task_snapshot(
    task_id="task-1", run_id="run-1", project_id="project-1", config=None
):
    timestamp = datetime(2026, 8, 27, 10, 0, tzinfo=timezone.utc).isoformat()
    return {
        "task_id": task_id,
        "run_id": run_id,
        "project_id": project_id,
        "status": "queued",
        "created_at": timestamp,
        "updated_at": timestamp,
        "config": {} if config is None else config,
    }


def stream_descriptor(**changes):
    value = {
        "stream_id": "stream-opaque-1",
        "run_id": "run-1",
        "task_id": "task-1",
        "camera_id": "overview",
        "status": "READY",
        "webrtc_url": "https://media.example/stream-opaque-1",
        "whep_url": "https://media.example/stream-opaque-1/whep",
        "width": 640,
        "height": 480,
        "fps": 15,
    }
    value.update(changes)
    return value


class TrackingTransport(httpx.BaseTransport):
    def __init__(self):
        self.closed = False

    def handle_request(self, request):
        return httpx.Response(200, json=task_snapshot(), request=request)

    def close(self):
        self.closed = True


class SimulationClientTests(unittest.TestCase):
    base_url = "https://simulation.internal"

    def client(self, handler, token="test-service-token"):
        return SimulationClient(
            self.base_url,
            timeout=0.5,
            service_token=token,
            transport=httpx.MockTransport(handler),
        )

    def test_create_sends_method_path_json_and_deep_copies_input(self):
        config = {"nested": {"seed": 7}}

        def handler(request):
            self.assertEqual("POST", request.method)
            self.assertEqual("/simulation/tasks", request.url.path)
            self.assertEqual(
                {"run_id": "run-1", "project_id": "project-1", "config": config},
                json.loads(request.content),
            )
            config["nested"]["seed"] = 99
            return httpx.Response(
                202,
                json=task_snapshot(config={"seed": 7}),
            )

        with self.client(handler) as client:
            task = client.create_task("run-1", "project-1", config)
        self.assertEqual("task-1", task.task_id)
        self.assertEqual(SimulationTaskStatus.QUEUED, task.status)
        copied = task.to_dict()
        copied["config"]["seed"] = 101
        self.assertEqual(7, task.config["seed"])

    def test_get_and_stop_use_the_task_id(self):
        seen = []

        def handler(request):
            seen.append((request.method, request.url.path, request.content))
            return httpx.Response(200, json=task_snapshot(task_id="task-42"))

        with self.client(handler) as client:
            self.assertEqual("task-42", client.get_task("task-42").task_id)
            self.assertEqual("task-42", client.stop_task("task-42").task_id)
        self.assertEqual(
            [("GET", "/simulation/tasks/task-42", b""),
             ("POST", "/simulation/tasks/task-42/stop", b"")],
            seen,
        )

    def test_create_stream_session_uses_existing_client_auth_and_contract(self):
        seen = []

        def handler(request):
            seen.append(
                (
                    request.method,
                    request.url.path,
                    json.loads(request.content),
                    request.headers.get("Authorization"),
                )
            )
            return httpx.Response(200, json=stream_descriptor())

        with self.client(handler, token="stream-service-token") as client:
            descriptor = client.create_stream_session("task-1", "overview")
        self.assertEqual(descriptor.status, StreamSessionStatus.READY)
        self.assertEqual(descriptor.stream_id, "stream-opaque-1")
        self.assertEqual(
            seen,
            [(
                "POST",
                "/simulation/tasks/task-1/stream-session",
                {"camera_id": "overview"},
                "Bearer stream-service-token",
            )],
        )

    def test_input_event_requires_202_and_strict_matching_ack(self):
        event = InputEventRequest.from_mapping(
            {
                "event_id": "input-1",
                "run_id": "run-1",
                "task_id": "task-1",
                "type": "KEYBOARD",
                "occurred_at": "2026-08-28T00:00:00+00:00",
                "key": "W",
                "action": "pressed",
            }
        )
        seen = []

        def handler(request):
            seen.append(
                (
                    request.method,
                    request.url.path,
                    request.headers.get("Authorization"),
                    json.loads(request.content),
                )
            )
            return httpx.Response(
                202,
                json={"accepted": True, "event_id": "input-1", "duplicate": False},
            )

        with self.client(handler, token="input-service-token") as client:
            acknowledgement = client.send_input_event(event)
        self.assertTrue(acknowledgement.accepted)
        self.assertFalse(acknowledgement.duplicate)
        self.assertEqual(
            [
                (
                    "POST",
                    "/simulation/tasks/task-1/input-events",
                    "Bearer input-service-token",
                    event.to_dict(),
                )
            ],
            seen,
        )

        invalid = [
            (200, {"accepted": True, "event_id": "input-1", "duplicate": False}),
            (201, {"accepted": True, "event_id": "input-1", "duplicate": False}),
            (202, {"accepted": False, "event_id": "input-1", "duplicate": False}),
            (202, {"accepted": True, "event_id": "other", "duplicate": False}),
            (202, {"accepted": True, "event_id": "input-1", "duplicate": 0}),
            (202, {"accepted": True, "event_id": "input-1", "duplicate": False, "extra": True}),
        ]
        for status, payload in invalid:
            with self.subTest(status=status, payload=payload):
                with self.client(
                    lambda request, value=status, body=payload: httpx.Response(
                        value, json=body
                    )
                ) as client:
                    with self.assertRaises(SimulationResponseError):
                        client.send_input_event(event)

    def test_input_event_contract_rejects_invalid_fields_and_values(self):
        base = {
            "event_id": "input-1",
            "run_id": "run-1",
            "task_id": "task-1",
            "type": "KEYBOARD",
            "occurred_at": "2026-08-28T00:00:00+00:00",
            "key": "W",
            "action": "pressed",
        }
        invalid = [
            dict(base, unknown=True),
            dict(base, key="Escape"),
            dict(base, action="repeat"),
            dict(base, occurred_at="2026-08-28T08:00:00+08:00"),
            dict(base, event_id="../unsafe"),
            dict(base, type="MOUSE_MOVE", key=None, action=None, x=True, y=0.5),
            dict(base, type="MOUSE_MOVE", key=None, action=None, x=float("nan"), y=0.5),
            dict(base, type="MOUSE_MOVE", key=None, action=None, x=float("inf"), y=0.5),
            dict(base, type="MOUSE_MOVE", key=None, action=None, x=-0.1, y=0.5),
            dict(base, type="MOUSE_BUTTON", key=None, button="other", x=0.5, y=0.5),
        ]
        for payload in invalid:
            payload = {name: value for name, value in payload.items() if value is not None}
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                InputEventRequest.from_mapping(payload)

    def test_stream_accepts_all_six_descriptor_states(self):
        for status in StreamSessionStatus:
            with self.subTest(status=status):
                with self.client(
                    lambda request, value=status: httpx.Response(
                        200, json=stream_descriptor(status=value.value)
                    )
                ) as client:
                    self.assertEqual(
                        status, client.create_stream_session("task-1").status
                    )

    def test_stream_descriptor_rejects_unsafe_or_incomplete_responses(self):
        invalid = [
            stream_descriptor(stream_id="../derived-from-run"),
            stream_descriptor(status="UNKNOWN"),
            stream_descriptor(webrtc_url="rtsp://media.internal/stream"),
            stream_descriptor(whep_url="https://user:password@media.example/whep"),
            stream_descriptor(whep_url="https://media.example/whep?token=secret"),
            stream_descriptor(width=0),
            stream_descriptor(height=-1),
            stream_descriptor(fps=0),
            stream_descriptor(fps=float("inf")),
            dict(stream_descriptor(), rtsp_internal_url="rtsp://internal/stream"),
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.client(
                    lambda request, value=payload: httpx.Response(200, json=value)
                ) as client:
                    with self.assertRaises(SimulationResponseError):
                        client.create_stream_session("task-1")

    def test_stream_response_cannot_expose_service_token(self):
        token = "unique-stream-service-secret"
        payload = stream_descriptor(
            whep_url="https://media.example/unique-stream-service-secret/whep"
        )
        with self.client(
            lambda request: httpx.Response(200, json=payload), token=token
        ) as client:
            with self.assertRaises(SimulationResponseError) as caught:
                client.create_stream_session("task-1")
        self.assertNotIn(token, str(caught.exception))

    def test_stream_request_validation_and_status_handling(self):
        with self.client(
            lambda request: httpx.Response(200, json=stream_descriptor())
        ) as client:
            for task_id, camera_id in (
                ("../task", "overview"),
                ("task-1", "bad/camera"),
                ("task-1", "wrist"),
            ):
                with self.subTest(task_id=task_id, camera_id=camera_id):
                    with self.assertRaises(SimulationConfigurationError):
                        client.create_stream_session(task_id, camera_id)
        with self.client(
            lambda request: httpx.Response(201, json=stream_descriptor())
        ) as client:
            with self.assertRaises(SimulationResponseError):
                client.create_stream_session("task-1")
        with self.client(
            lambda request: httpx.Response(302, json=stream_descriptor())
        ) as client:
            with self.assertRaises(SimulationResponseError):
                client.create_stream_session("task-1")

    def test_stream_stable_error_envelopes_are_parsed_and_redacted(self):
        cases = [
            (400, "invalid_request", SimulationRejectedError),
            (400, "invalid_camera_id", SimulationRejectedError),
            (401, "unauthorized", SimulationRejectedError),
            (404, "task_not_found", SimulationRejectedError),
            (409, "stream_session_not_allowed", SimulationRejectedError),
            (503, "stream_not_ready", SimulationServerError),
        ]
        for status, code, error_type in cases:
            with self.subTest(status=status, code=code):
                with self.client(
                    lambda request, value=status, stable_code=code: httpx.Response(
                        value,
                        json={
                            "code": stable_code,
                            "message": "internal upstream message",
                        },
                    )
                ) as client:
                    with self.assertRaises(error_type) as caught:
                        client.create_stream_session("task-1")
                self.assertEqual(code, caught.exception.code)
                self.assertEqual(
                    "internal upstream message",
                    caught.exception.upstream_message,
                )
                self.assertNotIn("internal upstream message", str(caught.exception))

    def test_bearer_token_is_required_and_injected(self):
        headers = []

        def handler(request):
            headers.append(request.headers.get("Authorization"))
            return httpx.Response(200, json=task_snapshot())

        with self.client(handler, token="secret-token") as client:
            client.get_task("task-1")
        self.assertEqual(["Bearer secret-token"], headers)
        with self.assertRaises(SimulationConfigurationError):
            SimulationClient(self.base_url)

    def test_repeated_run_id_can_parse_the_same_task_id(self):
        def handler(request):
            return httpx.Response(
                202, json=task_snapshot(task_id="task-existing", run_id="run-same")
            )

        with self.client(handler) as client:
            first = client.create_task("run-same", "project-1", {"seed": 1})
            second = client.create_task("run-same", "project-1", {"seed": 1})
        self.assertEqual(first.task_id, second.task_id)

    def test_timeout_connection_4xx_and_5xx_are_distinct(self):
        cases = [
            (lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=request)),
             SimulationTimeoutError),
            (lambda request: (_ for _ in ()).throw(httpx.ConnectError("down", request=request)),
             SimulationNetworkError),
            (lambda request: httpx.Response(409, text="do not expose this body"),
             SimulationRejectedError),
            (lambda request: httpx.Response(503, text="do not expose this body"),
             SimulationServerError),
        ]
        for handler, expected in cases:
            with self.subTest(expected=expected.__name__):
                with self.client(handler) as client:
                    with self.assertRaises(expected):
                        client.get_task("task-1")

    def test_non_json_missing_task_id_and_invalid_fields_are_rejected(self):
        payloads = [
            httpx.Response(200, text="not-json"),
            httpx.Response(200, json={"status": "queued"}),
            httpx.Response(200, json={"task_id": "../unsafe"}),
            httpx.Response(200, json=dict(task_snapshot(), status="unknown")),
            httpx.Response(200, json=dict(task_snapshot(), unexpected=True)),
            httpx.Response(
                200,
                json=dict(
                    task_snapshot(), created_at="2026-08-27T18:00:00+08:00"
                ),
            ),
        ]
        for response in payloads:
            with self.subTest(body=response.content):
                with self.client(lambda request, item=response: item) as client:
                    with self.assertRaises(SimulationResponseError):
                        client.get_task("task-1")

    def test_input_validation_rejects_unsafe_identifiers(self):
        with self.client(lambda request: httpx.Response(200, json=task_snapshot())) as client:
            with self.assertRaises(SimulationConfigurationError):
                client.create_task("../run", "project-1")
            with self.assertRaises(SimulationConfigurationError):
                client.create_task("run-1", "project-1", [])
            with self.assertRaises(SimulationConfigurationError):
                client.get_task("task/one")

    def test_close_releases_transport_and_prevents_reuse(self):
        transport = TrackingTransport()
        client = SimulationClient(
            self.base_url, service_token="test-service-token", transport=transport
        )
        client.get_task("task-1")
        client.close()
        client.close()
        self.assertTrue(client.is_closed)
        self.assertTrue(transport.closed)
        with self.assertRaises(SimulationClientClosedError):
            client.get_task("task-1")

    def test_errors_do_not_leak_token_url_credentials_or_response_body(self):
        token = "very-secret-token"
        response_body = "sensitive upstream diagnostic"
        with self.client(
            lambda request: httpx.Response(400, text=response_body), token=token
        ) as client:
            with self.assertRaises(SimulationRejectedError) as caught:
                client.get_task("task-1")
        rendered = repr(caught.exception)
        self.assertNotIn(token, rendered)
        self.assertNotIn(self.base_url, rendered)
        self.assertNotIn(response_body, rendered)
        for invalid in (
            "https://user:password@simulation.internal",
            "https://simulation.internal/?token=secret",
            "https://simulation.internal/prefix",
        ):
            with self.assertRaises(SimulationConfigurationError) as invalid_url:
                SimulationClient(invalid, service_token="test-service-token")
            self.assertNotIn("password", str(invalid_url.exception))
            self.assertNotIn("secret", str(invalid_url.exception))


if __name__ == "__main__":
    unittest.main()
