import asyncio
from dataclasses import replace
from itertools import count
import unittest

import httpx

from simulation_core import (
    FakeBackendClient,
    FakeScheduler,
    InMemoryTaskRepository,
    SimulationService,
    RecordingInputEventSink,
    create_http_app,
    StreamSessionManager,
    TaskStatus,
    TrainingTaskConfig,
)


TOKEN = "test-service-token"
AUTH = {"Authorization": "Bearer " + TOKEN}


def push_config(**overrides):
    values = TrainingTaskConfig(
        task_type="training", environment_type="s1_push", seed=7,
        rollout_steps=4, num_updates=1, checkpoint_every=1,
        episode_steps=8, physics_steps_per_action=1, batch_size=4,
    ).to_dict()
    values.update(overrides)
    return values


class ApiClient:
    def __init__(self, app):
        self.loop = asyncio.new_event_loop()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        )

    def get(self, *args, **kwargs):
        return self.loop.run_until_complete(self.client.get(*args, **kwargs))

    def post(self, *args, **kwargs):
        return self.loop.run_until_complete(self.client.post(*args, **kwargs))

    def close(self):
        self.loop.run_until_complete(self.client.aclose())
        self.loop.close()


class SimulationHttpApiTests(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryTaskRepository()
        self.scheduler = FakeScheduler()
        self.service = SimulationService(
            self.repository,
            self.scheduler,
            FakeBackendClient(),
            task_id_factory=lambda: "task-http-1",
        )
        self.client = ApiClient(create_http_app(self.service, TOKEN))

    def tearDown(self):
        self.client.close()

    def create(self):
        return self.client.post(
            "/simulation/tasks",
            headers=AUTH,
            json={"run_id": "run-http-1", "project_id": "project-1", "config": push_config()},
        )

    def test_create_and_same_run_id_are_idempotent(self):
        first = self.create()
        second = self.create()
        self.assertEqual(202, first.status_code)
        self.assertEqual(first.json(), second.json())
        self.assertEqual("queued", first.json()["status"])
        self.assertEqual(["task-http-1"], self.scheduler.submitted_task_ids)

    def test_get_task(self):
        task_id = self.create().json()["task_id"]
        response = self.client.get("/simulation/tasks/" + task_id, headers=AUTH)
        self.assertEqual(200, response.status_code)
        self.assertEqual("run-http-1", response.json()["run_id"])
        self.assertIn("created_at", response.json())

    def test_stop_task_is_idempotent(self):
        task_id = self.create().json()["task_id"]
        first = self.client.post("/simulation/tasks/%s/stop" % task_id, headers=AUTH)
        second = self.client.post("/simulation/tasks/%s/stop" % task_id, headers=AUTH)
        self.assertEqual(200, first.status_code)
        self.assertEqual("stopped", first.json()["status"])
        self.assertEqual(first.json(), second.json())

    def test_task_not_found(self):
        response = self.client.get("/simulation/tasks/missing", headers=AUTH)
        self.assertEqual(404, response.status_code)
        self.assertEqual(
            {"code": "task_not_found", "message": "simulation task not found"},
            response.json(),
        )

    def test_invalid_request(self):
        cases = [
            {"run_id": "../unsafe", "project_id": "project-1", "config": {}},
            {"run_id": "run-1", "project_id": "project-1", "config": []},
            {"run_id": "run-1", "project_id": "project-1", "config": {}, "extra": True},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                response = self.client.post("/simulation/tasks", headers=AUTH, json=payload)
                self.assertEqual(422, response.status_code)
                self.assertEqual("invalid_request", response.json()["code"])

    def test_missing_and_invalid_token(self):
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            with self.subTest(headers=headers):
                response = self.client.get("/simulation/tasks/task-1", headers=headers)
                self.assertEqual(401, response.status_code)
                self.assertEqual("unauthorized", response.json()["code"])
                self.assertEqual("Bearer", response.headers["www-authenticate"])

    def test_service_internal_exception_is_sanitized(self):
        class FailingService:
            def get_task(self, task_id):
                raise RuntimeError("sensitive repository diagnostic")

        client = ApiClient(create_http_app(FailingService(), TOKEN))
        try:
            response = client.get("/simulation/tasks/task-1", headers=AUTH)
        finally:
            client.close()
        self.assertEqual(500, response.status_code)
        self.assertEqual(
            {"code": "internal_error", "message": "simulation service internal error"},
            response.json(),
        )
        self.assertNotIn("sensitive", response.text)


class StreamSessionHttpApiTests(unittest.TestCase):
    class Runtime:
        def __init__(self, session, owner, fail=False):
            self.session = session
            self.owner = owner
            self.fail = fail

        def start(self):
            self.owner.starts.append(self.session.stream_id)
            if self.fail:
                raise RuntimeError("video component unavailable")

        def stop(self):
            self.owner.stops.append(self.session.stream_id)

    def setUp(self):
        self.repository = InMemoryTaskRepository()
        self.service = SimulationService(
            self.repository, FakeScheduler(), FakeBackendClient(),
            task_id_factory=lambda: "task-stream-1",
        )
        self.starts, self.stops = [], []
        self.runtime_fail = False
        self.manager = StreamSessionManager(
            lambda session: self.Runtime(session, self, self.runtime_fail)
        )
        self.client = ApiClient(
            create_http_app(self.service, TOKEN, stream_manager=self.manager)
        )

    def tearDown(self):
        self.client.close()

    def create_task(self):
        response = self.client.post(
            "/simulation/tasks", headers=AUTH,
            json={
                "run_id": "run-stream-1", "project_id": "project-1",
                "config": push_config(video_enabled=True),
            },
        )
        return response.json()["task_id"]

    def mark_running(self, task_id):
        self.service.transition_task_status(task_id, TaskStatus.INITIALIZING)
        self.service.transition_task_status(task_id, TaskStatus.RUNNING)

    def post_stream(self, task_id="task-stream-1", **kwargs):
        return self.client.post(
            "/simulation/tasks/%s/stream-session" % task_id,
            headers=kwargs.pop("headers", AUTH), **kwargs
        )

    def test_authenticated_overview_response_is_complete_and_idempotent(self):
        task_id = self.create_task()
        self.mark_running(task_id)
        before = self.service.get_task(task_id).status
        first = self.post_stream(task_id, json={"camera_id": "overview"})
        second = self.post_stream(task_id, json={"camera_id": "overview"})
        expected = {
            "stream_id", "run_id", "task_id", "camera_id", "status",
            "webrtc_url", "whep_url", "width", "height", "fps",
        }
        self.assertEqual(200, first.status_code)
        self.assertEqual(200, second.status_code)
        self.assertEqual(expected, set(first.json()))
        self.assertEqual(first.json()["stream_id"], second.json()["stream_id"])
        self.assertEqual(1, len(self.manager.list_sessions_for_run("run-stream-1")))
        self.assertEqual(1, len(self.starts))
        self.assertEqual(before, self.service.get_task(task_id).status)
        self.assertEqual([], self.stops)

    def test_missing_body_uses_compatibility_default(self):
        task_id = self.create_task()
        self.mark_running(task_id)
        response = self.post_stream(task_id)
        self.assertEqual(200, response.status_code)
        self.assertEqual("overview", response.json()["camera_id"])

    def test_invalid_token_is_rejected(self):
        response = self.post_stream(headers={"Authorization": "Bearer wrong"},
                                    json={"camera_id": "overview"})
        self.assertEqual(401, response.status_code)
        self.assertEqual("unauthorized", response.json()["code"])

    def test_missing_task_is_404(self):
        response = self.post_stream("missing", json={"camera_id": "overview"})
        self.assertEqual(404, response.status_code)
        self.assertEqual("task_not_found", response.json()["code"])

    def test_non_running_task_is_conflict(self):
        task_id = self.create_task()
        response = self.post_stream(task_id, json={"camera_id": "overview"})
        self.assertEqual(409, response.status_code)
        self.assertEqual("stream_session_not_allowed", response.json()["code"])
        self.assertEqual([], self.starts)

    def test_video_start_failure_is_not_ready_and_does_not_stop_training(self):
        task_id = self.create_task()
        self.mark_running(task_id)
        self.runtime_fail = True
        response = self.post_stream(task_id, json={"camera_id": "overview"})
        self.assertEqual(503, response.status_code)
        self.assertEqual("stream_not_ready", response.json()["code"])
        self.assertEqual(TaskStatus.RUNNING, self.service.get_task(task_id).status)
        self.assertEqual([], self.stops)

    def test_bad_json_and_camera_are_400(self):
        task_id = self.create_task()
        self.mark_running(task_id)
        bad_json = self.post_stream(
            task_id, content="{", headers=dict(AUTH, **{"Content-Type": "application/json"})
        )
        bad_camera = self.post_stream(task_id, json={"camera_id": "side"})
        self.assertEqual((400, "invalid_request"),
                         (bad_json.status_code, bad_json.json()["code"]))
        self.assertEqual((400, "invalid_camera_id"),
                         (bad_camera.status_code, bad_camera.json()["code"]))


class InputEventsHttpApiTests(unittest.TestCase):
    def setUp(self):
        self.repository = InMemoryTaskRepository()
        self.scheduler = FakeScheduler()
        task_numbers = count(1)
        self.service = SimulationService(
            self.repository,
            self.scheduler,
            FakeBackendClient(),
            task_id_factory=lambda: "task-input-%d" % next(task_numbers),
        )
        self.sink = RecordingInputEventSink(max_events_per_task=3)
        self.client = ApiClient(
            create_http_app(self.service, TOKEN, input_sink=self.sink)
        )

    def tearDown(self):
        self.client.close()

    def create_task(self, run_id="run-input-1", status=TaskStatus.RUNNING):
        response = self.client.post(
            "/simulation/tasks",
            headers=AUTH,
            json={"run_id": run_id, "project_id": "project-1", "config": push_config()},
        )
        task = self.service.get_task(response.json()["task_id"])
        if status != task.status:
            self.repository.update(replace(task, status=status))
        return task.task_id

    @staticmethod
    def payload(task_id, run_id="run-input-1", kind="KEYBOARD", **values):
        body = {
            "event_id": "event-1",
            "run_id": run_id,
            "task_id": task_id,
            "type": kind,
            "occurred_at": "2026-08-28T00:00:00+00:00",
        }
        body.update(values)
        return body

    def post_input(self, task_id, body, headers=AUTH):
        return self.client.post(
            "/simulation/tasks/%s/input-events" % task_id,
            headers=headers,
            json=body,
        )

    def test_three_event_types_are_accepted_by_asgi_http_smoke(self):
        task_id = self.create_task()
        before = self.service.get_task(task_id)
        payloads = [
            self.payload(task_id, event_id="keyboard-1", key="W", action="pressed"),
            self.payload(task_id, event_id="move-1", kind="MOUSE_MOVE", x=0.0, y=1.0),
            self.payload(task_id, event_id="button-1", kind="MOUSE_BUTTON", button="left", action="released", x=0.4, y=0.6),
        ]
        for body in payloads:
            with self.subTest(kind=body["type"]):
                response = self.post_input(task_id, body)
                self.assertEqual(202, response.status_code)
                self.assertEqual(
                    {"accepted": True, "event_id": body["event_id"], "duplicate": False},
                    response.json(),
                )
        self.assertEqual(3, len(self.sink.events_for_task(task_id)))
        self.assertEqual(before, self.service.get_task(task_id))
        self.assertEqual([], self.scheduler.cancelled_task_ids)

    def test_invalid_fields_keys_actions_and_buttons_are_400(self):
        task_id = self.create_task()
        cases = [
            dict(self.payload(task_id, key="W", action="pressed"), unknown=True),
            self.payload(task_id, key="Escape", action="pressed"),
            self.payload(task_id, key="W", action="repeat"),
            self.payload(task_id, kind="MOUSE_BUTTON", button="other", action="pressed", x=.5, y=.5),
            self.payload(task_id, kind="MOUSE_MOVE", x=.5, y=.5, action="pressed"),
        ]
        for body in cases:
            with self.subTest(body=body):
                response = self.post_input(task_id, body)
                self.assertEqual((400, "invalid_request"), (response.status_code, response.json()["code"]))

    def test_bad_json_missing_fields_and_invalid_ids_are_400(self):
        task_id = self.create_task()
        bad_json = self.client.post(
            "/simulation/tasks/%s/input-events" % task_id,
            headers=dict(AUTH, **{"Content-Type": "application/json"}),
            content="{",
        )
        missing = self.post_input(task_id, {"event_id": "event-1"})
        invalid_id = self.post_input(
            task_id,
            self.payload(task_id, event_id="../unsafe", key="W", action="pressed"),
        )
        for response in (bad_json, missing, invalid_id):
            self.assertEqual((400, "invalid_request"), (response.status_code, response.json()["code"]))

    def test_invalid_coordinates_are_400(self):
        task_id = self.create_task()
        for coordinate in (True, -0.01, 1.01, float("nan"), float("inf"), -float("inf")):
            with self.subTest(coordinate=coordinate):
                response = self.post_input(
                    task_id,
                    self.payload(task_id, kind="MOUSE_MOVE", x=coordinate, y=0.5),
                )
                self.assertEqual((400, "invalid_request"), (response.status_code, response.json()["code"]))

    def test_invalid_non_utc_and_naive_times_are_400(self):
        task_id = self.create_task()
        for value in ("2026-08-28T01:00:00+01:00", "2026-08-28T00:00:00", "not-a-time"):
            body = self.payload(task_id, key="W", action="pressed")
            body["occurred_at"] = value
            with self.subTest(occurred_at=value):
                response = self.post_input(task_id, body)
                self.assertEqual((400, "invalid_request"), (response.status_code, response.json()["code"]))

    def test_path_body_and_run_task_mismatches_are_400(self):
        task_id = self.create_task()
        path_mismatch = self.post_input(
            task_id, self.payload("task-other", key="W", action="pressed")
        )
        run_mismatch = self.post_input(
            task_id, self.payload(task_id, run_id="run-other", key="W", action="pressed")
        )
        for response in (path_mismatch, run_mismatch):
            self.assertEqual((400, "invalid_request"), (response.status_code, response.json()["code"]))

    def test_authentication_and_unknown_task_errors_are_stable(self):
        body = self.payload("missing", key="W", action="pressed")
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            response = self.post_input("missing", body, headers=headers)
            self.assertEqual((401, "unauthorized"), (response.status_code, response.json()["code"]))
            self.assertEqual("Bearer", response.headers["www-authenticate"])
        response = self.post_input("missing", body)
        self.assertEqual((404, "task_not_found"), (response.status_code, response.json()["code"]))

    def test_only_running_tasks_accept_input(self):
        for index, status in enumerate(TaskStatus):
            run_id = "run-status-%d" % index
            task_id = self.create_task(run_id=run_id, status=status)
            response = self.post_input(
                task_id, self.payload(task_id, run_id=run_id, key="W", action="pressed")
            )
            if status == TaskStatus.RUNNING:
                self.assertEqual(202, response.status_code)
            else:
                self.assertEqual((409, "input_not_allowed"), (response.status_code, response.json()["code"]))

    def test_replay_conflict_and_cross_task_event_id(self):
        first_task = self.create_task()
        body = self.payload(first_task, key="W", action="pressed")
        first = self.post_input(first_task, body)
        duplicate = self.post_input(first_task, body)
        conflict = self.post_input(
            first_task, self.payload(first_task, key="A", action="pressed")
        )
        second_task = self.create_task(run_id="run-input-2")
        other_task = self.post_input(
            second_task,
            self.payload(second_task, run_id="run-input-2", key="A", action="pressed"),
        )
        self.assertFalse(first.json()["duplicate"])
        self.assertEqual({"accepted": True, "event_id": "event-1", "duplicate": True}, duplicate.json())
        self.assertEqual((409, "event_conflict"), (conflict.status_code, conflict.json()["code"]))
        self.assertEqual(202, other_task.status_code)
        self.assertEqual(1, len(self.sink.events_for_task(first_task)))
        self.assertEqual(1, len(self.sink.events_for_task(second_task)))

    def test_capacity_evicts_oldest_per_task(self):
        task_id = self.create_task()
        for index in range(4):
            body = self.payload(task_id, event_id="event-%d" % index, key="W", action="pressed")
            self.assertEqual(202, self.post_input(task_id, body).status_code)
        self.assertEqual(
            ["event-1", "event-2", "event-3"],
            [event.event_id for event in self.sink.events_for_task(task_id)],
        )

    def test_concurrent_duplicate_http_requests_record_once(self):
        task_id = self.create_task()
        body = self.payload(task_id, key="W", action="pressed")

        async def send_all():
            return await asyncio.gather(
                *[
                    self.client.client.post(
                        "/simulation/tasks/%s/input-events" % task_id,
                        headers=AUTH,
                        json=body,
                    )
                    for _ in range(32)
                ]
            )

        responses = self.client.loop.run_until_complete(send_all())
        self.assertTrue(all(response.status_code == 202 for response in responses))
        self.assertEqual(1, sum(not response.json()["duplicate"] for response in responses))
        self.assertEqual(1, len(self.sink.events_for_task(task_id)))

    def test_sink_failure_is_sanitized_503_and_task_is_unchanged(self):
        class FailingSink:
            def record(self, event):
                raise RuntimeError("secret /absolute/path token-value")

        task_id = self.create_task()
        before = self.service.get_task(task_id)
        failing_client = ApiClient(
            create_http_app(self.service, TOKEN, input_sink=FailingSink())
        )
        try:
            response = failing_client.post(
                "/simulation/tasks/%s/input-events" % task_id,
                headers=AUTH,
                json=self.payload(task_id, key="W", action="pressed"),
            )
        finally:
            failing_client.close()
        self.assertEqual((503, "input_sink_unavailable"), (response.status_code, response.json()["code"]))
        self.assertNotIn("secret", response.text)
        self.assertNotIn("/absolute/path", response.text)
        self.assertEqual(before, self.service.get_task(task_id))
        self.assertEqual([], self.scheduler.cancelled_task_ids)


if __name__ == "__main__":
    unittest.main()
