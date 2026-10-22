"""Non-GPU Simulation HTTP -> PostgreSQL -> canonical API contract closure."""

import asyncio
import json
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from uuid import uuid4

from sqlalchemy import delete, func, select

from apps.api.database import create_postgres_engine, create_session_factory
from apps.api.main import create_app
from robotics.projects.models import (
    TrainingProfile,
    evaluation_rubric,
    parameter_schema,
    starter_code_digest,
    worker_config,
)
from robotics.runs import ExecutionMode, IdempotentRunCreateCommand, RunKind
from robotics.runs.db_models import (
    CourseRow,
    EnrollmentRow,
    EvaluationReportRow,
    OrganizationMembershipRow,
    OrganizationRow,
    RunArtifactRow,
    RunEventRow,
    RunIdempotencyKeyRow,
    RunRow,
    TeachingClassRow,
    TeachingProjectRow,
    UserRow,
)
from robotics.runs.postgres_repository import PostgreSQLRunRepository
from tests.auth import StaticCurrentUserProvider
from tests.http_client import ASGITestClient
from tests.test_projects import SAFE_POSTGRES_URL
from tests.test_v1_run_websocket import ASGIWebSocketDriver


ROOT = Path(__file__).resolve().parents[1]
SIMULATION_SRC = ROOT / "simulation" / "src"
if str(SIMULATION_SRC) not in sys.path:
    sys.path.insert(0, str(SIMULATION_SRC))
# The Simulation aggregate package also exports camera/GPU adapters and therefore
# imports NumPy.  This contract test deliberately loads only the real non-GPU
# production modules it exercises, so the Backend interpreter needs no GPU stack.
if "simulation_core" not in sys.modules:
    simulation_package = types.ModuleType("simulation_core")
    simulation_package.__path__ = [str(SIMULATION_SRC / "simulation_core")]
    simulation_package.__package__ = "simulation_core"
    sys.modules["simulation_core"] = simulation_package

from simulation_core.adapters import FakeScheduler, InMemoryTaskRepository  # noqa: E402
from simulation_core.backend_events import (  # noqa: E402
    BackendMessageFactory,
    SimulationEvent,
    utc_now_text,
)
from simulation_core.domain import TaskStatus  # noqa: E402
from simulation_core.http_backend import (  # noqa: E402
    BackendRequestRejected,
    HttpBackendClient,
)
from simulation_core.service import SimulationService  # noqa: E402
from simulation_core.training_task_runner import TrainingTaskResult  # noqa: E402


SERVICE_TOKEN = "phase-7-4-5-test-service-token"
SHA256 = "7" * 64
FIRST_RECORD = {
    "update_index": 1,
    "training_steps": 16,
    "average_reward": 0.1,
    "episode_length": 16.0,
    "episodes": 0.0,
    "policy_loss": 0.2,
    "value_loss": 0.3,
    "entropy": 7.4,
}
SECOND_RECORD = {
    "update_index": 2,
    "training_steps": 32,
    "average_reward": 0.25,
    "episode_length": 16.0,
    "episodes": 1.0,
    "policy_loss": 0.12,
    "value_loss": 0.08,
    "entropy": 7.3,
}
FORBIDDEN_METRICS = {
    "success_rate", "distance_to_goal", "total_loss", "progress"
}


class _GapSequence:
    """Leave room for an otherwise valid late Event sequence."""

    def __init__(self):
        self.value = 0

    def next(self, run_id):
        self.value += 10
        return self.value


class _LoopbackResponse:
    def __init__(self, response):
        self.response = response

    def getcode(self):
        return self.response.status_code

    def read(self, size=-1):
        return self.response.content[:size] if size >= 0 else self.response.content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ASGILoopbackOpener:
    """Drive the real urllib serializer into the existing HTTPX ASGI client."""

    def __init__(self, client):
        self.client = client
        self.calls = []

    def __call__(self, request, timeout):
        parsed = urlsplit(request.full_url)
        path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        body = json.loads(request.data.decode("utf-8"))
        response = self.client.post(
            path,
            json=body,
            headers=dict(request.header_items()),
        )
        self.calls.append(
            {"url": request.full_url, "body": body, "status": response.status_code,
             "response": response.json()}
        )
        if response.status_code >= 400:
            raise HTTPError(
                request.full_url,
                response.status_code,
                "ASGI loopback rejection",
                dict(response.headers),
                BytesIO(response.content),
            )
        return _LoopbackResponse(response)


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class Phase745LiveMetricsIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "test" not in (SAFE_POSTGRES_URL.database or "").lower():
            raise RuntimeError("Phase 7.4.5 requires a test-named PostgreSQL database")
        cls.cleanup_engine = create_postgres_engine(SAFE_POSTGRES_URL)
        cls.cleanup_factory = create_session_factory(cls.cleanup_engine)

    @classmethod
    def tearDownClass(cls):
        cls.cleanup_engine.dispose()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.prefix = "p745-%s" % uuid4().hex
        self.ids = {
            name: "%s-%s" % (self.prefix, name)
            for name in (
                "user", "org", "membership", "course", "class",
                "enrollment", "project", "run", "task",
            )
        }
        self.clients = []
        self._seed_teaching_context()
        self.app, self.client = self._make_app("before-restart")

    def tearDown(self):
        for client in reversed(self.clients):
            client.close()
        with self.cleanup_factory() as session:
            run_ids = select(RunRow.run_id).where(
                RunRow.run_id.like(self.prefix + "%")
            )
            session.execute(
                delete(EvaluationReportRow).where(
                    EvaluationReportRow.run_id.in_(run_ids)
                )
            )
            session.execute(
                delete(RunArtifactRow).where(RunArtifactRow.run_id.in_(run_ids))
            )
            session.execute(
                delete(RunIdempotencyKeyRow).where(
                    RunIdempotencyKeyRow.run_id.in_(run_ids)
                )
            )
            session.execute(
                delete(RunEventRow).where(RunEventRow.run_id.in_(run_ids))
            )
            session.execute(
                delete(RunRow).where(RunRow.run_id.like(self.prefix + "%"))
            )
            session.execute(
                delete(TeachingProjectRow).where(
                    TeachingProjectRow.project_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(EnrollmentRow).where(
                    EnrollmentRow.enrollment_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(TeachingClassRow).where(
                    TeachingClassRow.class_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(CourseRow).where(CourseRow.course_id.like(self.prefix + "%"))
            )
            session.execute(
                delete(OrganizationMembershipRow).where(
                    OrganizationMembershipRow.membership_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(OrganizationRow).where(
                    OrganizationRow.organization_id.like(self.prefix + "%")
                )
            )
            session.execute(
                delete(UserRow).where(UserRow.user_id.like(self.prefix + "%"))
            )
            session.commit()
            self.assertEqual(
                session.scalar(
                    select(func.count()).select_from(RunRow).where(
                        RunRow.run_id.like(self.prefix + "%")
                    )
                ),
                0,
            )
        self.temp.cleanup()

    def _seed_teaching_context(self):
        now = datetime.now(timezone.utc)
        starter_code = "# Phase 7.4.5 contract-only starter"
        with self.cleanup_factory() as session:
            session.add(
                UserRow(
                    user_id=self.ids["user"], username=self.ids["user"],
                    display_name="Phase 7.4.5 User", password_hash="test-only",
                    enabled=True, created_at=now, updated_at=now,
                )
            )
            session.add(
                OrganizationRow(
                    organization_id=self.ids["org"], slug=self.ids["org"],
                    name="Phase 7.4.5", enabled=True,
                    created_at=now, updated_at=now,
                )
            )
            session.flush()
            session.add(
                OrganizationMembershipRow(
                    membership_id=self.ids["membership"],
                    organization_id=self.ids["org"], user_id=self.ids["user"],
                    role="student", enabled=True, created_at=now, updated_at=now,
                )
            )
            session.add(
                CourseRow(
                    course_id=self.ids["course"], organization_id=self.ids["org"],
                    code=self.prefix, title="Phase 7.4.5 Course", description="",
                    owner_instructor_user_id=self.ids["user"], enabled=True,
                    created_at=now, updated_at=now,
                )
            )
            session.flush()
            session.add(
                TeachingClassRow(
                    class_id=self.ids["class"], course_id=self.ids["course"],
                    code="short", name="Phase 7.4.5 Class", term=self.prefix,
                    instructor_user_id=self.ids["user"], enabled=True,
                    created_at=now, updated_at=now,
                )
            )
            session.flush()
            session.add(
                EnrollmentRow(
                    enrollment_id=self.ids["enrollment"],
                    class_id=self.ids["class"], user_id=self.ids["user"],
                    enabled=True, created_at=now, updated_at=now,
                )
            )
            session.add(
                TeachingProjectRow(
                    project_id=self.ids["project"], course_id=self.ids["course"],
                    title=self.prefix, description="short contract test",
                    instructions="Do not start GPU training.",
                    starter_code=starter_code,
                    starter_code_sha256=starter_code_digest(starter_code),
                    training_profile="short", video_enabled=False,
                    parameter_schema=parameter_schema(),
                    evaluation_rubric=evaluation_rubric(TrainingProfile.SHORT),
                    config_version=1, enabled=True,
                    created_by_user_id=self.ids["user"],
                    created_at=now, updated_at=now,
                )
            )
            session.commit()

    def _make_app(self, name):
        root = Path(self.temp.name) / name
        app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository_mode="postgresql",
            database_url=SAFE_POSTGRES_URL,
            current_user_provider=StaticCurrentUserProvider(self.ids["user"]),
            internal_run_event_service_token=SERVICE_TOKEN,
            run_event_poll_interval_seconds=0.01,
        )
        client = ASGITestClient(app)
        self.clients.append(client)
        self.assertIsInstance(app.state.repository, PostgreSQLRunRepository)
        return app, client

    def _create_formal_run(self):
        self.app.state.run_service._run_id_factory = lambda: self.ids["run"]
        result = self.app.state.run_creation_service.create_run(
            IdempotentRunCreateCommand(
                user_id=self.ids["user"],
                idempotency_key=self.prefix + "-create",
                project_id=self.ids["project"],
                class_id=self.ids["class"],
                project_config_version=1,
                execution_mode=ExecutionMode.SIMULATION,
                run_kind=RunKind.TRAIN,
                parameters={"seed": 745},
                code="student code is hashed but never executed",
            )
        )
        self.assertFalse(result.replayed)
        snapshot = result.run.project_config_snapshot
        self.assertEqual(snapshot["training_profile"], "short")
        self.assertEqual(snapshot["expected_training_steps"], 32)
        self.assertEqual(snapshot["project_id"], self.ids["project"])
        self.assertEqual(snapshot["class_id"], self.ids["class"])
        self.assertEqual(snapshot["course_id"], self.ids["course"])
        self.assertEqual(snapshot["resolved_parameters"], {"seed": 745})
        self.assertEqual(
            snapshot["training_config"],
            worker_config(TrainingProfile.SHORT, False, 745),
        )
        self.assertEqual(len(snapshot["training_config"]), 16)
        return result.run

    @staticmethod
    def _simulation_event(body):
        return SimulationEvent(
            schema_version=body["schema_version"],
            event_id=body["event_id"],
            sequence=body["sequence"],
            type=body["type"],
            run_id=body["run_id"],
            task_id=body["task_id"],
            occurred_at=body["occurred_at"],
            payload=body["payload"],
        )

    @staticmethod
    def _assert_wire_record(testcase, body, record):
        testcase.assertEqual(body["schema_version"], "1.0")
        testcase.assertTrue(body["event_id"])
        testcase.assertGreater(body["sequence"], 0)
        testcase.assertTrue(body["run_id"])
        testcase.assertTrue(body["task_id"])
        testcase.assertEqual(
            datetime.fromisoformat(body["occurred_at"].replace("Z", "+00:00"))
            .utcoffset(),
            timezone.utc.utcoffset(None),
        )
        testcase.assertEqual(set(body["payload"]), {"step", "metrics"})
        testcase.assertEqual(body["payload"]["step"], record["training_steps"])
        testcase.assertNotIn("update_index", body["payload"])
        expected_metrics = {
            key: value for key, value in record.items() if key != "training_steps"
        }
        testcase.assertEqual(body["payload"]["metrics"], expected_metrics)
        testcase.assertTrue(
            FORBIDDEN_METRICS.isdisjoint(body["payload"]["metrics"])
        )

    @staticmethod
    def _collect_ws(app, url, count, expect_close=False):
        async def collect():
            socket = ASGIWebSocketDriver(app, url)
            accepted = await socket.connect()
            if accepted["type"] != "websocket.accept":
                raise AssertionError(accepted)
            messages = [await socket.receive_json() for _ in range(count)]
            if expect_close:
                close = await socket.receive()
                if close.get("code") != 1000:
                    raise AssertionError(close)
                await socket.finish()
            else:
                await socket.disconnect()
            return messages

        return asyncio.run(collect())

    def test_non_gpu_real_http_asgi_contract_closure(self):
        run = self._create_formal_run()
        opener = ASGILoopbackOpener(self.client)
        http_backend = HttpBackendClient(
            "http://backend.test",
            request_timeout=2,
            service_token=SERVICE_TOKEN,
            opener=opener,
        )
        scheduler = FakeScheduler()
        task_repository = InMemoryTaskRepository()
        event_counter = {"value": 0}

        def event_id():
            event_counter["value"] += 1
            return "%s-event-%d" % (self.prefix, event_counter["value"])

        simulation = SimulationService(
            task_repository,
            scheduler,
            http_backend,
            task_id_factory=lambda: self.ids["task"],
            message_factory=BackendMessageFactory(
                sequence_generator=_GapSequence(), event_id_factory=event_id
            ),
        )
        task = simulation.create_task(
            run.run_id,
            run.project_id,
            run.project_config_snapshot["training_config"],
        )
        self.assertEqual(scheduler.submitted_task_ids, [self.ids["task"]])
        self.assertEqual(len(task.config.values), 16)
        self.app.state.run_service.bind_simulation_task(run.run_id, task.task_id)
        simulation.transition_task_status(task.task_id, TaskStatus.INITIALIZING)
        simulation.transition_task_status(task.task_id, TaskStatus.RUNNING)

        simulation.publish_metric_record(task.task_id, FIRST_RECORD)
        first_body = [
            call["body"] for call in opener.calls
            if call["body"].get("type") == "metric"
        ][-1]
        self._assert_wire_record(self, first_body, FIRST_RECORD)
        first_detail = self.client.get("/api/v1/runs/%s" % run.run_id)
        self.assertEqual(first_detail.status_code, 200, first_detail.text)
        self.assertEqual(first_detail.json()["training_steps"], 16)
        self.assertEqual(first_detail.json()["progress"], 0.5)
        self.assertEqual(first_detail.json()["metrics_summary"]["update_index"], 1)
        self.assertEqual(first_detail.json()["metrics_summary"]["step"], 16)
        first_history = self.client.get("/api/v1/runs/%s/events" % run.run_id)
        self.assertEqual(first_history.status_code, 200, first_history.text)
        self.assertEqual(first_history.json()[-1]["payload"], first_body["payload"])
        stop_service = self.app.state.simulation_stop_service
        original_stop = stop_service.stop
        stop_calls = []

        def record_stop(*args, **kwargs):
            stop_calls.append((args, kwargs))
            return original_stop(*args, **kwargs)

        stop_service.stop = record_stop
        try:
            first_ws = self._collect_ws(
                self.app,
                "/api/v1/runs/%s/events" % run.run_id,
                3,
            )
        finally:
            stop_service.stop = original_stop
        self.assertEqual(first_ws[-1]["payload"]["metrics"]["update_index"], 1)
        self.assertEqual(stop_calls, [])
        self.assertEqual(simulation.get_task(task.task_id).status, TaskStatus.RUNNING)

        simulation.publish_metric_record(task.task_id, SECOND_RECORD)
        second_body = [
            call["body"] for call in opener.calls
            if call["body"].get("type") == "metric"
        ][-1]
        self._assert_wire_record(self, second_body, SECOND_RECORD)
        second_detail = self.client.get("/api/v1/runs/%s" % run.run_id).json()
        self.assertEqual(second_detail["training_steps"], 32)
        self.assertEqual(second_detail["progress"], 0.99)
        self.assertEqual(second_detail["metrics_summary"]["update_index"], 2)
        self.assertEqual(second_detail["metrics_summary"]["step"], 32)
        for key, value in SECOND_RECORD.items():
            if key not in {"training_steps", "update_index"}:
                self.assertEqual(second_detail["metrics_summary"][key], value)
        second_ws = self._collect_ws(
            self.app,
            "/api/v1/runs/%s/events?after_sequence=%d"
            % (run.run_id, first_body["sequence"]),
            1,
        )
        self.assertEqual(second_ws[0]["payload"]["metrics"]["update_index"], 2)

        second_event = self._simulation_event(second_body)
        event_count = len(self.client.get(
            "/api/v1/runs/%s/events" % run.run_id
        ).json())
        http_backend.publish_event(second_event)
        self.assertEqual(opener.calls[-1]["response"]["duplicate"], True)
        self.assertEqual(
            len(self.client.get("/api/v1/runs/%s/events" % run.run_id).json()),
            event_count,
        )
        changed_payload = deepcopy(second_body["payload"])
        changed_payload["metrics"]["average_reward"] = 0.26
        conflicting = SimulationEvent(
            event_id=second_event.event_id,
            sequence=second_event.sequence,
            type=second_event.type,
            run_id=second_event.run_id,
            task_id=second_event.task_id,
            occurred_at=second_event.occurred_at,
            payload=changed_payload,
        )
        with self.assertRaises(BackendRequestRejected) as conflict:
            http_backend.publish_event(conflicting)
        self.assertEqual(conflict.exception.status_code, 409)

        self.client.close()
        self.app, self.client = self._make_app("after-live-restart")
        opener.client = self.client
        recovered = self.client.get("/api/v1/runs/%s" % run.run_id).json()
        self.assertEqual(
            (recovered["training_steps"], recovered["progress"],
             recovered["metrics_summary"]["update_index"]),
            (32, 0.99, 2),
        )
        self.assertEqual(
            self.client.get("/api/v1/runs/%s/events" % run.run_id).json()[-1]
            ["event_id"],
            second_event.event_id,
        )

        regressing = dict(SECOND_RECORD, update_index=1, training_steps=16)
        simulation.publish_metric_record(task.task_id, regressing)
        after_regression = self.client.get("/api/v1/runs/%s" % run.run_id).json()
        self.assertEqual(
            (after_regression["training_steps"], after_regression["progress"],
             after_regression["metrics_summary"]["update_index"],
             after_regression["metrics_summary"]["step"]),
            (32, 0.99, 2, 32),
        )
        late = SimulationEvent(
            event_id=self.prefix + "-late-event",
            sequence=35,
            type="metric",
            run_id=run.run_id,
            task_id=task.task_id,
            occurred_at=utc_now_text(),
            payload={
                "step": 8,
                "metrics": {
                    key: value for key, value in FIRST_RECORD.items()
                    if key != "training_steps"
                },
            },
        )
        http_backend.publish_event(late)
        self.assertEqual(opener.calls[-1]["response"]["late"], True)
        after_late = self.client.get("/api/v1/runs/%s" % run.run_id).json()
        self.assertEqual(after_late["metrics_summary"], after_regression["metrics_summary"])
        sequences = [
            event["sequence"] for event in self.client.get(
                "/api/v1/runs/%s/events" % run.run_id
            ).json()
        ]
        self.assertEqual(sequences, sorted(sequences))
        self.assertIn(35, sequences)

        failing_repository = InMemoryTaskRepository()
        failing_http = HttpBackendClient(
            "http://unavailable.test",
            opener=lambda request, timeout: (_ for _ in ()).throw(
                OSError("test transport unavailable")
            ),
        )
        failing_simulation = SimulationService(
            failing_repository, FakeScheduler(), failing_http,
            task_id_factory=lambda: self.prefix + "-failing-task",
        )
        failing_task = failing_simulation.create_task(
            self.prefix + "-failing-run", self.ids["project"],
            worker_config(TrainingProfile.SHORT, False, 745),
        )
        failing_simulation.transition_task_status(
            failing_task.task_id, TaskStatus.INITIALIZING
        )
        failing_simulation.transition_task_status(
            failing_task.task_id, TaskStatus.RUNNING
        )
        failing_simulation.publish_metric_record(failing_task.task_id, FIRST_RECORD)
        self.assertEqual(
            failing_simulation.get_task(failing_task.task_id).status,
            TaskStatus.RUNNING,
        )

        simulation.transition_task_status(task.task_id, TaskStatus.FINALIZING)
        simulation.transition_task_status(task.task_id, TaskStatus.COMPLETED)
        result = TrainingTaskResult(
            task_id=task.task_id,
            run_id=run.run_id,
            status=TaskStatus.COMPLETED,
            training_steps=32,
            update_steps=2,
            metrics_path="metrics/training.jsonl",
            checkpoint_path="checkpoints/model_latest.pt",
            final_metrics_summary=deepcopy(SECOND_RECORD),
            manifest_path="manifest.json",
            best_checkpoint_path="checkpoints/model_best.pt",
            artifacts=(
                {
                    "type": "checkpoint_best",
                    "path": "checkpoints/model_best.pt",
                    "size_bytes": 745,
                    "sha256": SHA256,
                },
                {
                    "type": "training_metrics",
                    "path": "metrics/training.jsonl",
                    "size_bytes": 245,
                    "sha256": SHA256,
                },
                {
                    "type": "manifest",
                    "path": "manifest.json",
                    "size_bytes": 345,
                    "sha256": SHA256,
                },
            ),
            started_at="2026-09-03T10:00:00+00:00",
            ended_at="2026-09-03T10:01:00+00:00",
        )
        delivered = simulation.enqueue_terminal_callback(task.task_id, result)
        self.assertEqual(delivered.delivery_status.value, "delivered")
        callback = delivered.callback
        callback_call = opener.calls[-1]
        self.assertTrue(callback_call["url"].endswith("/callback"))
        self.assertEqual(callback_call["body"]["training_steps"], 32)
        self.assertEqual(callback_call["body"]["update_steps"], 2)
        self.assertEqual(callback_call["body"]["metrics"]["update_index"], 2)
        terminal = self.client.get("/api/v1/runs/%s" % run.run_id).json()
        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(terminal["progress"], 1.0)
        self.assertEqual(terminal["training_steps"], 32)
        self.assertEqual(terminal["update_steps"], 2)
        terminal_event = self.client.get(
            "/api/v1/runs/%s/events" % run.run_id
        ).json()[-1]
        self.assertEqual(terminal_event["payload"]["training_steps"], 32)
        self.assertEqual(terminal_event["payload"]["update_steps"], 2)
        self.assertEqual(terminal_event["payload"]["metrics"]["update_index"], 2)
        artifacts = self.client.get(
            "/api/v1/runs/%s/artifacts" % run.run_id
        ).json()
        self.assertEqual(
            {artifact["path"] for artifact in artifacts},
            {"checkpoints/model_best.pt", "manifest.json", "metrics/training.jsonl"},
        )
        self.assertTrue(all("content" not in artifact for artifact in artifacts))

        http_backend.publish_terminal_callback(callback)
        self.assertEqual(opener.calls[-1]["response"]["duplicate"], True)
        self.assertEqual(
            len(self.client.get("/api/v1/runs/%s/artifacts" % run.run_id).json()),
            3,
        )
        terminal_ws = self._collect_ws(
            self.app,
            "/api/v1/runs/%s/events?after_sequence=50" % run.run_id,
            2,
            expect_close=True,
        )
        self.assertEqual(terminal_ws[-1]["payload"]["status"], "completed")
        self.assertEqual(
            terminal_ws[-1]["payload"]["metrics"]["update_index"], 2
        )

        self.client.close()
        self.app, self.client = self._make_app("after-terminal-restart")
        opener.client = self.client
        terminal_recovered = self.client.get(
            "/api/v1/runs/%s" % run.run_id
        ).json()
        self.assertEqual(
            (terminal_recovered["status"], terminal_recovered["progress"],
             terminal_recovered["training_steps"],
             terminal_recovered["update_steps"],
             terminal_recovered["metrics_summary"]["update_index"]),
            ("completed", 1.0, 32, 2, 2),
        )
        recovered_events = self.client.get(
            "/api/v1/runs/%s/events" % run.run_id
        ).json()
        self.assertEqual(recovered_events[-1]["event_id"], callback.event_id)
        with self.cleanup_factory() as session:
            row = session.get(RunRow, run.run_id)
            self.assertEqual(row.training_steps, 32)
            self.assertEqual(row.progress, 1.0)
            self.assertEqual(row.metrics_summary["update_index"], 2)
            metric_rows = session.execute(
                select(RunEventRow).where(
                    RunEventRow.run_id == run.run_id,
                    RunEventRow.type == "metric",
                )
            ).scalars().all()
            self.assertTrue(any(
                item.payload.get("metrics", {}).get("update_index") == 1
                for item in metric_rows
            ))
            self.assertTrue(any(
                item.payload.get("metrics", {}).get("update_index") == 2
                for item in metric_rows
            ))
            self.assertEqual(
                session.scalar(
                    select(func.count()).select_from(RunArtifactRow).where(
                        RunArtifactRow.run_id == run.run_id
                    )
                ),
                3,
            )


if __name__ == "__main__":
    unittest.main()
