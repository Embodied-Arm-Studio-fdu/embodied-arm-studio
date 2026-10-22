import tempfile
import threading
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from apps.api.main import create_app
from tests.auth import StaticCurrentUserProvider
from apps.api.dependencies.simulation import build_simulation_assembly
from robotics.runs import (
    IdempotencyConflict,
    IdempotentRunCreateCommand,
    IdempotentRunCreateResult,
    InMemoryRunRepository,
    RepositoryIntegrityError,
    RunService,
    PostgreSQLProjectRunCreationService,
    fingerprint_run_create,
)
from robotics.projects.models import (
    TrainingProfile,
    evaluation_rubric,
    parameter_schema,
    success_condition,
    worker_config,
)
from robotics.simulation import (
    SimulationNetworkError,
    SimulationRejectedError,
    SimulationResponseError,
    SimulationServerError,
    SimulationSnapshotMismatch,
    SimulationTaskResponse,
    SimulationTimeoutError,
    build_simulation_config,
)
from tests.http_client import ASGITestClient


class MemoryCreationService:
    def __init__(self, repository):
        self.service = RunService(repository)
        self.records = {}
        self.lock = threading.RLock()

    def create_run(self, command):
        self.assert_command(command)
        fingerprint = fingerprint_run_create(command)
        key = (command.user_id, command.idempotency_key)
        with self.lock:
            existing = self.records.get(key)
            if existing is not None:
                old_fingerprint, run_id = existing
                if old_fingerprint != fingerprint:
                    raise IdempotencyConflict(*key)
                return IdempotentRunCreateResult(
                    self.service.get_run(run_id), replayed=True
                )
            seed = command.parameters["seed"]
            config = worker_config(TrainingProfile.SHORT, True, seed)
            run = self.service.create_run(
                user_id=command.user_id,
                project_id=command.project_id,
                class_id=command.class_id,
                execution_mode=command.execution_mode,
                run_kind=command.run_kind,
                parameters={"seed": seed},
                project_config_snapshot={
                    "schema_version": "1.0",
                    "project_id": command.project_id,
                    "course_id": "course-dispatch",
                    "class_id": command.class_id,
                    "title": "Dispatch",
                    "description": "",
                    "instructions": "",
                    "task_type": "training",
                    "environment_type": "s1_push",
                    "algorithm": "PPO",
                    "success_condition": success_condition(),
                    "training_profile": "short",
                    "expected_training_steps": 32,
                    "video_enabled": True,
                    "parameter_schema": parameter_schema(),
                    "resolved_parameters": {"seed": seed},
                    "evaluation_rubric": evaluation_rubric(TrainingProfile.SHORT),
                    "config_version": command.project_config_version,
                    "enabled": True,
                    "project_enabled_at_creation": True,
                    "starter_code_sha256": "0" * 64,
                    "training_config": config,
                },
                code=command.code,
            )
            self.records[key] = (fingerprint, run.run_id)
            return IdempotentRunCreateResult(run, replayed=False)

    @staticmethod
    def assert_command(command):
        if not isinstance(command, IdempotentRunCreateCommand):
            raise TypeError("command must be IdempotentRunCreateCommand")


class FakeSimulationClient:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.error = None
        self.run_id_override = None
        self.project_id_override = None
        self.task_id = "task-dispatch-1"
        self.stop_calls = []

    def create_task(self, run_id, project_id, config):
        self.calls.append((run_id, project_id, deepcopy(config)))
        if self.error is not None:
            raise self.error
        now = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
        return SimulationTaskResponse(
            task_id=self.task_id,
            run_id=self.run_id_override or run_id,
            project_id=self.project_id_override or project_id,
            status="queued",
            created_at=now,
            updated_at=now,
            config=config,
        )

    def close(self):
        self.closed = True

    def stop_task(self, task_id):
        self.stop_calls.append(task_id)
        if self.error is not None:
            raise self.error
        call = self.calls[-1]
        now = datetime(2026, 8, 27, 12, 1, tzinfo=timezone.utc)
        return SimulationTaskResponse(
            task_id=task_id,
            run_id=self.run_id_override or call[0],
            project_id=self.project_id_override or call[1],
            status="stopped",
            created_at=now,
            updated_at=now,
            config=call[2],
        )


class SimulationDispatchApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = InMemoryRunRepository()
        self.creation = MemoryCreationService(self.repository)
        self.simulation = FakeSimulationClient()
        self.app = self.make_app(self.simulation)
        self.client = ASGITestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def make_app(self, simulation_client=None, simulation_environment=None):
        root = Path(self.temp.name)
        return create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=self.repository,
            run_creation_service=self.creation,
            simulation_client=simulation_client,
            simulation_environment=simulation_environment,
            current_user_provider=StaticCurrentUserProvider("canonical-api"),
        )

    @staticmethod
    def payload(mode="simulation"):
        return {
            "project_id": "project-dispatch",
            "class_id": "class-dispatch",
            "project_config_version": 1,
            "execution_mode": mode,
            "run_kind": "train",
            "parameters": {"seed": 7},
            "code": "print('dispatch')\n",
        }

    def post(self, key="dispatch-key", payload=None):
        return self.client.post(
            "/api/v1/runs",
            json=self.payload() if payload is None else payload,
            headers={"Idempotency-Key": key},
        )

    def test_simulation_create_binds_task_and_maps_worker_config(self):
        response = self.post()
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["simulation_task_id"], "task-dispatch-1")
        self.assertEqual(len(self.simulation.calls), 1)
        run_id, project_id, config = self.simulation.calls[0]
        self.assertEqual(run_id, response.json()["run_id"])
        self.assertEqual(project_id, "project-dispatch")
        self.assertEqual(
            config,
            {
                "task_type": "training", "environment_type": "s1_push",
                "seed": 7, "rollout_steps": 16, "num_updates": 2,
                "checkpoint_every": 1, "episode_steps": 32,
                "physics_steps_per_action": 2, "update_epochs": 1,
                "batch_size": 16, "device": "cuda:0",
                "video_enabled": True, "camera_id": "overview",
                "video_fps": 15.0, "video_width": 640,
                "video_height": 480,
            },
        )
        persisted = self.app.state.run_service.get_run(run_id)
        mapped = build_simulation_config(persisted)
        mapped["seed"] = 999
        self.assertEqual(persisted.parameters["seed"], 7)

    def test_standard_v1_and_v2_dispatch_from_frozen_config_version(self):
        response = self.post(key="versioned-standard-source")
        source = self.repository.get_run(response.json()["run_id"])

        def standard_run(config_version):
            snapshot = deepcopy(source.project_config_snapshot)
            snapshot["training_profile"] = TrainingProfile.STANDARD.value
            snapshot["config_version"] = config_version
            snapshot["evaluation_rubric"] = evaluation_rubric(
                TrainingProfile.STANDARD, config_version
            )
            config = worker_config(
                TrainingProfile.STANDARD,
                True,
                7,
                config_version=config_version,
            )
            snapshot["training_config"] = config
            snapshot["expected_training_steps"] = (
                config["rollout_steps"] * config["num_updates"]
            )
            return replace(source, project_config_snapshot=snapshot)

        legacy = build_simulation_config(standard_run(1))
        self.assertNotIn("run_mode", legacy)
        self.assertEqual(
            (legacy["rollout_steps"], legacy["num_updates"]), (16, 20)
        )

        production = build_simulation_config(standard_run(2))
        self.assertEqual(production["run_mode"], "training")
        self.assertEqual(
            (
                production["rollout_steps"], production["num_updates"],
                production["episode_steps"],
                production["physics_steps_per_action"],
                production["checkpoint_every"], production["update_epochs"],
                production["batch_size"], production["seed"],
            ),
            (200, 100, 256, 8, 10, 1, 16, 7),
        )
        self.assertTrue(
            {
                "learning_rate", "gamma", "gae_lambda", "clip_epsilon",
                "value_loss_coefficient", "entropy_coefficient",
                "max_grad_norm",
            }.isdisjoint(production)
        )

    def test_frozen_s1_scene_configures_worker_push_geometry(self):
        response = self.post(key="scene-worker-geometry")
        source = self.repository.get_run(response.json()["run_id"])
        snapshot = deepcopy(source.project_config_snapshot)
        snapshot["scene"] = {
            "scene_ref": {"scene_id": "scene-s1-v2-visual", "revision": 3},
            "document": {
                "schema_version": "1.0",
                "scene_id": "scene-s1-v2-visual",
                "revision": 3,
                "capability": "simulation",
                "robot": {"id": "yhrg-s1-v2", "dof": 6},
                "objects": [
                    {"object_id": "push-object", "type": "box", "static": False,
                     "position": [0.48, 0.04, -0.08], "scale": [0.10, 0.08, 0.12]},
                    {"object_id": "target-01", "type": "target", "static": True,
                     "position": [0.66, 0.01, 0.14], "scale": [0.18, 0.02, 0.18]},
                ],
            },
        }

        mapped = build_simulation_config(
            replace(source, project_config_snapshot=snapshot)
        )

        self.assertEqual(mapped["object_size"], [0.10, 0.12, 0.08])
        self.assertEqual(mapped["object_start"], [0.48, -0.08, 0.415])
        self.assertEqual(mapped["goal_position"], [0.66, 0.14, 0.415])

    def test_postgres_snapshot_derives_production_steps_without_constants(self):
        project = SimpleNamespace(
            project_id="project-production",
            course_id="course-production",
            title="Production",
            description="",
            instructions="",
            training_profile=TrainingProfile.STANDARD.value,
            video_enabled=True,
            parameter_schema=parameter_schema(),
            evaluation_rubric=evaluation_rubric(TrainingProfile.STANDARD, 2),
            config_version=2,
            enabled=True,
            starter_code_sha256="0" * 64,
        )
        teaching_class = SimpleNamespace(class_id="class-production")
        config = worker_config(
            TrainingProfile.STANDARD, True, 42, config_version=2
        )
        snapshot = PostgreSQLProjectRunCreationService._snapshot(
            project, teaching_class, 42, config, None
        )
        self.assertEqual(snapshot["expected_training_steps"], 20000)
        self.assertEqual(snapshot["training_config"], config)

    def test_tampered_frozen_config_and_parameter_override_fail_closed(self):
        response = self.post(key="tamper-source")
        run = self.repository.get_run(response.json()["run_id"])
        missing_title = dict(run.project_config_snapshot)
        missing_title.pop("title")
        for snapshot, parameters in (
            (dict(run.project_config_snapshot, training_config={}), run.parameters),
            (
                dict(
                    run.project_config_snapshot,
                    training_config=dict(
                        run.project_config_snapshot["training_config"],
                        num_envs=8,
                    ),
                ),
                run.parameters,
            ),
            (
                dict(
                    run.project_config_snapshot,
                    training_config=dict(
                        run.project_config_snapshot["training_config"],
                        batch_size=1,
                    ),
                ),
                run.parameters,
            ),
            (run.project_config_snapshot, {"seed": 7, "batch_size": 1}),
            (missing_title, run.parameters),
            (
                dict(run.project_config_snapshot, success_condition={}),
                run.parameters,
            ),
            (run.project_config_snapshot, {"seed": True}),
        ):
            with self.subTest(parameters=parameters, keys=list(snapshot.get("training_config", {}))):
                with self.assertRaises(SimulationSnapshotMismatch):
                    build_simulation_config(
                        replace(
                            run,
                            project_config_snapshot=snapshot,
                            parameters=parameters,
                        )
                    )

    def test_formal_v1_rejects_mock_without_calling_simulation(self):
        response = self.post(payload=self.payload("mock"))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.simulation.calls, [])

    def test_missing_configuration_keeps_run_queued_and_returns_503(self):
        self.client.close()
        app = self.make_app(simulation_environment={})
        self.client = ASGITestClient(app)
        response = self.post(key="missing-config")
        self.assertEqual(response.status_code, 503)
        runs = app.state.run_service.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status.value, "queued")
        self.assertIsNone(runs[0].simulation_task_id)

    def test_replay_of_bound_run_does_not_call_create_again(self):
        first = self.post()
        replay = self.post()
        self.assertEqual(first.status_code, replay.status_code)
        self.assertEqual(replay.headers["Idempotency-Replayed"], "true")
        self.assertEqual(len(self.simulation.calls), 1)

    def test_dispatch_failure_replay_retries_and_binds(self):
        self.simulation.error = SimulationTimeoutError("timeout")
        failed = self.post(key="recover")
        self.assertEqual(failed.status_code, 503)
        run = self.repository.list_runs()[0]
        self.assertIsNone(run.simulation_task_id)
        self.simulation.error = None
        recovered = self.post(key="recover")
        self.assertEqual(recovered.status_code, 202)
        self.assertEqual(recovered.json()["simulation_task_id"], self.simulation.task_id)
        self.assertEqual(len(self.simulation.calls), 2)


    def test_binding_failure_replay_recovers_without_new_run(self):
        original = self.repository.bind_task_id
        attempts = {"count": 0}

        def fail_once(run_id, task_id):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RepositoryIntegrityError("bind_task_id")
            return original(run_id, task_id)

        self.repository.bind_task_id = fail_once
        first = self.post(key="bind-recover")
        second = self.post(key="bind-recover")
        self.assertEqual(first.status_code, 503)
        self.assertEqual(second.status_code, 202)
        self.assertEqual(len(self.repository.list_runs()), 1)
        self.assertEqual(len(self.simulation.calls), 2)

    def test_mismatched_snapshot_returns_502_without_binding(self):
        self.simulation.project_id_override = "wrong-project"
        response = self.post(key="mismatch")
        self.assertEqual(response.status_code, 502)
        self.assertIsNone(self.repository.list_runs()[0].simulation_task_id)

    def test_failure_status_mapping_is_stable_and_redacted(self):
        cases = [
            (SimulationTimeoutError("secret https://private/path"), 503),
            (SimulationNetworkError("secret https://private/path"), 503),
            (SimulationServerError(500), 503),
            (SimulationRejectedError(401), 503),
            (SimulationRejectedError(409), 409),
            (SimulationRejectedError(422), 502),
            (SimulationResponseError("secret https://private/path"), 502),
        ]
        for index, (error, status) in enumerate(cases):
            with self.subTest(error=type(error).__name__):
                self.simulation.error = error
                response = self.post(key="error-%d" % index)
                self.assertEqual(response.status_code, status)
                rendered = response.text
                self.assertNotIn("secret", rendered)
                self.assertNotIn("https://", rendered)

    def test_shutdown_closes_injected_client(self):
        self.assertFalse(self.simulation.closed)
        self.client.close()
        self.assertTrue(self.simulation.closed)
        self.app = self.make_app(FakeSimulationClient())
        self.client = ASGITestClient(self.app)

    def test_concurrent_same_request_binds_one_task_id(self):
        barrier = threading.Barrier(2)
        original_create = self.simulation.create_task

        def synchronized_create(run_id, project_id, config):
            barrier.wait(timeout=3)
            return original_create(run_id, project_id, config)

        self.simulation.create_task = synchronized_create
        responses = []
        failures = []

        def submit(app):
            try:
                with ASGITestClient(app) as client:
                    responses.append(
                        client.post(
                            "/api/v1/runs",
                            json=self.payload(),
                            headers={"Idempotency-Key": "concurrent-key"},
                        )
                    )
            except Exception as exc:
                failures.append(exc)

        # The setup client owns its app lifespan, so concurrent requests use fresh apps.
        threads = [
            threading.Thread(target=submit, args=(self.make_app(self.simulation),))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(failures, [])
        self.assertEqual([response.status_code for response in responses], [202, 202])
        self.assertEqual(len(self.repository.list_runs()), 1)
        self.assertEqual(
            self.repository.list_runs()[0].simulation_task_id,
            "task-dispatch-1",
        )
        self.assertEqual(len(self.simulation.calls), 2)


class SimulationAssemblyTest(unittest.TestCase):
    def test_environment_builds_real_client_with_five_second_default(self):
        assembly = build_simulation_assembly(
            RunService(InMemoryRunRepository()),
            environment={
                "SIMULATION_BASE_URL": "https://simulation.example.invalid",
                "SIMULATION_SERVICE_TOKEN": "injected-test-value",
            },
        )
        try:
            self.assertIsNotNone(assembly.client)
            self.assertIsNotNone(assembly.dispatch_service)
            self.assertIs(
                assembly.stream_service._client, assembly.client
            )
            self.assertEqual(assembly.client._client.timeout.connect, 5.0)
        finally:
            assembly.client.close()


if __name__ == "__main__":
    unittest.main()
