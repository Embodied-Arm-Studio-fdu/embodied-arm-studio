import unittest
from types import SimpleNamespace

from robotics.runs import ExecutionMode, IdempotentRunCreateCommand, RunKind
from robotics.runs.postgres_idempotency import PostgreSQLProjectRunCreationService
from robotics.simulation.contracts import SimulationTaskCatalog
from simulation.src.simulation_core.task_catalog import task_catalog_document


class TrainingVideoDefaults(unittest.TestCase):
    def test_new_catalog_runs_default_to_video_even_for_older_projects(self):
        service = PostgreSQLProjectRunCreationService.__new__(PostgreSQLProjectRunCreationService)
        project = SimpleNamespace(project_id="p", course_id="c", title="P", description="", instructions="", training_profile="standard", video_enabled=False, config_version=4, enabled=True, starter_code_sha256="a" * 64)
        catalog = SimulationTaskCatalog.from_dict(task_catalog_document())
        for parameters, expected in [({"seed": 7}, True), ({"seed": 7, "video_enabled": False}, False)]:
            command = IdempotentRunCreateCommand(user_id="u", idempotency_key="test", project_id="p", class_id="class", project_config_version=4, execution_mode=ExecutionMode.SIMULATION, run_kind=RunKind.TRAIN, environment_type="s1_push", parameters=parameters)
            resolved, snapshot = service._catalog_snapshot(command, catalog, project, SimpleNamespace(class_id="class"))
            self.assertIs(resolved["video_enabled"], expected)
            self.assertIs(snapshot["task_config"]["video_enabled"], expected)
            self.assertIs(snapshot["video_enabled"], expected)