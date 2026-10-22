import unittest
from dataclasses import dataclass
from datetime import datetime, timezone

from agent.context import AgentContextService
from robotics.evaluations.models import EvaluationReport, EvaluationStatus
from robotics.runs import ExecutionMode, Run, RunKind, RunStatus


@dataclass(frozen=True)
class PageContext:
    path: str
    project_id: str
    run_id: str
    scene_id: str = ""
    evaluation_id: str = ""


class EmptyWorkspace:
    def list_files(self, project_id):
        return []

    def read_file(self, project_id, path):
        raise AssertionError("workspace read was not requested")


class EmptyScenes:
    def get(self, scene_id):
        raise AssertionError("scene read was not requested")


class StubRuns:
    def __init__(self, run):
        self.run = run

    def get_run(self, run_id):
        if run_id != self.run.run_id:
            raise KeyError(run_id)
        return self.run

    def list_events(self, run_id, limit):
        return []


class StubEvaluations:
    def __init__(self, report):
        self.report = report
        self.get_calls = []

    def get(self, evaluation_id, trusted_user_id):
        self.get_calls.append((evaluation_id, trusted_user_id))
        return self.report

    def list_for_run(self, run_id, trusted_user_id):
        return [self.report]


class AgentContextTest(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.run = Run(
            run_id="run-one",
            user_id="test-user",
            project_id="project-one",
            status=RunStatus.COMPLETED,
            execution_mode=ExecutionMode.SIMULATION,
            code_sha256="a" * 64,
            run_kind=RunKind.TRAIN,
            created_at=now,
            updated_at=now,
        )
        self.report = EvaluationReport(
            evaluation_id="evaluation-one",
            run_id=self.run.run_id,
            requested_by_user_id="test-user",
            status=EvaluationStatus.INSUFFICIENT_EVIDENCE,
            score=None,
            grade="not_assessed",
            rubric_version="1.0",
            rubric_snapshot={},
            evidence={},
            evidence_sha256="b" * 64,
            summary={},
            limitations=["Missing terminal evidence"],
            evaluator_version="1.0",
            created_at=now,
        )

    def build_context(self, report):
        evaluations = StubEvaluations(report)
        service = AgentContextService(
            EmptyWorkspace(),
            EmptyScenes(),
            StubRuns(self.run),
            evaluations,
        )
        page = PageContext(
            path="/evaluation",
            project_id=self.run.project_id,
            run_id=self.run.run_id,
            evaluation_id=report.evaluation_id,
        )
        return service, evaluations, page

    def test_injects_authorized_evaluation_without_requester_identity(self):
        service, evaluations, page = self.build_context(self.report)

        context, tools = service.build(
            {"read_page_context", "read_run_evidence"},
            self.run.project_id,
            page,
            trusted_user_id="test-user",
        )

        reports = context["run_evidence"]["canonical_evaluations"]
        self.assertEqual(evaluations.get_calls, [("evaluation-one", "test-user")])
        self.assertEqual(tools, ["page_context", "run_evidence_read"])
        self.assertEqual(reports[0]["evaluation_id"], "evaluation-one")
        self.assertNotIn("requested_by_user_id", reports[0])

    def test_rejects_evaluation_from_another_run(self):
        mismatched = EvaluationReport(
            **{
                **self.report.__dict__,
                "evaluation_id": "evaluation-two",
                "run_id": "run-two",
            }
        )
        service, _, page = self.build_context(mismatched)

        with self.assertRaisesRegex(ValueError, "Evaluation 与当前 Run 不匹配"):
            service.build(
                {"read_page_context", "read_run_evidence"},
                self.run.project_id,
                page,
                trusted_user_id="test-user",
            )


if __name__ == "__main__":
    unittest.main()