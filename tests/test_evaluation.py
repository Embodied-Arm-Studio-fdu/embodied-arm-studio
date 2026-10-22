import tempfile
import time
import unittest
import json
from pathlib import Path

from robotics.adapters.mock import MockSimulatorAdapter
from robotics.contracts import RunRequest
from robotics.evaluation import EvaluationService


class EvaluationServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.adapter = MockSimulatorAdapter(Path(self.temp_dir.name), tick_seconds=0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def evaluate_run(self, environment_id):
        record = self.adapter.launch_training(RunRequest(environment_id, "joint-p-control", max_steps=8))
        deadline = time.monotonic() + 2
        while record.status.value in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.005)
        return EvaluationService(Path(self.temp_dir.name) / "evaluations").evaluate(record, self.adapter.stream_events(record.run_id), self.adapter.list_artifacts(record.run_id))

    def completed_sample(self, seed):
        record = self.adapter.launch_training(RunRequest("mock-pick-place", "joint-p-control", seed=seed, max_steps=8))
        deadline = time.monotonic() + 2
        while record.status.value in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.005)
        return record, self.adapter.stream_events(record.run_id), self.adapter.list_artifacts(record.run_id)

    def test_missing_evidence_is_not_scored_as_failure(self):
        report = self.evaluate_run("mock-reach")
        safety = next(item for item in report["criteria"] if item["criterion_id"] == "safety_constraints")
        self.assertEqual(safety["status"], "not_assessed")
        self.assertIsNone(safety["score"])
        self.assertEqual(report["overall"]["status"], "not_assessed")
        self.assertIsNone(report["overall"]["score"])
        self.assertFalse(report["sim2real"]["real_robot_control_enabled"])
        reproducibility = next(item for item in report["criteria"] if item["criterion_id"] == "reproducibility")
        self.assertEqual(reproducibility["status"], "not_assessed")

    def test_pick_place_uses_termination_and_collision_evidence(self):
        report = self.evaluate_run("mock-pick-place")
        assessed = {item["criterion_id"] for item in report["criteria"] if item["status"] == "assessed"}
        self.assertIn("termination_quality", assessed)
        self.assertIn("safety_constraints", assessed)

    def test_cohort_assesses_reproducibility_from_multiple_runs(self):
        samples = [self.completed_sample(11), self.completed_sample(29)]
        report = EvaluationService(Path(self.temp_dir.name) / "evaluations").evaluate_cohort(samples)
        reproducibility = next(item for item in report["criteria"] if item["criterion_id"] == "reproducibility")
        self.assertEqual(report["run_ids"], [samples[0][0].run_id, samples[1][0].run_id])
        self.assertEqual(reproducibility["status"], "assessed")
        self.assertEqual(len(reproducibility["evidence_refs"]), 2)
        generalization = next(item for item in report["criteria"] if item["criterion_id"] == "generalization")
        self.assertEqual(generalization["status"], "not_assessed")

    def test_report_comparison_preserves_assessment_state_changes(self):
        service = EvaluationService(Path(self.temp_dir.name) / "evaluations")
        before = {
            "report_id": "evaluation-before",
            "rubric_version": "1.0",
            "evidence_completeness": 0.5,
            "overall": {"status": "not_assessed", "score": None},
            "criteria": [
                {"criterion_id": "stability", "name": "Stability", "status": "not_assessed", "score": None, "evidence_refs": []},
                {"criterion_id": "task_success", "name": "Task success", "status": "assessed", "score": 4, "evidence_refs": ["event:old:success_rate"]},
            ],
        }
        after = {
            "report_id": "evaluation-after",
            "rubric_version": "1.0",
            "evidence_completeness": 0.7,
            "overall": {"status": "not_assessed", "score": None},
            "criteria": [
                {"criterion_id": "stability", "name": "Stability", "status": "assessed", "score": 3, "evidence_refs": ["event:new:trajectory"]},
                {"criterion_id": "task_success", "name": "Task success", "status": "assessed", "score": 2, "evidence_refs": ["event:new:success_rate"]},
            ],
        }
        for report in (before, after):
            (service.root / (report["report_id"] + ".json")).write_text(json.dumps(report), encoding="utf-8")

        comparison = service.compare_reports("evaluation-before", "evaluation-after")

        changes = {item["criterion_id"]: item for item in comparison["criteria"]}
        self.assertEqual(changes["stability"]["change"], "newly_assessed")
        self.assertIsNone(changes["stability"]["delta"])
        self.assertEqual(changes["task_success"]["change"], "regressed")
        self.assertEqual(changes["task_success"]["delta"], -2)
        self.assertEqual(comparison["completeness_delta"], 0.2)


if __name__ == "__main__":
    unittest.main()