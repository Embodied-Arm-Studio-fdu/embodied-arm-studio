from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

from robotics.contracts import RunEvent, RunRecord, RunStatus


RUBRIC_VERSION = "1.0"


@dataclass(frozen=True)
class CriterionDefinition:
    criterion_id: str
    name: str
    weight: int
    mandatory: bool = False


CRITERIA = [
    CriterionDefinition("configuration_validity", "Configuration validity", 8),
    CriterionDefinition("task_success", "Task success", 18),
    CriterionDefinition("reward_quality", "Reward quality", 10),
    CriterionDefinition("stability", "Stability", 10),
    CriterionDefinition("termination_quality", "Termination quality", 8),
    CriterionDefinition("safety_constraints", "Safety constraints", 16, True),
    CriterionDefinition("reproducibility", "Reproducibility", 14),
    CriterionDefinition("efficiency", "Efficiency", 6),
    CriterionDefinition("generalization", "Generalization", 6),
    CriterionDefinition("artifact_completeness", "Artifact completeness", 4),
]


class EvaluationService:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def evaluate(self, record: RunRecord, events: Iterable[RunEvent], artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.evaluate_cohort([(record, events, artifacts)])

    def evaluate_cohort(
        self,
        samples: List[Tuple[RunRecord, Iterable[RunEvent], List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        if not samples:
            raise ValueError("评估 cohort 不能为空")
        records = [sample[0] for sample in samples]
        environment_ids = {record.environment_id for record in records}
        strategy_ids = {record.strategy_id for record in records}
        if len(environment_ids) != 1 or len(strategy_ids) != 1:
            raise ValueError("Cohort 运行必须使用相同环境和策略")
        primary_record, primary_events, primary_artifacts = samples[0]
        metric_groups = [
            [event for event in events if event.type == "metric"]
            for _, events, _ in samples
        ]
        criteria = [
            self._evaluate_criterion(definition, primary_record, metric_groups[0], primary_artifacts, records, metric_groups)
            for definition in CRITERIA
        ]
        assessed = [criterion for criterion in criteria if criterion["status"] == "assessed"]
        assessed_weight = sum(criterion["weight"] for criterion in assessed)
        total_weight = sum(criterion.weight for criterion in CRITERIA)
        completeness = assessed_weight / total_weight
        mandatory_missing = any(criterion["mandatory"] and criterion["status"] != "assessed" for criterion in criteria)
        score: Optional[int] = None
        if completeness >= 0.8 and not mandatory_missing:
            weighted_score = sum(criterion["weight"] * criterion["score"] for criterion in assessed)
            score = round(100 * weighted_score / (4 * assessed_weight))
        report = {
            "schema_version": "1.0",
            "report_id": "evaluation-" + uuid4().hex[:12],
            "rubric_version": RUBRIC_VERSION,
            "run_ids": [record.run_id for record in records],
            "environment_ref": {"id": primary_record.environment_id, "revision": 1},
            "pipeline_ref": None,
            "evaluator": {"id": "builtin.mock-evaluator", "version": "1.0"},
            "capability": "mock",
            "criteria": criteria,
            "evidence_completeness": round(completeness, 3),
            "overall": {"status": "assessed" if score is not None else "not_assessed", "score": score},
            "missing_evidence": [criterion["missing_evidence"] for criterion in criteria if criterion["status"] == "not_assessed"],
            "limitations": [
                "Mock approximate dynamics; no physical transfer evidence.",
                "Different seeds on one Mock environment do not establish generalization.",
                "Evaluation does not enable real-robot control or Sim2Real deployment.",
            ],
            "sim2real": {"status": "blocked", "real_robot_control_enabled": False},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._report_path(report["report_id"]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report

    def get_report(self, report_id: str) -> Dict[str, Any]:
        path = self._report_path(report_id)
        if not path.exists():
            raise KeyError(report_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_reports(self) -> List[Dict[str, Any]]:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("evaluation-*.json")]
        return sorted(reports, key=lambda report: report.get("created_at", ""), reverse=True)

    def compare_reports(self, before_id: str, after_id: str) -> Dict[str, Any]:
        before = self.get_report(before_id)
        after = self.get_report(after_id)
        if before.get("rubric_version") != after.get("rubric_version"):
            raise ValueError("只能比较相同 rubric version 的评估报告")
        before_criteria = {item["criterion_id"]: item for item in before.get("criteria", [])}
        changes = []
        for current in after.get("criteria", []):
            previous = before_criteria.get(current["criterion_id"])
            if not previous:
                continue
            before_score = previous.get("score")
            after_score = current.get("score")
            delta = after_score - before_score if before_score is not None and after_score is not None else None
            if previous.get("status") != current.get("status"):
                change = "newly_assessed" if current.get("status") == "assessed" else "lost_evidence"
            elif delta is None or delta == 0:
                change = "unchanged"
            else:
                change = "improved" if delta > 0 else "regressed"
            changes.append({
                "criterion_id": current["criterion_id"],
                "name": current["name"],
                "before": {"status": previous.get("status"), "score": before_score},
                "after": {"status": current.get("status"), "score": after_score},
                "delta": delta,
                "change": change,
                "evidence_refs": current.get("evidence_refs", []),
            })
        before_overall = before.get("overall", {})
        after_overall = after.get("overall", {})
        overall_delta = None
        if before_overall.get("score") is not None and after_overall.get("score") is not None:
            overall_delta = after_overall["score"] - before_overall["score"]
        return {
            "before_report_id": before_id,
            "after_report_id": after_id,
            "rubric_version": after.get("rubric_version"),
            "overall": {
                "before": before_overall,
                "after": after_overall,
                "delta": overall_delta,
            },
            "completeness_delta": round(after.get("evidence_completeness", 0) - before.get("evidence_completeness", 0), 3),
            "criteria": changes,
        }

    def export_gate(self, run_id: str, report_id: Optional[str] = None) -> Dict[str, Any]:
        if not report_id:
            return {
                "eligible": False,
                "report_id": None,
                "reasons": ["A persisted evaluation report is required for model export."],
            }
        report = self.get_report(report_id)
        reasons = []
        if run_id not in report.get("run_ids", []):
            reasons.append("Evaluation report does not belong to this run.")
        if report.get("evidence_completeness", 0) < 0.8:
            reasons.append("Evidence completeness must be at least 0.8.")
        if report.get("overall", {}).get("status") != "assessed":
            reasons.append("Overall evaluation must be assessed.")
        mandatory_missing = any(
            criterion.get("mandatory") and criterion.get("status") != "assessed"
            for criterion in report.get("criteria", [])
        )
        if mandatory_missing:
            reasons.append("All mandatory criteria must be assessed.")
        return {"eligible": not reasons, "report_id": report_id, "reasons": reasons}

    def _report_path(self, report_id: str) -> Path:
        if not report_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in report_id):
            raise ValueError("无效 report_id")
        return self.root / (report_id + ".json")

    def _evaluate_criterion(
        self,
        definition: CriterionDefinition,
        record: RunRecord,
        metrics: List[RunEvent],
        artifacts: List[Dict[str, Any]],
        cohort_records: List[RunRecord],
        cohort_metrics: List[List[RunEvent]],
    ) -> Dict[str, Any]:
        result = {
            "criterion_id": definition.criterion_id,
            "name": definition.name,
            "weight": definition.weight,
            "mandatory": definition.mandatory,
            "status": "not_assessed",
            "score": None,
            "evidence_refs": [],
            "finding": "Evidence is insufficient for this criterion.",
            "missing_evidence": "",
        }
        if definition.criterion_id == "configuration_validity":
            return self._assessed(result, 4 if record.error is None else 0, ["run:%s:config" % record.run_id], "Run configuration passed adapter validation.")
        if definition.criterion_id == "task_success" and metrics:
            final = metrics[-1].payload.get("metrics", {})
            success = float(final.get("success_rate", 0))
            score = 4 if success >= 0.9 else 3 if success >= 0.8 else 2 if success >= 0.5 else 1
            return self._assessed(result, score, ["event:%s:success_rate" % metrics[-1].event_id], "Final Mock success rate is %.3f." % success)
        if definition.criterion_id == "reward_quality" and len(metrics) >= 2:
            first = float(metrics[0].payload.get("metrics", {}).get("mean_reward", 0))
            final = float(metrics[-1].payload.get("metrics", {}).get("mean_reward", 0))
            score = 3 if final > first else 1
            return self._assessed(result, score, ["event:%s:mean_reward" % metrics[0].event_id, "event:%s:mean_reward" % metrics[-1].event_id], "Mean reward changed from %.3f to %.3f." % (first, final))
        if definition.criterion_id == "termination_quality":
            termination_events = [event for event in metrics if "termination" in event.payload]
            if termination_events:
                reason = termination_events[-1].payload["termination"].get("reason", "unknown")
                score = 3 if reason != "unknown" else 0
                return self._assessed(result, score, ["event:%s:termination" % termination_events[-1].event_id], "Terminal reason is classified as %s." % reason)
            result["missing_evidence"] = "termination classification channel"
            return result
        if definition.criterion_id == "safety_constraints":
            collision_events = [event for event in metrics if "collision" in event.payload]
            if collision_events:
                violations = sum(1 for event in collision_events if event.payload["collision"].get("detected"))
                score = 3 if violations == 0 else 0
                return self._assessed(result, score, ["run:%s:collision" % record.run_id], "%d geometric collision approximation events detected." % violations)
            result["missing_evidence"] = "joint-limit and collision diagnostic channels"
            return result
        if definition.criterion_id == "efficiency" and metrics:
            completed = record.status == RunStatus.COMPLETED and len(metrics) <= record.max_steps
            return self._assessed(result, 3 if completed else 1, ["run:%s:status" % record.run_id], "%d metric samples were recorded within a %d-step budget." % (len(metrics), record.max_steps))
        if definition.criterion_id == "artifact_completeness" and artifacts:
            return self._assessed(result, 2, ["run:%s:artifacts" % record.run_id], "%d run artifacts are present; checksum verification is not yet available." % len(artifacts))
        if definition.criterion_id == "reproducibility" and len(cohort_records) >= 2 and all(cohort_metrics):
            success_rates = [float(events[-1].payload.get("metrics", {}).get("success_rate", 0)) for events in cohort_metrics]
            spread = max(success_rates) - min(success_rates)
            score = 4 if spread <= 0.05 else 3 if spread <= 0.15 else 1
            evidence_refs = ["event:%s:success_rate" % events[-1].event_id for events in cohort_metrics]
            finding = "%d Mock runs across seeds %s have a final success-rate spread of %.3f." % (
                len(cohort_records),
                ", ".join(str(item.seed) for item in cohort_records),
                spread,
            )
            return self._assessed(result, score, evidence_refs, finding)

        missing = {
            "stability": "action trajectory and late-run oscillation analysis",
            "reproducibility": "repeated run with matching revision and seed",
            "generalization": "declared held-out target set",
            "artifact_completeness": "run artifacts and checksum metadata",
            "task_success": "metric success channel",
            "reward_quality": "at least two reward metric samples",
        }
        result["missing_evidence"] = missing.get(definition.criterion_id, "required evidence channel")
        return result

    @staticmethod
    def _assessed(result: Dict[str, Any], score: int, evidence_refs: List[str], finding: str) -> Dict[str, Any]:
        result.update({"status": "assessed", "score": score, "evidence_refs": evidence_refs, "finding": finding, "missing_evidence": ""})
        return result