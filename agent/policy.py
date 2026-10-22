from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


ASSISTANCE_LEVELS = {
    1: "explain_only",
    2: "socratic_hints",
    3: "review_student_proposal",
    4: "suggest_configuration",
    5: "workspace_agent",
}

ASSISTANCE_CAPABILITIES = {
    1: {"explain"},
    2: {"explain", "socratic_hints"},
    3: {"explain", "socratic_hints", "review_student_proposal"},
    4: {
        "explain",
        "socratic_hints",
        "review_student_proposal",
        "read_page_context",
        "suggest_configuration",
    },
    5: {
        "explain",
        "socratic_hints",
        "review_student_proposal",
        "read_page_context",
        "read_workspace",
        "read_scene",
        "read_run_evidence",
        "suggest_configuration",
        "propose_patch",
    },
}


@dataclass(frozen=True)
class PolicyDecision:
    outcome: str
    reason: str
    intent: str
    assistance_level: int


@dataclass(frozen=True)
class StudentDesign:
    observation: str
    action: str
    reward: str
    termination: str
    expected_result: str = ""

    def is_complete(self) -> bool:
        return all(
            item.strip()
            for item in (self.observation, self.action, self.reward, self.termination)
        )


class TeachingPolicy:
    _DEFAULT_DENIED_PATTERNS: Dict[str, List[str]] = {
        "generic_shell": [r"(?:执行|运行|打开).{0,12}(?:shell|终端|命令行)", r"\b(?:bash|sudo|rm -rf)\b"],
        "secret_access": [r"(?:读取|显示|泄露|告诉我).{0,12}(?:api.?key|密钥|token|密码)"],
        "real_robot_control": [r"(?:直接|立即|绕过.{0,8})?(?:控制|驱动).{0,12}(?:真实|实体).{0,6}(?:机械臂|机器人)"],
        "bypass_approval": [r"(?:绕过|跳过|取消|自动).{0,12}(?:审批|批准|确认|安全限制|关节限位)"],
        "fabricate_evidence": [r"(?:伪造|篡改|删除).{0,12}(?:训练|成功|指标|证据|失败运行)"],
    }
    _FULL_IMPLEMENTATION_PATTERNS = [
        r"(?:全部|整个|完整).{0,18}(?:设计|实现|答案|代码)",
        r"替我.{0,16}(?:完成|实现|写好).{0,20}(?:observation|action|reward|termination|环境)",
        r"不要解释.{0,12}(?:直接|只).{0,8}(?:答案|代码|实现)",
    ]
    _PATCH_PATTERNS = [r"(?:生成|给出|创建|写).{0,10}(?:patch|补丁|diff)"]

    def __init__(self, blocked_intents_path: Optional[Path] = None) -> None:
        self._denied_patterns = self._DEFAULT_DENIED_PATTERNS
        self._denied_reasons: Dict[str, str] = {}
        if blocked_intents_path is not None:
            document = json.loads(blocked_intents_path.read_text(encoding="utf-8"))
            intents = document.get("intents")
            if not isinstance(intents, list) or not intents:
                raise ValueError("blocked intent 配置必须包含非空 intents 列表")
            self._denied_patterns = {}
            for intent in intents:
                intent_id = intent.get("id")
                patterns = intent.get("patterns")
                if not isinstance(intent_id, str) or not isinstance(patterns, list) or not patterns:
                    raise ValueError("blocked intent 必须声明 id 和 patterns")
                self._denied_patterns[intent_id] = patterns
                self._denied_reasons[intent_id] = str(intent.get("reason", "该请求超出 Teaching Agent 的安全权限。"))

    def evaluate_message(
        self,
        content: str,
        assistance_level: int,
        design: Optional[StudentDesign] = None,
        full_assistance_authorized: bool = False,
    ) -> PolicyDecision:
        if assistance_level not in ASSISTANCE_LEVELS:
            return PolicyDecision(
                "deny",
                "无效的 Agent 权限档位。可用档位为 Level 1 到 Level 5。",
                "invalid_assistance_level",
                assistance_level,
            )
        normalized = content.casefold()
        for intent, patterns in self._denied_patterns.items():
            if self._matches(normalized, patterns):
                return PolicyDecision("deny", self._denied_reasons.get(intent, "该请求超出 Teaching Agent 的安全权限。"), intent, assistance_level)

        if self._matches(normalized, self._FULL_IMPLEMENTATION_PATTERNS):
            if assistance_level < 5:
                return PolicyDecision(
                    "deny",
                    "完整实现只能由 Level 5 生成受控 Patch 提案，不能直接修改文件。",
                    "full_implementation",
                    assistance_level,
                )
            if design is None or not design.is_complete():
                return PolicyDecision(
                    "require_student_design",
                    "请先提交 Observation、Action、Reward 和 Termination 四项设计草案。",
                    "full_implementation",
                    assistance_level,
                )

        if self._matches(normalized, self._PATCH_PATTERNS):
            if "propose_patch" not in ASSISTANCE_CAPABILITIES[assistance_level]:
                return PolicyDecision(
                    "deny",
                    "当前 Agent 权限只允许解释、提示或配置建议，不能生成 Patch。",
                    "patch_request",
                    assistance_level,
                )
            if design is None or not design.is_complete():
                return PolicyDecision(
                    "require_student_design",
                    "生成 Patch 前需提交完整的环境设计草案。",
                    "patch_request",
                    assistance_level,
                )

        return PolicyDecision("allow", "请求符合当前教学与安全策略。", "teaching_help", assistance_level)

    @staticmethod
    def _matches(content: str, patterns: List[str]) -> bool:
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns)