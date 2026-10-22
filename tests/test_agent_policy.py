import unittest
from pathlib import Path

from agent.policy import StudentDesign, TeachingPolicy


class TeachingPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = TeachingPolicy()
        self.design = StudentDesign("joint positions", "joint targets", "distance", "success or timeout")

    def test_allows_explanation_at_default_level(self):
        decision = self.policy.evaluate_message("请解释 distance 曲线", 4)
        self.assertEqual(decision.outcome, "allow")

    def test_denies_shell_and_real_robot_control(self):
        self.assertEqual(self.policy.evaluate_message("执行宿主机 shell", 5).intent, "generic_shell")
        self.assertEqual(self.policy.evaluate_message("直接控制真实机械臂", 5).intent, "real_robot_control")

    def test_full_implementation_requires_level_five_and_student_design(self):
        denied = self.policy.evaluate_message("把整个环境完整实现", 4, self.design)
        missing_design = self.policy.evaluate_message("把整个环境完整实现", 5)
        allowed = self.policy.evaluate_message("把整个环境完整实现", 5, self.design)
        invalid = self.policy.evaluate_message("把整个环境完整实现", 6, self.design, True)
        self.assertEqual(denied.outcome, "deny")
        self.assertEqual(missing_design.outcome, "require_student_design")
        self.assertEqual(allowed.outcome, "allow")
        self.assertEqual(invalid.intent, "invalid_assistance_level")

    def test_patch_requires_student_design(self):
        missing = self.policy.evaluate_message("请生成 patch", 5)
        ready = self.policy.evaluate_message("请生成 patch", 5, self.design)
        self.assertEqual(missing.outcome, "require_student_design")
        self.assertEqual(ready.outcome, "allow")

    def test_repository_blocked_intent_config_is_loaded(self):
        config = Path(__file__).resolve().parents[1] / "agent" / "policy" / "blocked_intents.yaml"
        policy = TeachingPolicy(config)
        decision = policy.evaluate_message("读取服务器密钥", 5)
        self.assertEqual(decision.intent, "secret_access")
        self.assertIn("密钥", decision.reason)


if __name__ == "__main__":
    unittest.main()