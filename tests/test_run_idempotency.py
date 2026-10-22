import hashlib
import unittest

from robotics.runs import (
    ExecutionMode,
    IdempotencyConflict,
    IdempotentRunCreateCommand,
    InvalidIdempotentRunCreateRequest,
    RunKind,
    canonical_run_create_json,
    fingerprint_run_create,
)


class RunIdempotencyDomainTest(unittest.TestCase):
    def command(self, **changes):
        values = {
            "user_id": "user-1",
            "idempotency_key": "create-1",
            "project_id": "project-1",
            "execution_mode": ExecutionMode.MOCK,
            "parameters": {"seed": 7, "nested": {"b": 2, "a": 1}},
            "project_config_snapshot": {"scene": "push", "version": 3},
            "code": "print('train')\n",
            "run_kind": RunKind.TRAIN,
        }
        values.update(changes)
        return IdempotentRunCreateCommand(**values)

    def test_fingerprint_is_stable_for_dictionary_key_order(self):
        first = self.command(
            parameters={"seed": 7, "nested": {"b": 2, "a": 1}}
        )
        second = self.command(
            parameters={"nested": {"a": 1, "b": 2}, "seed": 7}
        )
        self.assertEqual(
            canonical_run_create_json(first), canonical_run_create_json(second)
        )
        self.assertEqual(
            fingerprint_run_create(first), fingerprint_run_create(second)
        )
        self.assertEqual(len(fingerprint_run_create(first)), 64)

    def test_fingerprint_is_sha256_of_canonical_utf8_json(self):
        command = self.command(code="机械臂 = 'push'\n")
        expected = hashlib.sha256(
            canonical_run_create_json(command).encode("utf-8")
        ).hexdigest()
        self.assertEqual(fingerprint_run_create(command), expected)

    def test_every_run_affecting_business_field_changes_fingerprint(self):
        base = fingerprint_run_create(self.command())
        variants = [
            self.command(user_id="user-2"),
            self.command(project_id="project-2"),
            self.command(execution_mode=ExecutionMode.SIMULATION),
            self.command(parameters={"seed": 8}),
            self.command(project_config_snapshot={"scene": "reach"}),
            self.command(code="print('changed')\n"),
            self.command(run_kind=RunKind.PLAY),
        ]
        for variant in variants:
            self.assertNotEqual(base, fingerprint_run_create(variant))

    def test_idempotency_key_is_scope_not_request_content(self):
        first = self.command(idempotency_key="key-a")
        second = self.command(idempotency_key="key-b")
        self.assertEqual(
            fingerprint_run_create(first), fingerprint_run_create(second)
        )

    def test_formal_fingerprint_contains_only_trusted_request_facts(self):
        values = {
            "user_id": "user-1",
            "idempotency_key": "key-a",
            "project_id": "project-1",
            "class_id": "class-1",
            "project_config_version": 4,
            "execution_mode": ExecutionMode.SIMULATION,
            "run_kind": RunKind.TRAIN,
            "parameters": {"seed": 7},
            "code": "code",
        }
        first = IdempotentRunCreateCommand(**values)
        second = IdempotentRunCreateCommand(
            **dict(values, idempotency_key="key-b", project_config_snapshot={"ignored": True})
        )
        payload = canonical_run_create_json(first)
        self.assertEqual(fingerprint_run_create(first), fingerprint_run_create(second))
        self.assertNotIn("idempotency", payload)
        self.assertNotIn("snapshot", payload)
        self.assertIn('"seed":7', payload)

    def test_every_formal_request_fact_changes_the_fingerprint(self):
        values = {
            "user_id": "user-1",
            "idempotency_key": "stable-key",
            "project_id": "project-1",
            "class_id": "class-1",
            "project_config_version": 4,
            "execution_mode": ExecutionMode.SIMULATION,
            "run_kind": RunKind.TRAIN,
            "parameters": {"seed": 7},
            "code": "code",
        }
        baseline = fingerprint_run_create(IdempotentRunCreateCommand(**values))
        variants = (
            {"user_id": "user-2"},
            {"project_id": "project-2"},
            {"class_id": "class-2"},
            {"project_config_version": 5},
            {"execution_mode": ExecutionMode.MOCK},
            {"run_kind": RunKind.PLAY},
            {"parameters": {"seed": 8}},
            {"code": "changed"},
        )
        for changes in variants:
            with self.subTest(field=next(iter(changes))):
                command = IdempotentRunCreateCommand(
                    **dict(values, **changes)
                )
                self.assertNotEqual(baseline, fingerprint_run_create(command))

    def test_formal_fingerprint_rejects_noncanonical_parameters(self):
        for parameters in ({"seed": True}, {"seed": 1, "num_envs": 8}):
            with self.subTest(parameters=parameters):
                with self.assertRaises(InvalidIdempotentRunCreateRequest):
                    IdempotentRunCreateCommand(
                        user_id="user-1",
                        idempotency_key="key",
                        project_id="project-1",
                        class_id="class-1",
                        project_config_version=1,
                        execution_mode=ExecutionMode.SIMULATION,
                        run_kind=RunKind.TRAIN,
                        parameters=parameters,
                    )

    def test_command_deep_copies_inputs_and_returned_kwargs(self):
        parameters = {"nested": {"seed": 7}}
        snapshot = {"scene": {"name": "push"}}
        command = self.command(
            parameters=parameters, project_config_snapshot=snapshot
        )
        parameters["nested"]["seed"] = 99
        snapshot["scene"]["name"] = "changed"
        kwargs = command.create_kwargs()
        kwargs["parameters"]["nested"]["seed"] = 55
        self.assertEqual(command.parameters["nested"]["seed"], 7)
        self.assertEqual(
            command.project_config_snapshot["scene"]["name"], "push"
        )

    def test_invalid_or_non_json_input_is_rejected_without_content_echo(self):
        invalid_values = [
            {"user_id": ""},
            {"idempotency_key": "x" * 256},
            {"execution_mode": "mock"},
            {"parameters": {"bad": object()}},
            {"parameters": {"bad": float("nan")}},
        ]
        for changes in invalid_values:
            with self.subTest(changes=list(changes)):
                with self.assertRaises(InvalidIdempotentRunCreateRequest):
                    self.command(**changes)

    def test_conflict_exposes_scope_but_not_request_body(self):
        conflict = IdempotencyConflict("user-1", "key-1")
        self.assertEqual(conflict.user_id, "user-1")
        self.assertEqual(conflict.idempotency_key, "key-1")
        self.assertNotIn("parameters", str(conflict))


if __name__ == "__main__":
    unittest.main()
