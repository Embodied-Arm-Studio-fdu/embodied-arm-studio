import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "callback_dispatcher.py"
SPEC = importlib.util.spec_from_file_location("callback_dispatcher_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def args(cli_token=None, token_file=None):
    return argparse.Namespace(
        service_token=cli_token, service_token_file=token_file
    )


class CallbackDispatcherTokenTests(unittest.TestCase):
    def test_environment_token_is_read_without_argv_value(self):
        token = "test-env-token-not-a-real-secret"
        argv = ["callback_dispatcher.py", "--backend-base-url", "http://backend"]
        self.assertNotIn(token, argv)
        self.assertEqual(
            token,
            MODULE.resolve_service_token(
                args(), {MODULE.TOKEN_ENVIRONMENT_VARIABLE: token}
            ),
        )

    def test_mode_0600_secret_file_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text("test-file-token-not-a-real-secret\n", encoding="utf-8")
            path.chmod(0o600)
            self.assertEqual(
                "test-file-token-not-a-real-secret",
                MODULE.resolve_service_token(args(token_file=str(path)), {}),
            )

    def test_secret_file_rejects_insecure_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text("test-token", encoding="utf-8")
            path.chmod(0o640)
            with self.assertRaisesRegex(ValueError, "mode 0600"):
                MODULE.resolve_service_token(args(token_file=str(path)), {})

    def test_secret_file_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_text("test-token", encoding="utf-8")
            target.chmod(0o600)
            link = Path(directory) / "token-link"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "cannot be read"):
                MODULE.resolve_service_token(args(token_file=str(link)), {})

    def test_all_token_sources_are_mutually_exclusive(self):
        environment = {MODULE.TOKEN_ENVIRONMENT_VARIABLE: "test-env-token"}
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            MODULE.resolve_service_token(args(cli_token="test-cli-token"), environment)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            MODULE.resolve_service_token(args(token_file="unused"), environment)

    def test_missing_token_preserves_existing_optional_contract(self):
        self.assertIsNone(MODULE.resolve_service_token(args(), {}))

    def test_errors_and_help_do_not_disclose_token(self):
        token = "test-sensitive-token-not-a-real-secret"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(ValueError) as raised:
                MODULE.resolve_service_token(
                    args(cli_token=token),
                    {MODULE.TOKEN_ENVIRONMENT_VARIABLE: token},
                )
        combined = stdout.getvalue() + stderr.getvalue() + str(raised.exception)
        self.assertNotIn(token, combined)

    def test_legacy_cli_is_explicitly_marked_in_help(self):
        help_text = MODULE.build_parser().format_help()
        self.assertIn("legacy/insecure", help_text)
        self.assertIn(MODULE.TOKEN_ENVIRONMENT_VARIABLE, help_text)


if __name__ == "__main__":
    unittest.main()
