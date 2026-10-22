import unittest

from tests.postgres_safety import (
    database_fingerprint,
    fingerprint_text,
    safe_test_database_url,
)


SAFE_URL = "postgresql+psycopg://tester:secret@127.0.0.1:55432/embodied_arm_ci_test"
RUNTIME_URL = "postgresql+psycopg://runtime:secret@127.0.0.1:5433/embodied_arm_test"


class PostgreSQLTestSafetyTest(unittest.TestCase):
    def test_ordinary_tests_do_not_select_postgresql(self):
        self.assertIsNone(safe_test_database_url({}))

    def test_url_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(RuntimeError, "POSTGRES_TEST_OPT_IN=1"):
            safe_test_database_url({"TEST_DATABASE_URL": SAFE_URL})

    def test_opt_in_requires_explicit_url(self):
        with self.assertRaisesRegex(RuntimeError, "requires TEST_DATABASE_URL"):
            safe_test_database_url({"POSTGRES_TEST_OPT_IN": "1"})

    def test_non_postgresql_and_non_test_targets_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "PostgreSQL URL"):
            safe_test_database_url({
                "POSTGRES_TEST_OPT_IN": "1",
                "TEST_DATABASE_URL": "sqlite:///tmp/test.db",
            })
        with self.assertRaisesRegex(RuntimeError, "must contain 'test'"):
            safe_test_database_url({
                "POSTGRES_TEST_OPT_IN": "1",
                "TEST_DATABASE_URL": "postgresql://tester:x@db/app",
            })

    def test_known_midterm_runtime_database_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "known midterm runtime"):
            safe_test_database_url({
                "POSTGRES_TEST_OPT_IN": "1",
                "TEST_DATABASE_URL": RUNTIME_URL,
            })

    def test_runtime_url_fingerprint_match_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "matches runtime"):
            safe_test_database_url({
                "POSTGRES_TEST_OPT_IN": "1",
                "TEST_DATABASE_URL": SAFE_URL,
                "DATABASE_URL": SAFE_URL.replace("secret", "different"),
            })

    def test_explicit_midterm_fingerprint_match_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "matches midterm runtime"):
            safe_test_database_url({
                "POSTGRES_TEST_OPT_IN": "1",
                "TEST_DATABASE_URL": SAFE_URL,
                "MIDTERM_RUNTIME_DATABASE_FINGERPRINT": fingerprint_text(SAFE_URL),
            })

    def test_distinct_explicit_test_target_is_accepted_without_connection(self):
        resolved = safe_test_database_url({
            "POSTGRES_TEST_OPT_IN": "1",
            "TEST_DATABASE_URL": SAFE_URL,
            "DATABASE_URL": RUNTIME_URL,
        })
        self.assertEqual(resolved.database, "embodied_arm_ci_test")

    def test_fingerprint_normalizes_loopback_and_omits_password(self):
        left = database_fingerprint(SAFE_URL)
        right = database_fingerprint(
            "postgresql://tester:other@localhost:55432/embodied_arm_ci_test"
        )
        self.assertEqual(left, right)
        self.assertNotIn("secret", fingerprint_text(SAFE_URL))


if __name__ == "__main__":
    unittest.main()
