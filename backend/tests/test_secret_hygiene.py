from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.secret_hygiene import (
    PROJECT_DIR,
    _eligible_path,
    build_report,
    scan_runtime_logs,
    scan_text,
    source_paths,
)


class SecretDetectorTest(unittest.TestCase):
    def test_provider_keys_are_detected_without_returning_the_value(self) -> None:
        provider_value = "sk-" + "evren-" + ("a" * 32)
        findings = scan_text(
            "EVREN_API_KEY=" + provider_value,
            "fixture.txt",
        )
        rendered = json.dumps(
            [
                {
                    "detector": item.detector,
                    "location": item.location,
                    "snapshot": item.snapshot,
                }
                for item in findings
            ]
        )
        self.assertEqual([item.detector for item in findings], ["evren_api_key"])
        self.assertNotIn(provider_value, rendered)

    def test_sensitive_quoted_literal_is_detected(self) -> None:
        sensitive_value = "A9" * 12
        findings = scan_text(
            "database_password = \"" + sensitive_value + "\"",
            "settings.py",
        )
        self.assertEqual(
            [item.detector for item in findings],
            ["sensitive_literal"],
        )

    def test_unquoted_environment_secret_is_detected_and_redacted(self) -> None:
        sensitive_value = "LiveSecret" + ("7" * 18)
        findings = scan_text(
            "DB_PASSWORD=" + sensitive_value,
            "settings.env.local",
        )
        self.assertEqual(
            [item.detector for item in findings],
            ["sensitive_env_literal"],
        )
        self.assertNotIn(sensitive_value, repr(findings))

    def test_database_url_and_bearer_literals_are_detected(self) -> None:
        database_secret = "unit-test-db-secret-" + ("4" * 12)
        bearer_secret = "unit-test-bearer-" + ("5" * 16)
        text = (
            "postgresql://app:" + database_secret + "@db.example/app\n"
            "Authorization: Bearer " + bearer_secret
        )
        findings = scan_text(text, "fixture.txt")
        self.assertEqual(
            {item.detector for item in findings},
            {"database_url_credential", "bearer_token"},
        )
        self.assertNotIn(database_secret, repr(findings))
        self.assertNotIn(bearer_secret, repr(findings))

    def test_only_the_documented_root_dotenv_is_excluded(self) -> None:
        root_env = PROJECT_DIR / ".env"
        nested_env = PROJECT_DIR / "config" / ".env"
        local_env = PROJECT_DIR / ".env.production"
        self.assertFalse(_eligible_path(root_env, PROJECT_DIR))
        self.assertTrue(_eligible_path(nested_env, PROJECT_DIR))
        self.assertTrue(_eligible_path(local_env, PROJECT_DIR))

    def test_obvious_fixture_and_environment_names_are_allowed(self) -> None:
        text = "\n".join(
            (
                'admin_password = "runtime-password-long"',
                'ADMIN_API_KEY_ENV = "HITITFINLEX_ADMIN_API_KEY"',
                'secret = "unit-test-secret-value"',
            )
        )
        self.assertEqual(scan_text(text, "tests/fixture.py"), [])

    def test_runtime_log_findings_are_redacted(self) -> None:
        provider_value = "qdr-" + ("x" * 32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "service.log").write_text(provider_value, encoding="utf-8")
            findings = scan_runtime_logs(root)
        self.assertEqual([item.detector for item in findings], ["qdrant_api_key"])
        self.assertNotIn(provider_value, repr(findings))


class RepositorySecretHygieneTest(unittest.TestCase):
    def test_root_dotenv_is_not_a_source_asset(self) -> None:
        paths = source_paths(PROJECT_DIR)
        self.assertNotIn(PROJECT_DIR / ".env", paths)
        self.assertIn(PROJECT_DIR / ".env.example", paths)

    def test_source_logs_and_git_history_have_no_detected_credentials(self) -> None:
        report = build_report(PROJECT_DIR)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["counts"]["total"], 0)
        self.assertEqual(report["findings"], [])
        self.assertTrue(report["policy"]["matched_values_are_never_reported"])
        self.assertTrue(report["policy"]["git_history_scanned"])


if __name__ == "__main__":
    unittest.main()
