from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.router_compare import (
    DEFAULT_DATASET,
    ROUTING_METRICS,
    build_router_records,
    main,
    run_router_comparison,
)
from evaluation.rag_v2_metrics import load_scenarios
from rag_v2.models import SessionState
from rag_v2.routing import QueryRouter


class RouterComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = load_scenarios(DEFAULT_DATASET)
        cls.records = build_router_records(cls.scenarios)
        cls.index = {item["record_id"]: item for item in cls.records}

    def test_executes_the_complete_dataset_without_providers(self) -> None:
        self.assertEqual(len(self.scenarios), 32)
        self.assertEqual(len(self.records), 66)
        self.assertTrue(all(item["legacy"] is not None for item in self.records))
        self.assertTrue(all(item["v2"] is not None for item in self.records))

    def test_verified_fixture_state_is_used_for_follow_up(self) -> None:
        record = self.index["mt_001_bank_product_inheritance/t2"]
        self.assertEqual(record["legacy"]["inherited_context"], {})
        self.assertEqual(
            record["v2"]["inherited_context"]["banks"],
            ["Ziraat Katılım"],
        )
        self.assertEqual(
            record["v2"]["inherited_context"]["product_types"],
            ["konut finansmanı"],
        )
        self.assertEqual(
            record["v2"]["inherited_context"]["offer_ids"],
            ["eval:mt001:offer:1"],
        )

    def test_session_keys_prevent_fixture_state_leakage(self) -> None:
        record = self.index["mt_030_session_bank_isolation/t2"]
        self.assertEqual(record["v2"]["banks"], [])
        self.assertEqual(record["v2"]["product_types"], [])
        self.assertEqual(record["v2"]["offer_ids"], [])
        self.assertTrue(
            set(record["v2"]["inherited_context"]).issubset({"scope"})
        )
        self.assertNotIn("Türkiye Finans", record["v2"]["standalone_query"])

    def test_untrusted_prior_answer_is_not_used_as_state(self) -> None:
        record = self.index["mt_028_wrong_prior_numeric_answer/t2"]
        self.assertNotIn("120", record["v2"]["standalone_query"])
        self.assertEqual(
            record["v2"]["inherited_context"]["banks"],
            ["Vakıf Katılım"],
        )

    def test_actual_standalone_text_is_not_copied_from_label(self) -> None:
        record = self.index["mt_001_bank_product_inheritance/t1"]
        query = next(
            item
            for item in self.scenarios[0]["turns"]
            if item["turn_id"] == "t1"
        )["user_query"]
        actual = QueryRouter().resolve(query, state=SessionState()).standalone_query
        self.assertEqual(record["v2"]["standalone_query"], actual)

    def test_report_scores_only_routing_capabilities(self) -> None:
        report = run_router_comparison(
            self.scenarios,
            dataset_path=DEFAULT_DATASET,
        )
        self.assertEqual(report["dataset"]["scenario_count"], 32)
        self.assertEqual(report["dataset"]["turn_count"], 66)
        self.assertEqual(
            set(report["metrics"]["v2_deterministic_router"]),
            set(ROUTING_METRICS),
        )
        self.assertEqual(
            report["metrics"]["v2_deterministic_router"]
            ["standalone_query_accuracy"]["evaluated_records"],
            66,
        )
        self.assertGreater(
            report["metrics"]["v2_deterministic_router"]
            ["standalone_context_accuracy"]["evaluated_records"],
            50,
        )
        self.assertGreater(
            report["metrics"]["v2_deterministic_router"]
            ["standalone_context_accuracy"]["value"],
            report["metrics"]["legacy_no_memory"]
            ["standalone_context_accuracy"]["value"],
        )
        self.assertEqual(
            report["metrics"]["v2_deterministic_router"]
            ["inherited_date_accuracy"]["evaluated_records"],
            2,
        )
        self.assertEqual(
            report["unexecuted_metrics"]["recall_at_1"]["status"],
            "unavailable",
        )
        self.assertEqual(
            report["unexecuted_metrics"]["numeric_accuracy"]["status"],
            "unavailable",
        )
        self.assertEqual(
            report["metrics"]["v2_deterministic_router"]
            ["session_isolation_pass_rate"]["value"],
            1.0,
        )
        self.assertNotIn("records", report)

    def test_comparison_is_deterministic(self) -> None:
        first = run_router_comparison(self.scenarios, dataset_path=DEFAULT_DATASET)
        second = run_router_comparison(self.scenarios, dataset_path=DEFAULT_DATASET)
        self.assertEqual(first, second)

    def test_cli_writes_the_same_report_it_prints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "router-report.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--dataset",
                        str(DEFAULT_DATASET),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                json.loads(stdout.getvalue()),
            )


if __name__ == "__main__":
    unittest.main()
