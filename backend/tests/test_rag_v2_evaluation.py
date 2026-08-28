from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluation.rag_v2_metrics import (
    DATASET_LABEL,
    DatasetValidationError,
    compare_records,
    load_comparison,
    load_scenarios,
    merge_dataset_and_comparison,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    validate_scenarios,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "evaluation" / "multiturn_scenarios.silver_unverified.json"


class ScenarioSchemaTest(unittest.TestCase):
    def test_checked_in_dataset_is_silver_and_covers_required_categories(self) -> None:
        scenarios = load_scenarios(DATASET_PATH)
        self.assertGreaterEqual(len(scenarios), 30)
        self.assertTrue(all(item["label"] == DATASET_LABEL for item in scenarios))
        self.assertTrue(all(len(item["turns"]) >= 2 for item in scenarios))
        categories = {category for item in scenarios for category in item["categories"]}
        self.assertTrue(
            {
                "bank_inheritance",
                "product_inheritance",
                "date_inheritance",
                "year_inheritance",
                "ordinal_reference",
                "topic_change",
                "scope_change",
                "ambiguous_reference",
                "no_data",
                "wrong_prior_answer",
                "session_isolation",
            }.issubset(categories)
        )
        turn_count = sum(len(item["turns"]) for item in scenarios)
        self.assertGreaterEqual(turn_count, 60)

    def test_non_silver_label_is_rejected(self) -> None:
        scenarios = load_scenarios(DATASET_PATH)
        scenarios[0]["label"] = "gold"
        with self.assertRaisesRegex(DatasetValidationError, "label must be exactly silver_unverified"):
            validate_scenarios(scenarios)

    def test_missing_route_field_is_rejected(self) -> None:
        scenarios = load_scenarios(DATASET_PATH)
        del scenarios[0]["turns"][0]["expected"]["scope"]
        with self.assertRaisesRegex(DatasetValidationError, "missing expected keys"):
            validate_scenarios(scenarios)

    def test_comparison_rejects_conflicting_common_label(self) -> None:
        scenarios = load_scenarios(DATASET_PATH)
        comparison = {
            "dataset_label": DATASET_LABEL,
            "records": [
                {
                    "record_id": "mt_001_bank_product_inheritance/t1",
                    "labels": {"standalone_query": "conflicting query"},
                    "legacy": {},
                    "v2": {},
                }
            ],
        }
        with self.assertRaisesRegex(DatasetValidationError, "conflicts"):
            merge_dataset_and_comparison(scenarios, comparison)

    def test_comparison_file_rejects_duplicate_record_ids(self) -> None:
        payload = {
            "dataset_label": DATASET_LABEL,
            "records": [
                {"record_id": "same", "legacy": {}, "v2": {}},
                {"record_id": "same", "legacy": {}, "v2": {}},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DatasetValidationError, "duplicate comparison record_id"):
                load_comparison(path)

    def test_invalid_calendar_date_is_rejected(self) -> None:
        scenarios = load_scenarios(DATASET_PATH)
        scenarios[0]["turns"][0]["expected"]["date_from"] = "2025-02-30"
        with self.assertRaisesRegex(DatasetValidationError, "valid calendar date"):
            validate_scenarios(scenarios)

    def test_invalid_ranked_offer_fixture_is_rejected(self) -> None:
        scenarios = load_scenarios(DATASET_PATH)
        scenarios[0]["turns"][0]["context_after_turn"]["ranked_offers"] = [
            {"offer_id": "first", "bank": "Bank A", "rank": 1},
            {"offer_id": "second", "bank": "Bank B", "rank": 1},
        ]
        with self.assertRaisesRegex(DatasetValidationError, "unique ranks and ids"):
            validate_scenarios(scenarios)


class RankingFormulaTest(unittest.TestCase):
    def test_recall_deduplicates_ranked_ids(self) -> None:
        ranked = ["noise", "a", "a", "b"]
        relevant = ["a", "b"]
        self.assertEqual(recall_at_k(ranked, relevant, 1), 0.0)
        self.assertEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(recall_at_k(ranked, relevant, 3), 1.0)

    def test_mrr_uses_first_relevant_rank(self) -> None:
        self.assertEqual(reciprocal_rank_at_k(["noise", "a", "b"], ["a", "b"], 10), 0.5)
        self.assertEqual(reciprocal_rank_at_k(["noise", "other"], ["a"], 10), 0.0)

    def test_binary_ndcg_formula(self) -> None:
        actual = ndcg_at_k(["noise", "a", "b"], ["a", "b"], 10)
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        ideal = 1.0 + 1.0 / math.log2(3)
        self.assertAlmostEqual(actual, dcg / ideal)

    def test_graded_ndcg_formula(self) -> None:
        actual = ndcg_at_k(["noise", "b", "a"], ["a", "b"], 10, {"a": 3, "b": 1, "noise": 99})
        dcg = 1.0 / math.log2(3) + 7.0 / math.log2(4)
        ideal = 7.0 + 1.0 / math.log2(3)
        self.assertAlmostEqual(actual, dcg / ideal)

    def test_empty_relevance_is_not_silently_scored(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k(["a"], [], 1)
        with self.assertRaises(ValueError):
            reciprocal_rank_at_k(["a"], [], 10)
        with self.assertRaises(ValueError):
            ndcg_at_k(["a"], [], 10)


def _paired_metric_records() -> list[dict[str, object]]:
    return [
        {
            "record_id": "r1",
            "labels": {
                "standalone_query": "Banka A vadesi nedir?",
                "inherited_context": {"banks": ["Banka A"], "scope": "historical", "year": 2025},
                "cleared_fields": [],
                "needs_clarification": False,
                "relevant_ids": ["a", "b"],
                "relevance_grades": {"a": 2, "b": 1},
                "citation": {"required_ids": ["a"], "allowed_ids": ["a", "b"]},
                "numbers": {"required": ["36 ay"], "allowed": ["36 ay"]},
                "isolation_expected": True,
            },
            "legacy": {
                "standalone_query": "Yanlis sorgu",
                "inherited_context": {"banks": ["Banka A"], "scope": "historical", "year": 2025},
                "cleared_fields": [],
                "needs_clarification": False,
                "retrieved_ids": ["noise", "a", "a", "b"],
                "answer": "Vade 36 ay [S1].",
                "evidence": [{"source_id": "S1", "offer_id": "a"}],
                "isolation_passed": True,
            },
            "v2": {
                "standalone_query": "Banka A vadesi nedir!",
                "inherited_context": {"banks": ["Banka A"], "scope": "historical", "year": 2025},
                "cleared_fields": [],
                "needs_clarification": False,
                "retrieved_ids": ["a", "b"],
                "cited_ids": ["a"],
                "answer_numbers": ["36 ay"],
                "isolation_passed": True,
            },
        },
        {
            "record_id": "r2",
            "labels": {
                "standalone_query": "Urun B tutari nedir?",
                "inherited_context": {"product_types": ["Urun B"]},
                "cleared_fields": ["banks"],
                "needs_clarification": True,
                "relevant_ids": ["c"],
                "citation": {"required_ids": ["c"], "allowed_ids": ["c"]},
                "numbers": {"required": ["1000 TL"], "allowed": ["1000 TL"]},
                "should_reject": True,
            },
            "legacy": {
                "standalone_query": "Urun B tutari nedir?",
                "inherited_context": {"product_types": ["Urun B"]},
                "cleared_fields": [],
                "needs_clarification": False,
                "retrieved_ids": ["noise", "c"],
                "cited_ids": ["noise"],
                "answer_numbers": ["900 TL"],
                "status": "verified",
            },
            "v2": {
                "standalone_query": "Urun B tutari nedir?",
                "inherited_context": {"product_types": ["Urun B"]},
                "cleared_fields": ["banks"],
                "needs_clarification": True,
                "retrieved_ids": ["c"],
                "cited_ids": ["c"],
                "answer_numbers": ["1000 TL"],
                "status": "rejected",
            },
        },
    ]


class PairedComparisonTest(unittest.TestCase):
    def test_all_requested_metrics_use_the_same_paired_records(self) -> None:
        report = compare_records(_paired_metric_records())
        self.assertEqual(report["legacy"]["standalone_query_accuracy"]["value"], 0.5)
        self.assertEqual(report["v2"]["standalone_query_accuracy"]["value"], 1.0)
        self.assertEqual(report["legacy"]["clarification_accuracy"]["value"], 0.5)
        self.assertEqual(report["v2"]["clarification_accuracy"]["value"], 1.0)
        self.assertEqual(report["legacy"]["recall_at_1"]["value"], 0.0)
        self.assertEqual(report["v2"]["recall_at_1"]["value"], 0.75)
        self.assertEqual(report["legacy"]["recall_at_3"]["value"], 1.0)
        self.assertEqual(report["v2"]["recall_at_3"]["value"], 1.0)
        self.assertEqual(report["legacy"]["mrr_at_10"]["value"], 0.5)
        self.assertEqual(report["v2"]["mrr_at_10"]["value"], 1.0)
        self.assertEqual(report["legacy"]["citation_accuracy"]["value"], 0.5)
        self.assertEqual(report["v2"]["citation_accuracy"]["value"], 1.0)
        self.assertEqual(report["legacy"]["numeric_accuracy"]["value"], 0.5)
        self.assertEqual(report["v2"]["numeric_accuracy"]["value"], 1.0)
        self.assertEqual(report["legacy"]["unsupported_rejection_rate"]["value"], 0.0)
        self.assertEqual(report["v2"]["unsupported_rejection_rate"]["value"], 1.0)
        self.assertEqual(report["legacy"]["session_isolation_pass_rate"]["value"], 1.0)
        self.assertEqual(report["v2"]["session_isolation_pass_rate"]["value"], 1.0)
        self.assertEqual(report["delta_v2_minus_legacy"]["recall_at_1"]["value"], 0.75)
        self.assertEqual(report["summary"]["status"], "measured")

    def test_topic_clear_and_inheritance_metrics(self) -> None:
        report = compare_records(_paired_metric_records())
        self.assertEqual(report["legacy"]["inherited_bank_accuracy"]["value"], 1.0)
        self.assertEqual(report["v2"]["inherited_product_accuracy"]["value"], 1.0)
        self.assertEqual(report["legacy"]["topic_change_clear_accuracy"]["value"], 0.5)
        self.assertEqual(report["v2"]["topic_change_clear_accuracy"]["value"], 1.0)
        self.assertEqual(report["legacy"]["inherited_date_accuracy"]["value"], 1.0)
        self.assertEqual(report["v2"]["inherited_date_accuracy"]["value"], 1.0)

    def test_missing_provider_or_labels_are_explicitly_unavailable(self) -> None:
        records = [
            {
                "record_id": "missing-v2",
                "labels": {"relevant_ids": ["a"]},
                "legacy": {"retrieved_ids": ["a"]},
                "v2": None,
            }
        ]
        report = compare_records(records)
        self.assertEqual(report["legacy"]["recall_at_1"]["status"], "unavailable")
        self.assertEqual(report["v2"]["recall_at_1"]["status"], "unavailable")
        self.assertIn("paired legacy and v2", report["legacy"]["recall_at_1"]["reason"])
        self.assertEqual(report["legacy"]["citation_accuracy"]["status"], "unavailable")
        self.assertEqual(report["summary"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
