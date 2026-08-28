from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.retrieval_compare import (
    DEFAULT_CASES,
    DEFAULT_CORPUS,
    DATASET_LABEL,
    CorpusManifest,
    CorpusRecord,
    RetrievalAssetError,
    RetrieverUnavailable,
    compare_with_backends,
    load_corpus_manifest,
    load_retrieval_cases,
    run_live_comparison,
    validate_cases_against_corpus,
)


class FakeBackend:
    def __init__(self, name: str, mode: str, fail_at: str | None = None) -> None:
        self.name = name
        self.mode = mode
        self.fail_at = fail_at
        self.calls: list[tuple[str, str, int]] = []
        self.closed = False

    def retrieve(self, case, top_k: int) -> list[str]:
        self.calls.append((case.case_id, case.query, top_k))
        if case.case_id == self.fail_at:
            raise RetrieverUnavailable(f"{self.name}_fixture_unavailable")
        relevant = case.relevant_record_keys[0]
        if self.mode == "rank_one":
            return [relevant, relevant, "later-noise"]
        return ["noise", relevant]

    def close(self) -> None:
        self.closed = True


class LeakyFailureBackend(FakeBackend):
    def retrieve(self, case, top_k: int) -> list[str]:
        raise RuntimeError("SENSITIVE_SENTINEL_DO_NOT_COPY")


class RetrievalDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases, cls.dataset_sha256 = load_retrieval_cases(DEFAULT_CASES)
        cls.corpus = load_corpus_manifest(DEFAULT_CORPUS)

    def test_fixed_dataset_has_at_least_thirty_silver_cases(self) -> None:
        self.assertGreaterEqual(len(self.cases), 30)
        self.assertEqual(len(self.cases), 32)
        self.assertEqual(self.corpus.record_count, 771)
        self.assertEqual(len({case.case_id for case in self.cases}), len(self.cases))
        self.assertEqual(len(self.dataset_sha256), 64)
        self.assertTrue(all(case.relevant_record_keys for case in self.cases))

    def test_every_source_label_matches_the_checked_in_corpus(self) -> None:
        validate_cases_against_corpus(self.cases, self.corpus)
        labeled_keys = {
            label.record_key
            for case in self.cases
            for label in case.source_labels
        }
        self.assertEqual(len(labeled_keys), 32)
        self.assertTrue(labeled_keys.issubset(self.corpus.records_by_key))

    def test_corpus_url_resolution_is_canonical_and_stable(self) -> None:
        record = CorpusRecord(
            record_key="stable-record",
            bank_key="bank",
            page_title="Title",
            source_url="https://Example.com/path/?utm_source=test&b=2&a=1",
        )
        manifest = CorpusManifest([record], "digest")
        self.assertEqual(
            manifest.resolve_url("https://example.com/path?a=1&b=2"),
            "stable-record",
        )
        self.assertTrue(manifest.resolve_url("https://example.com/other").startswith("unmapped:"))

    def test_non_silver_retrieval_asset_is_rejected(self) -> None:
        payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
        payload["cases"][0]["label"] = "gold"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RetrievalAssetError, "label must be exactly silver_unverified"):
                load_retrieval_cases(path)

    def test_changed_corpus_metadata_is_rejected(self) -> None:
        first = self.cases[0].source_labels[0]
        changed = CorpusManifest(
            [
                CorpusRecord(
                    record_key=first.record_key,
                    bank_key=first.bank_key,
                    page_title="Changed title",
                    source_url=first.source_url,
                )
            ],
            "digest",
        )
        with self.assertRaisesRegex(RetrievalAssetError, "corpus title label changed"):
            validate_cases_against_corpus([self.cases[0]], changed)


class SameSetRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases, cls.dataset_sha256 = load_retrieval_cases(DEFAULT_CASES)
        cls.corpus = load_corpus_manifest(DEFAULT_CORPUS)

    def test_both_backends_receive_identical_ordered_cases(self) -> None:
        legacy = FakeBackend("legacy", "rank_two")
        v2 = FakeBackend("v2", "rank_one")
        report = compare_with_backends(
            self.cases,
            self.corpus,
            legacy,
            v2,
            dataset_sha256=self.dataset_sha256,
        )
        expected_calls = [(case.case_id, case.query, 10) for case in self.cases]
        self.assertEqual(legacy.calls, expected_calls)
        self.assertEqual(v2.calls, expected_calls)
        self.assertTrue(legacy.closed)
        self.assertTrue(v2.closed)
        self.assertEqual(report["dataset_label"], DATASET_LABEL)
        self.assertEqual(report["case_count"], 32)
        self.assertEqual(report["status"], "measured")

    def test_fixed_formula_results_are_reported_for_the_paired_set(self) -> None:
        report = compare_with_backends(
            self.cases,
            self.corpus,
            FakeBackend("legacy", "rank_two"),
            FakeBackend("v2", "rank_one"),
            dataset_sha256=self.dataset_sha256,
        )
        legacy = report["metrics"]["legacy"]
        v2 = report["metrics"]["v2"]
        self.assertEqual(legacy["recall_at_1"]["value"], 0.0)
        self.assertEqual(v2["recall_at_1"]["value"], 1.0)
        self.assertEqual(legacy["recall_at_3"]["value"], 1.0)
        self.assertEqual(v2["recall_at_10"]["value"], 1.0)
        self.assertEqual(legacy["mrr_at_10"]["value"], 0.5)
        self.assertEqual(v2["mrr_at_10"]["value"], 1.0)
        self.assertEqual(v2["ndcg_at_10"]["value"], 1.0)
        self.assertEqual(
            report["metrics"]["delta_v2_minus_legacy"]["recall_at_1"]["value"],
            1.0,
        )

    def test_partial_provider_failure_makes_all_paired_metrics_unavailable(self) -> None:
        report = compare_with_backends(
            self.cases,
            self.corpus,
            FakeBackend("legacy", "rank_two"),
            FakeBackend("v2", "rank_one", fail_at=self.cases[1].case_id),
        )
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["providers"]["v2"]["reason"], "v2_fixture_unavailable")
        for name in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"):
            self.assertEqual(report["metrics"]["legacy"][name]["status"], "unavailable")
            self.assertEqual(report["metrics"]["v2"][name]["status"], "unavailable")

    def test_missing_corpus_is_honestly_unavailable_without_secret_values(self) -> None:
        report = run_live_comparison(
            DEFAULT_CASES,
            Path("definitely-missing-corpus.jsonl"),
            validate_only=True,
        )
        rendered = json.dumps(report)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["providers"]["legacy"]["reason"], "corpus_missing_or_label_mismatch")
        self.assertNotIn("DB_PASSWORD", rendered)
        self.assertNotIn("Authorization", rendered)
        self.assertNotIn("api-key", rendered)

    def test_top_k_below_required_cutoff_is_rejected(self) -> None:
        legacy = FakeBackend("legacy", "rank_two")
        v2 = FakeBackend("v2", "rank_one")
        with self.assertRaisesRegex(RetrievalAssetError, "top_k must be at least 10"):
            compare_with_backends(
                self.cases,
                self.corpus,
                legacy,
                v2,
                top_k=5,
            )
        self.assertTrue(legacy.closed)
        self.assertTrue(v2.closed)

    def test_unexpected_backend_error_text_is_never_copied_to_report(self) -> None:
        report = compare_with_backends(
            self.cases,
            self.corpus,
            LeakyFailureBackend("legacy", "rank_two"),
            FakeBackend("v2", "rank_one"),
        )
        rendered = json.dumps(report)
        self.assertEqual(report["providers"]["legacy"]["reason"], "backend_execution_failed")
        self.assertNotIn("SENSITIVE_SENTINEL_DO_NOT_COPY", rendered)


if __name__ == "__main__":
    unittest.main()
