from __future__ import annotations

import unittest

from rag_v2.models import QueryRoute, SearchRecord, StructuredFact
from rag_v2.retrieval import HybridRetriever, rank_relevant_records
from rag_v2.settings import RagV2Settings


def settings(**overrides) -> RagV2Settings:
    values = {
        "db_host": "127.0.0.1",
        "db_port": 5432,
        "db_name": "test",
        "db_user": "test",
        "db_password": "test",
    }
    values.update(overrides)
    return RagV2Settings(**values)


def amount_fact(value: float) -> StructuredFact:
    return StructuredFact(
        fact_type="FINANSMAN_TUTARI",
        fact_text=f"{value} TRY",
        normalized_value={"value": value, "currency": "TRY"},
        evidence_text=f"Amount {value} TRY",
        confidence=0.9,
    )


def fee_fact(value: float) -> StructuredFact:
    return StructuredFact(
        fact_type="DIGER_UCRET",
        fact_text=f"{value} TRY",
        normalized_value={"value": value, "currency": "TRY"},
        evidence_text=f"Fee {value} TRY",
        confidence=0.9,
    )


def record(
    chunk_id: str,
    offer_id: str,
    document_id: str,
    bank_key: str,
    product: str,
    *,
    facts: list[StructuredFact] | None = None,
    rrf_score: float = 0.0,
    confidence: float = 0.9,
    status: str = "accepted",
    conflict: bool = False,
) -> SearchRecord:
    return SearchRecord(
        chunk_id=chunk_id,
        offer_id=offer_id,
        document_id=document_id,
        bank_key=bank_key,
        bank_name=bank_key,
        primary_product=product,
        product_types=[product],
        scope="current",
        content="Evidence content",
        facts=facts or [],
        classification_confidence=confidence,
        classification_status=status,
        classification_conflict=conflict,
        rrf_score=rrf_score,
    )


class StaticHybridRetriever(HybridRetriever):
    def __init__(
        self,
        dense: list[SearchRecord],
        lexical=None,
        *,
        enforce_classification_policy: bool = False,
    ) -> None:
        super().__init__(
            None,
            settings(
                enforce_classification_policy=enforce_classification_policy
            ),
            None,
            None,
        )
        self.dense = dense
        self.lexical = lexical or []

    def _dense(self, *_args, **_kwargs) -> list[SearchRecord]:
        return self.dense

    def _lexical(self, *_args, **_kwargs) -> list[SearchRecord]:
        return self.lexical


class RelevanceRankingTest(unittest.TestCase):
    def test_numeric_field_match_precedes_product_only_soft_match(self):
        rows = [
            record(
                "product-only",
                "offer-product",
                "doc-product",
                "bank-a",
                "KART_KAMPANYASI",
                rrf_score=0.9,
            ),
            record(
                "field-only",
                "offer-field",
                "doc-field",
                "bank-b",
                "YENI_MUSTERI",
                facts=[amount_fact(5000)],
                rrf_score=0.01,
            ),
        ]
        route = QueryRoute(
            standalone_query="kampanya tutarlarini karsilastir",
            intent="compare",
            product_types=["KART_KAMPANYASI"],
            field_types=["amount"],
        )

        ranked = rank_relevant_records(rows, route)

        self.assertEqual(ranked[0].chunk_id, "field-only")

    def test_longest_maturity_uses_maximum_fact_within_offer(self):
        def maturity(value: float) -> StructuredFact:
            return StructuredFact(
                fact_type="VADE_SURESI",
                fact_text=f"{value} ay",
                normalized_value={"value": value, "unit": "month"},
                evidence_text=f"Azami vade {value} aydir.",
                confidence=0.99,
            )

        rows = [
            record(
                "multi",
                "offer-multi",
                "doc-multi",
                "bank-a",
                "KONUT_FINANSMANI",
                facts=[maturity(84), maturity(120)],
            ),
            record(
                "single",
                "offer-single",
                "doc-single",
                "bank-b",
                "KONUT_FINANSMANI",
                facts=[maturity(100)],
            ),
        ]
        route = QueryRoute(
            standalone_query="en uzun konut finansmani vadesi",
            intent="compare",
            product_types=["KONUT_FINANSMANI"],
            field_types=["maturity"],
        )

        ranked = rank_relevant_records(rows, route)

        self.assertEqual(ranked[0].chunk_id, "multi")

    def test_day_maturity_is_normalized_before_month_ranking(self):
        day_fact = StructuredFact(
            fact_type="VADE_SURESI",
            fact_text="540 gun",
            normalized_value={"value": 540, "unit": "day"},
            evidence_text="Vade 540 gundur.",
            confidence=0.99,
        )
        month_fact = StructuredFact(
            fact_type="VADE_SURESI",
            fact_text="120 ay",
            normalized_value={"value": 120, "unit": "month"},
            evidence_text="Vade 120 aydir.",
            confidence=0.99,
        )
        rows = [
            record(
                "days",
                "offer-days",
                "doc-days",
                "bank-a",
                "KONUT_FINANSMANI",
                facts=[day_fact],
            ),
            record(
                "months",
                "offer-months",
                "doc-months",
                "bank-b",
                "KONUT_FINANSMANI",
                facts=[month_fact],
            ),
        ]
        route = QueryRoute(
            standalone_query="en uzun konut finansmani vadesi",
            intent="compare",
            product_types=["KONUT_FINANSMANI"],
            field_types=["maturity"],
        )

        ranked = rank_relevant_records(rows, route)

        self.assertEqual(ranked[0].chunk_id, "months")

    def test_wrong_product_numeric_value_cannot_outrank_joint_matches(self):
        rows = [
            record(
                "wrong",
                "offer-wrong",
                "doc-wrong",
                "bank-c",
                "TASIT_FINANSMANI",
                facts=[amount_fact(9_000_000)],
                rrf_score=0.5,
            ),
            record(
                "matched-low",
                "offer-low",
                "doc-low",
                "bank-a",
                "KONUT_FINANSMANI",
                facts=[amount_fact(500_000)],
                rrf_score=0.02,
            ),
            record(
                "matched-high",
                "offer-high",
                "doc-high",
                "bank-b",
                "KONUT_FINANSMANI",
                facts=[amount_fact(1_000_000)],
                rrf_score=0.01,
            ),
        ]
        route = QueryRoute(
            standalone_query="en yuksek konut finansmani tutari",
            intent="compare",
            product_types=["KONUT_FINANSMANI"],
            field_types=["amount"],
        )

        ranked = rank_relevant_records(rows, route)

        self.assertEqual(
            [item.chunk_id for item in ranked],
            ["matched-high", "matched-low", "wrong"],
        )

    def test_requested_field_coverage_precedes_rrf_within_product(self):
        rows = [
            record(
                "missing-field",
                "offer-missing",
                "doc-missing",
                "bank-a",
                "KONUT_FINANSMANI",
                facts=[fee_fact(10)],
                rrf_score=0.9,
            ),
            record(
                "field-match",
                "offer-match",
                "doc-match",
                "bank-b",
                "KONUT_FINANSMANI",
                facts=[amount_fact(100)],
                rrf_score=0.01,
            ),
        ]
        route = QueryRoute(
            standalone_query="konut finansmani tutari",
            intent="lookup",
            product_types=["KONUT_FINANSMANI"],
            field_types=["amount"],
        )

        ranked = rank_relevant_records(rows, route)

        self.assertEqual(ranked[0].chunk_id, "field-match")

    def test_product_preference_keeps_nonmatching_fallback(self):
        rows = [
            record(
                "wrong",
                "offer-wrong",
                "doc-wrong",
                "bank-a",
                "TASIT_FINANSMANI",
                rrf_score=0.9,
            ),
            record(
                "matched",
                "offer-match",
                "doc-match",
                "bank-b",
                "KONUT_FINANSMANI",
                rrf_score=0.01,
            ),
        ]
        route = QueryRoute(
            standalone_query="konut finansmani",
            product_types=["KONUT_FINANSMANI"],
        )

        ranked = rank_relevant_records(rows, route)

        self.assertEqual([item.chunk_id for item in ranked], ["matched", "wrong"])

    def test_numeric_tie_uses_rrf_before_stable_identifiers(self):
        rows = [
            record(
                "lower-rrf",
                "offer-a",
                "doc-a",
                "bank-a",
                "KONUT_FINANSMANI",
                facts=[amount_fact(100)],
                rrf_score=0.01,
            ),
            record(
                "higher-rrf",
                "offer-z",
                "doc-z",
                "bank-z",
                "KONUT_FINANSMANI",
                facts=[amount_fact(100)],
                rrf_score=0.03,
            ),
        ]
        route = QueryRoute(
            standalone_query="konut finansmani tutarlarini sirala",
            intent="compare",
            product_types=["KONUT_FINANSMANI"],
            field_types=["amount"],
        )

        ranked = rank_relevant_records(rows, route)

        self.assertEqual(ranked[0].chunk_id, "higher-rrf")

    def test_more_requested_fields_precede_partial_coverage(self):
        rows = [
            record(
                "partial",
                "offer-partial",
                "doc-partial",
                "bank-a",
                "KONUT_FINANSMANI",
                facts=[amount_fact(100)],
                rrf_score=0.9,
            ),
            record(
                "complete",
                "offer-complete",
                "doc-complete",
                "bank-b",
                "KONUT_FINANSMANI",
                facts=[amount_fact(100), fee_fact(5)],
                rrf_score=0.01,
            ),
        ]
        route = QueryRoute(
            standalone_query="konut finansmani tutar ve ucret bilgisi",
            intent="lookup",
            product_types=["KONUT_FINANSMANI"],
            field_types=["amount", "fee"],
        )

        ranked = rank_relevant_records(rows, route)

        self.assertEqual(ranked[0].chunk_id, "complete")


class RelevanceRetrievalIntegrationTest(unittest.TestCase):
    def test_default_policy_ranks_all_statuses_and_conflicts_equally(self):
        required_conflict = record(
            "a" * 64,
            "offer-required",
            "doc-required",
            "bank-a",
            "KONUT_FINANSMANI",
            facts=[amount_fact(1000)],
            confidence=0.2,
            status="required",
            conflict=True,
        )
        accepted = record(
            "b" * 64,
            "offer-accepted",
            "doc-accepted",
            "bank-b",
            "KONUT_FINANSMANI",
            facts=[amount_fact(1000)],
        )
        route = QueryRoute(
            standalone_query="konut finansmani tutari",
            product_types=["KONUT_FINANSMANI"],
            field_types=["amount"],
        )
        retriever = StaticHybridRetriever([required_conflict, accepted])

        selected, diagnostics, issues = retriever.retrieve(route, 1)

        self.assertEqual(issues, [])
        self.assertEqual(selected[0].offer_id, "offer-required")
        self.assertEqual(diagnostics["fallback_candidates"], 0)
        self.assertFalse(diagnostics["classification_policy_enforced"])
        self.assertIsNone(diagnostics["min_confidence"])

    def test_opt_in_policy_preserves_status_and_conflict_gates(self):
        required_conflict = record(
            "a" * 64,
            "offer-required",
            "doc-required",
            "bank-a",
            "KONUT_FINANSMANI",
            facts=[amount_fact(1000)],
            confidence=0.2,
            status="required",
            conflict=True,
        )
        accepted = record(
            "b" * 64,
            "offer-accepted",
            "doc-accepted",
            "bank-b",
            "KONUT_FINANSMANI",
            facts=[amount_fact(1000)],
        )
        route = QueryRoute(
            standalone_query="konut finansmani tutari",
            product_types=["KONUT_FINANSMANI"],
            field_types=["amount"],
        )
        retriever = StaticHybridRetriever(
            [required_conflict, accepted],
            enforce_classification_policy=True,
        )

        selected, diagnostics, issues = retriever.retrieve(route, 1)

        self.assertEqual(issues, [])
        self.assertEqual(selected[0].offer_id, "offer-accepted")
        self.assertEqual(diagnostics["fallback_candidates"], 1)
        self.assertTrue(diagnostics["classification_policy_enforced"])
        self.assertEqual(diagnostics["min_confidence"], 0.65)

    def test_relevance_precedes_dedupe_and_preserves_bank_diversity(self):
        rows = [
            record(
                "a" * 64,
                "offer-shared",
                "doc-wrong",
                "bank-a",
                "TASIT_FINANSMANI",
                facts=[amount_fact(9_000_000)],
            ),
            record(
                "b" * 64,
                "offer-shared",
                "doc-right",
                "bank-a",
                "KONUT_FINANSMANI",
                facts=[amount_fact(1_000)],
            ),
            record(
                "c" * 64,
                "offer-a-two",
                "doc-a-two",
                "bank-a",
                "KONUT_FINANSMANI",
                facts=[amount_fact(800)],
            ),
            record(
                "d" * 64,
                "offer-b",
                "doc-b",
                "bank-b",
                "KONUT_FINANSMANI",
                facts=[amount_fact(500)],
            ),
            record(
                "e" * 64,
                "offer-c",
                "doc-c",
                "bank-c",
                "TASIT_FINANSMANI",
                facts=[amount_fact(8_000_000)],
            ),
        ]
        route = QueryRoute(
            standalone_query="en yuksek konut finansmani tutarlari",
            intent="compare",
            product_types=["KONUT_FINANSMANI"],
            field_types=["amount"],
        )
        retriever = StaticHybridRetriever(rows)

        selected, diagnostics, issues = retriever.retrieve(route, 4)

        self.assertEqual(issues, [])
        self.assertEqual(
            [item.chunk_id for item in selected],
            ["b" * 64, "d" * 64, "c" * 64, "e" * 64],
        )
        self.assertEqual([item.bank_key for item in selected[:2]], ["bank-a", "bank-b"])
        self.assertNotIn("a" * 64, [item.chunk_id for item in selected])
        self.assertEqual(diagnostics["returned_product_matches"], 3)
        self.assertEqual(diagnostics["returned_field_matches"], 4)


if __name__ == "__main__":
    unittest.main()
