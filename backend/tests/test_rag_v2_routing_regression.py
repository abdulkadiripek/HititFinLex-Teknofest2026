from __future__ import annotations

import unittest

from rag_v2.identity import normalize_text
from rag_v2.models import SessionState
from rag_v2.routing import QueryRouter


class RoutingSemanticRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = QueryRouter()

    def test_explicit_semantic_qualifiers_survive_standalone_rewrite(self) -> None:
        cases = (
            (
                "Ziraat Katilim is yeri finansmani ekspertiz ucreti nedir?",
                ("is yeri", "ekspertiz"),
            ),
            (
                "Vakif Katilim konut finansmani 60 ay vadesi nedir?",
                ("60 ay",),
            ),
            (
                "Ziraat Katilim Exim kredisi vadesi nedir?",
                ("exim",),
            ),
        )
        for query, expected_terms in cases:
            with self.subTest(query=query):
                route = self.router.resolve(query, SessionState())
                standalone = normalize_text(route.standalone_query)
                for term in expected_terms:
                    self.assertIn(term, standalone)

    def test_model_query_cannot_drop_explicit_semantic_qualifier(self) -> None:
        route = self.router.resolve(
            "Ziraat Katilim Exim kredisi nedir?",
            SessionState(),
            {
                "standalone_query": "Ziraat Katilim guncel kredi nedir?",
                "intent": "lookup",
            },
        )
        self.assertIn("exim", normalize_text(route.standalone_query))

    def test_model_cannot_promote_maturity_lookup_to_calculate(self) -> None:
        route = self.router.resolve(
            "Vakif Katilim konut finansmani kac ay vadeli?",
            SessionState(),
            {
                "standalone_query": (
                    "Vakif Katilim guncel konut finansmani vade suresi nedir?"
                ),
                "intent": "calculate",
            },
        )
        self.assertEqual(route.intent, "lookup")
        self.assertEqual(route.field_types, ["maturity"])

    def test_model_cannot_promote_follow_up_maturity_lookup(self) -> None:
        state = SessionState(
            active_banks=["Ziraat Katilim"],
            active_products=["KONUT_FINANSMANI"],
            last_standalone_query="initial",
        )
        route = self.router.resolve(
            "Peki vadesi ne kadar?",
            state,
            {
                "standalone_query": (
                    "Ziraat Katilim guncel konut finansmani vade suresi nedir?"
                ),
                "intent": "calculate",
            },
        )
        self.assertEqual(route.intent, "lookup")
        self.assertEqual(route.banks, ["Ziraat Katilim"])
        self.assertEqual(route.product_types, ["KONUT_FINANSMANI"])
        self.assertEqual(route.field_types, ["maturity"])

    def test_untrusted_prior_answer_number_is_not_a_qualifier(self) -> None:
        state = SessionState(
            active_banks=["Vakif Katilim"],
            active_products=["KONUT_FINANSMANI"],
            active_offer_ids=["verified-offer"],
            last_field_types=["maturity"],
            last_standalone_query="initial",
        )
        route = self.router.resolve(
            "Onceki cevabinda 120 ay demistin; bu dogru mu?",
            state,
        )
        self.assertNotIn("120", route.standalone_query)
        self.assertEqual(route.offer_ids, ["verified-offer"])

    def test_bare_getiri_term_is_not_an_arithmetic_request(self) -> None:
        route = self.router.resolve(
            "Albaraka Turk katilma hesabi getiri orani nedir?",
            SessionState(),
            {
                "standalone_query": (
                    "Albaraka Turk guncel katilma hesabi getiri orani nedir?"
                ),
                "intent": "calculate",
            },
        )
        self.assertEqual(route.intent, "lookup")
        self.assertEqual(route.field_types, ["rate"])

    def test_explicit_arithmetic_request_remains_calculate(self) -> None:
        route = self.router.resolve(
            (
                "Ziraat Katilim konut finansmani icin 1000000 TL ile "
                "1200000 TL arasindaki farki hesapla."
            ),
            SessionState(),
        )
        self.assertEqual(route.intent, "calculate")


if __name__ == "__main__":
    unittest.main()
