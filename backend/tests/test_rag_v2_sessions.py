from __future__ import annotations

import hashlib
import json
import secrets
import unittest
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_v2.api_router import router as rag_v2_router
from rag_v2.database import RagDatabasePool
from rag_v2.models import (
    QueryRoute,
    RagV2ChatRequest,
    RagV2ChatResponse,
    SearchRecord,
    SessionState,
    StructuredFact,
)
from rag_v2.routing import QueryRouter
from rag_v2.retrieval import HybridRetriever, build_lexical_tsquery
from rag_v2.service import RagV2Components, RagV2Service
from rag_v2.sessions import (
    SessionAccessDenied,
    SessionConflict,
    SessionData,
    SessionExpired,
    SessionNotFound,
    SessionStore,
    inherited_context_for_route,
)
from rag_v2.settings import RagV2Settings


ROOT = Path(__file__).resolve().parents[1]


def base_settings(**overrides) -> RagV2Settings:
    values = {
        "db_host": "127.0.0.1",
        "db_port": 5432,
        "db_name": "test",
        "db_user": "test",
        "db_password": "test",
        "route_with_llm": False,
        "max_evidence": 4,
    }
    values.update(overrides)
    return RagV2Settings(**values)


def accepted_record() -> SearchRecord:
    return SearchRecord(
        chunk_id="a" * 64,
        offer_id="b" * 64,
        document_id="current:1",
        bank_key="ziraat_katilim",
        bank_name="Ziraat Katilim",
        primary_product="KONUT_FINANSMANI",
        product_types=["KONUT_FINANSMANI"],
        page_title="Konut Finansmani",
        section_heading="Teklif kosullari",
        source_url="https://example.test/konut",
        scope="current",
        content=(
            "Ziraat Katilim konut finansmani azami tutari 1000000 TL ve "
            "vade suresi 120 aydir."
        ),
        classification_confidence=0.91,
        classification_status="accepted",
        rrf_score=0.02,
    )


class _MemorySessions:
    def __init__(self) -> None:
        self.session: SessionData | None = None
        self.messages: list[str] = []
        self.conversation: list[dict] = []
        self.verified_evidence_refs: list[dict] = []

    def create(self, _owner_token=None) -> SessionData:
        now = datetime.now(timezone.utc)
        self.session = SessionData(
            token=secrets.token_urlsafe(32),
            internal_id=uuid.uuid4(),
            created_at=now,
            expires_at=now + timedelta(hours=1),
            state=SessionState(),
            version=1,
        )
        return self.session

    def get(self, token, _owner_token=None) -> SessionData:
        if self.session is None or token != self.session.token:
            raise AssertionError("unexpected session token")
        return self.session

    def recent_user_messages(self, _session_id) -> list[str]:
        return list(self.messages[-6:])

    def recent_conversation_messages(self, _session_id) -> list[dict]:
        return [dict(item) for item in self.conversation[-12:]]

    def recent_verified_evidence_refs(
        self,
        _session_id,
        limit,
    ) -> list[dict]:
        return list(reversed(self.verified_evidence_refs))[:limit]

    def add_user_message(self, _session_id, content):
        self.messages.append(content)
        return uuid.uuid4(), len(self.messages)

    def finish_turn(
        self,
        session,
        _turn_id,
        _answer,
        _status,
        _route,
        _evidence,
        new_state,
        *,
        user_content=None,
    ) -> int:
        if user_content is not None:
            self.messages.append(user_content)
            self.conversation.append(
                {"role": "user", "content": user_content, "status": None}
            )
        self.conversation.append(
            {
                "role": "assistant",
                "content": _answer,
                "status": _status,
                "route": _route.model_dump(mode="json"),
            }
        )
        if _status == "verified":
            for item in _evidence:
                if f"[{item.source_id}]" not in _answer:
                    continue
                self.verified_evidence_refs.append(
                    {
                        "turn_id": _turn_id,
                        "source_id": item.source_id,
                        "chunk_id": item.chunk_id,
                        "offer_id": item.offer_id,
                        "document_id": item.document_id,
                    }
                )
        session.state = new_state
        session.version += 1
        return session.version


class _CountingRetriever:
    def __init__(self) -> None:
        self.routes: list[QueryRoute] = []

    def retrieve(self, route, _top_k, *, use_reranker=False):
        self.routes.append(route.model_copy(deep=True))
        return [accepted_record()], {"reranker_used": False}, []


class _CarryoverRetriever:
    def __init__(self) -> None:
        self.retrieve_calls = 0
        self.hydrate_references: list[list[dict]] = []

    def retrieve(self, route, _top_k, *, use_reranker=False):
        self.retrieve_calls += 1
        records = [accepted_record()] if self.retrieve_calls == 1 else []
        issues = [] if records else ["no_matching_evidence"]
        return records, {"reranker_used": False}, issues

    def hydrate_context_records(self, _route, references):
        self.hydrate_references.append(list(references))
        return [accepted_record()] if references else []


class _MixedConfidenceRetriever:
    def retrieve(self, _route, _top_k, *, use_reranker=False):
        required = accepted_record().model_copy(
            update={
                "chunk_id": "c" * 64,
                "offer_id": "d" * 64,
                "document_id": "current:2",
                "classification_confidence": 0.40,
                "classification_status": "required",
                "classification_conflict": True,
            }
        )
        return [required, accepted_record()], {"reranker_used": False}, []


class _ComparisonRetriever:
    def retrieve(self, route, _top_k, *, use_reranker=False):
        records: list[SearchRecord] = []
        for suffix, value in (("low", 200), ("mid", 1250), ("high", 5000)):
            fact = StructuredFact(
                fact_type="ALISVERIS_PUANI",
                fact_text=f"{value} TL Worldpuan",
                normalized_value={
                    "value": value,
                    "unit": "currency",
                    "currency": "TRY",
                },
                evidence_text=f"Toplamda {value} TL Worldpuan kazanin.",
                confidence=0.99,
            )
            records.append(
                accepted_record().model_copy(
                    update={
                        "chunk_id": suffix * 16,
                        "offer_id": f"offer-{suffix}",
                        "document_id": f"current-{suffix}",
                        "primary_product": "ALISVERIS_PUANI",
                        "product_types": ["ALISVERIS_PUANI"],
                        "page_title": f"{value} TL Worldpuan Kampanyasi",
                        "content": f"Kampanya toplamda {value} TL Worldpuan kazandirir.",
                        "facts": [fact],
                    }
                )
            )
        if route.offer_ids:
            records = [
                item for item in records if item.offer_id in route.offer_ids
            ]
        return records, {"reranker_used": False}, []


class _SingleLowConfidenceNumericRetriever:
    def retrieve(self, _route, _top_k, *, use_reranker=False):
        amount = StructuredFact(
            fact_type="FINANSMAN_TUTARI",
            fact_text="6.000.000 TL",
            normalized_value={
                "value": 6000000,
                "unit": "currency",
                "currency": "TRY",
            },
            evidence_text="Azami finansman tutari 6.000.000 TL'dir.",
            confidence=0.55,
        )
        source = accepted_record().model_copy(
            update={
                "facts": [amount],
                "content": "Azami finansman tutari 6.000.000 TL'dir.",
                "classification_confidence": 0.40,
                "classification_status": "required",
                "classification_conflict": True,
            }
        )
        return [source], {"reranker_used": False}, []


class _UnexpectedAnswerEvren:
    def chat(self, _messages):
        raise AssertionError("EVREN must not generate a single-offer numeric answer")


class _BankCoverageRetriever:
    BANKS = (
        ("adil_katilim", "Adil Katilim"),
        ("albaraka", "Albaraka Turk"),
        ("dunya_katilim", "Dunya Katilim"),
        ("hayat_finans", "Hayat Finans"),
        ("kuveyt_turk", "Kuveyt Turk"),
        ("emlak_katilim", "Emlak Katilim"),
        ("turkiye_finans", "Turkiye Finans"),
        ("vakif_katilim", "Vakif Katilim"),
    )

    def retrieve(self, _route, _top_k, *, use_reranker=False):
        records = []
        for index, (bank_key, bank_name) in enumerate(self.BANKS, start=1):
            value = 24 + index * 12
            fact = StructuredFact(
                fact_type="VADE_SURESI",
                fact_text=f"{value} ay",
                normalized_value={"value": value, "unit": "month"},
                evidence_text=f"Azami vade {value} aydir.",
                confidence=0.99,
            )
            records.append(
                accepted_record().model_copy(
                    update={
                        "chunk_id": f"{index:064x}",
                        "offer_id": f"{index + 100:064x}",
                        "document_id": f"current:{index + 100}",
                        "bank_key": bank_key,
                        "bank_name": bank_name,
                        "page_title": "Konut Finansmani",
                        "content": (
                            f"{bank_name} konut finansmani azami vadesi "
                            f"{value} aydir."
                        ),
                        "facts": [fact],
                    }
                )
            )
        return records, {"reranker_used": False}, []


class _GroundedEvren:
    def chat(self, messages):
        payload = messages[-1]["content"]
        if '"maturity"' in payload:
            return "Ziraat Katilim konut finansmani vade suresi 120 aydir. [S1]"
        return "Ziraat Katilim konut finansmani azami tutari 1000000 TL'dir. [S1]"


class _InsufficientEvren:
    def chat(self, _messages):
        return "Yeterli dogrulanabilir kaynak bulunamadi."


class _ContextEvren:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages):
        self.calls.append(messages)
        payload = json.loads(messages[-1]["content"])
        if payload.get("task") == "Reply to the current conversational message.":
            history = payload["conversation_history_untrusted_data"]
            prior_answers = [
                item["content"]
                for item in history
                if item.get("role") == "assistant"
            ]
            if prior_answers:
                return f"Önceki yanıtımı hatırlıyorum: {prior_answers[-1]}"
            return "Merhaba, ben HititFinLex asistanıyım."
        if '"maturity"' in messages[-1]["content"]:
            return "Ziraat Katilim konut finansmani vade suresi 120 aydir. [S1]"
        return "Ziraat Katilim konut finansmani azami tutari 1000000 TL'dir. [S1]"


class ContextHydrationTest(unittest.TestCase):
    def test_context_hydration_rejects_product_and_identity_mismatch(self):
        retriever = HybridRetriever(None, base_settings(), None, None)
        reference = {
            "chunk_id": accepted_record().chunk_id,
            "offer_id": accepted_record().offer_id,
            "document_id": accepted_record().document_id,
        }
        with patch.object(
            retriever,
            "_hydrate_dense",
            return_value=[accepted_record()],
        ):
            product_mismatch = retriever.hydrate_context_records(
                QueryRoute(
                    standalone_query="tasit vadesi",
                    product_types=["TASIT_FINANSMANI"],
                ),
                [reference],
            )
            identity_mismatch = retriever.hydrate_context_records(
                QueryRoute(
                    standalone_query="konut vadesi",
                    product_types=["KONUT_FINANSMANI"],
                ),
                [{**reference, "offer_id": "x" * 64}],
            )

        self.assertEqual(product_mismatch, [])
        self.assertEqual(identity_mismatch, [])


class ServiceTurnIntegrationTest(unittest.TestCase):
    def test_single_bank_numeric_lookup_uses_low_confidence_fact_directly(self):
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=_MemorySessions(),
                router=QueryRouter(),
                retriever=_SingleLowConfidenceNumericRetriever(),
                evren=_UnexpectedAnswerEvren(),
                qdrant=None,
            )
        )

        response = service.chat(
            RagV2ChatRequest(
                query=(
                    "Ziraat Katilim konut finansmaninda en yuksek "
                    "tutar nedir?"
                )
            )
        )

        self.assertEqual(response.status, "verified")
        self.assertIn("6.000.000 TL", response.answer)
        self.assertIn("[S1]", response.answer)
        self.assertEqual(response.evidence[0].classification_status, "required")
        self.assertFalse(response.diagnostics["classification_policy_enforced"])

    def test_bank_wide_list_uses_extended_evidence_limit_and_one_line_per_bank(self):
        sessions = _MemorySessions()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(
                    max_evidence=2,
                    max_bank_evidence=12,
                ),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=_BankCoverageRetriever(),
                evren=None,
                qdrant=None,
            )
        )

        response = service.chat(
            RagV2ChatRequest(query="Konut finansmani vadelerini getir.")
        )

        self.assertEqual(response.status, "verified")
        self.assertEqual(len(response.evidence), 8)
        self.assertEqual(len(response.answer.splitlines()), 8)
        self.assertEqual(len(response.diagnostics["covered_banks"]), 8)
        self.assertEqual(len(response.diagnostics["missing_evidence_banks"]), 2)
        self.assertTrue(response.diagnostics["bank_coverage_mode"])
        self.assertIsNotNone(sessions.session)
        self.assertEqual(sessions.session.state.active_offer_ids, [])
        self.assertTrue(sessions.session.state.broad_bank_context)
        self.assertEqual(len(sessions.session.state.ranked_offers), 8)

    def test_evren_receives_prior_assistant_answer_as_untrusted_context(self):
        sessions = _MemorySessions()
        retriever = _CountingRetriever()
        evren = _ContextEvren()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=retriever,
                evren=evren,
                qdrant=None,
            )
        )

        first = service.chat(RagV2ChatRequest(query="Merhaba, sen kimsin?"))
        second = service.chat(
            RagV2ChatRequest(
                session_id=first.session_id,
                query="Bunu biraz daha samimi anlat.",
            )
        )

        self.assertEqual(first.status, "conversational")
        self.assertEqual(second.status, "conversational")
        self.assertEqual(len(retriever.routes), 0)
        second_payload = json.loads(evren.calls[-1][-1]["content"])
        prior_answers = [
            item["content"]
            for item in second_payload["conversation_history_untrusted_data"]
            if item.get("role") == "assistant"
        ]
        self.assertEqual(prior_answers, [first.answer])
        self.assertTrue(second.diagnostics["assistant_context_used"])
        self.assertIsNotNone(sessions.session)
        self.assertEqual(sessions.session.state.conversation_turn_count, 2)
        self.assertIn(first.answer, sessions.session.state.conversation_summary)

    def test_verified_turn_preserves_existing_conversation_memory(self):
        sessions = _MemorySessions()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=_CountingRetriever(),
                evren=_ContextEvren(),
                qdrant=None,
            )
        )

        first = service.chat(
            RagV2ChatRequest(query="Merhaba, bana Kirmizi Lale diye hitap et.")
        )
        second = service.chat(
            RagV2ChatRequest(
                session_id=first.session_id,
                query="Ziraat Katilim konut finansmani tutari nedir?",
            )
        )

        self.assertEqual(first.status, "conversational")
        self.assertEqual(second.status, "verified")
        self.assertIsNotNone(sessions.session)
        state = sessions.session.state
        self.assertEqual(state.conversation_turn_count, 2)
        self.assertIn(first.answer, state.conversation_summary)
        self.assertIn(second.answer, state.conversation_summary)

    def test_natural_financial_follow_up_retrieves_with_inherited_state(self):
        sessions = _MemorySessions()
        retriever = _CountingRetriever()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=retriever,
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )

        first = service.chat(
            RagV2ChatRequest(
                query="Ziraat Katilim konut finansmani tutari nedir?"
            )
        )
        follow = service.chat(
            RagV2ChatRequest(
                session_id=first.session_id,
                query="Devam et ve daha ayrintili anlat.",
            )
        )

        self.assertEqual(first.status, "verified")
        self.assertEqual(follow.status, "verified")
        self.assertEqual(follow.route.intent, "lookup")
        self.assertEqual(follow.route.banks, ["Ziraat Katilim"])
        self.assertEqual(follow.route.product_types, ["KONUT_FINANSMANI"])
        self.assertEqual(follow.route.field_types, ["amount"])
        self.assertTrue(follow.diagnostics["retrieval_performed"])
        self.assertEqual(len(retriever.routes), 2)

    def test_financial_model_route_prevents_early_chat_and_inherits_state(self):
        state = SessionState(
            active_banks=["Ziraat Katilim"],
            active_products=["KONUT_FINANSMANI"],
            active_scope="current",
            last_field_types=["maturity"],
            last_standalone_query=(
                "Ziraat Katilim guncel konut finansmani vade suresi nedir?"
            ),
        )
        model_route = {
            "standalone_query": (
                "Ziraat Katilim guncel konut finansmani vade detaylari nedir?"
            ),
            "intent": "lookup",
        }

        route = QueryRouter().resolve(
            "Baska hangi detaylari bilmeliyim?",
            state,
            model_route,
        )

        self.assertEqual(route.intent, "lookup")
        self.assertEqual(route.banks, ["Ziraat Katilim"])
        self.assertEqual(route.product_types, ["KONUT_FINANSMANI"])
        self.assertEqual(route.field_types, ["maturity"])
        self.assertIn("banks", route.inherited_fields)
        self.assertIn("product_types", route.inherited_fields)

        changed_bank = QueryRouter().resolve(
            "Vakif Katilim hakkinda bilgi ver.",
            state,
            model_route,
        )
        self.assertEqual(changed_bank.banks, ["Vakif Katilim"])
        self.assertEqual(changed_bank.product_types, [])
        self.assertNotIn("banks", changed_bank.inherited_fields)

    def test_casual_turn_preserves_verified_financial_context(self):
        sessions = _MemorySessions()
        retriever = _CountingRetriever()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=retriever,
                evren=_ContextEvren(),
                qdrant=None,
            )
        )

        first = service.chat(
            RagV2ChatRequest(
                query="Ziraat Katilim konut finansmani tutari nedir?"
            )
        )
        casual = service.chat(
            RagV2ChatRequest(
                session_id=first.session_id,
                query="Tesekkur ederim, nasilsin?",
            )
        )
        follow = service.chat(
            RagV2ChatRequest(
                session_id=first.session_id,
                query="Peki vadesi ne kadar?",
            )
        )

        self.assertEqual(first.status, "verified")
        self.assertEqual(casual.status, "conversational")
        self.assertEqual(follow.status, "verified")
        self.assertEqual(follow.route.banks, ["Ziraat Katilim"])
        self.assertEqual(follow.route.product_types, ["KONUT_FINANSMANI"])
        self.assertEqual(len(retriever.routes), 2)

    def test_numeric_answer_order_matches_session_offer_order(self):
        sessions = _MemorySessions()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=_ComparisonRetriever(),
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )

        response = service.chat(
            RagV2ChatRequest(
                query="Aktif kampanyalardaki alisveris puanlarini karsilastir"
            )
        )

        self.assertEqual(response.status, "verified")
        self.assertTrue(response.answer.splitlines()[0].endswith("[S1]."))
        self.assertEqual(
            [item.offer_id for item in response.evidence],
            ["offer-high", "offer-mid", "offer-low"],
        )
        self.assertIsNotNone(sessions.session)
        self.assertEqual(
            [item.offer_id for item in sessions.session.state.ranked_offers],
            ["offer-high", "offer-mid", "offer-low"],
        )
        follow = service.chat(
            RagV2ChatRequest(
                session_id=response.session_id,
                query="Ikincisi kac puan?",
            )
        )
        self.assertEqual(follow.status, "verified")
        self.assertEqual(follow.route.offer_ids, ["offer-mid"])
        self.assertIn("1250 TL Worldpuan", follow.answer)

    def test_every_follow_up_turn_retrieves_again_from_structured_state(self):
        sessions = _MemorySessions()
        retriever = _CountingRetriever()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=retriever,
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )

        first = service.chat(
            RagV2ChatRequest(
                query=(
                    "Ziraat Katilim konut finansmaninda en yuksek tutar nedir?"
                )
            )
        )
        second = service.chat(
            RagV2ChatRequest(
                session_id=first.session_id,
                query="Peki vadesi ne kadar?",
            )
        )

        self.assertEqual(first.status, "verified")
        self.assertEqual(second.status, "verified")
        self.assertEqual(len(retriever.routes), 2)
        self.assertEqual(second.route.banks, ["Ziraat Katilim"])
        self.assertEqual(second.route.product_types, ["KONUT_FINANSMANI"])
        self.assertIn("banks", second.inherited_context)
        self.assertIn("product_types", second.inherited_context)
        self.assertIn("vade suresi", second.standalone_query)

    def test_low_confidence_conflict_result_is_usable_by_default(self):
        sessions = _MemorySessions()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(max_evidence=1),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=_MixedConfidenceRetriever(),
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )
        response = service.chat(
            RagV2ChatRequest(
                query="Ziraat Katilim konut finansmani tutari nedir?"
            )
        )
        self.assertEqual(response.status, "verified")
        self.assertEqual(len(response.evidence), 1)
        self.assertEqual(response.evidence[0].classification_status, "required")
        self.assertEqual(response.evidence[0].source_id, "S1")
        self.assertFalse(response.diagnostics["classification_policy_enforced"])
        self.assertEqual(
            response.diagnostics["low_confidence_candidates_withheld"], 0
        )
        self.assertEqual(
            response.diagnostics["classification_conflict_candidates_withheld"],
            0,
        )

    def test_opt_in_classification_policy_withholds_low_confidence_result(self):
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(
                    max_evidence=1,
                    enforce_classification_policy=True,
                ),
                pool=None,
                sessions=_MemorySessions(),
                router=QueryRouter(),
                retriever=_MixedConfidenceRetriever(),
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )
        response = service.chat(
            RagV2ChatRequest(
                query="Ziraat Katilim konut finansmani tutari nedir?"
            )
        )
        self.assertEqual(response.status, "verified")
        self.assertEqual(response.evidence[0].classification_status, "accepted")
        self.assertTrue(response.diagnostics["classification_policy_enforced"])
        self.assertEqual(
            response.diagnostics["low_confidence_candidates_withheld"], 1
        )
        self.assertEqual(
            response.diagnostics["classification_conflict_candidates_withheld"],
            1,
        )

    def test_model_insufficient_answer_is_a_safe_refusal(self):
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=_MemorySessions(),
                router=QueryRouter(),
                retriever=_CountingRetriever(),
                evren=_InsufficientEvren(),
                qdrant=None,
            )
        )
        response = service.chat(
            RagV2ChatRequest(
                query="Ziraat Katilim konut finansmani tutari nedir?"
            )
        )
        self.assertEqual(response.status, "insufficient_evidence")
        self.assertEqual(
            response.answer,
            "Yeterli doğrulanabilir kaynak bulunamadı.",
        )
        self.assertEqual(len(response.evidence), 1)
        self.assertIn("model_reported_insufficient_evidence", response.issues)
        self.assertNotIn("answer_validation_failed", response.issues)

    def test_verified_source_package_is_reused_for_a_follow_up(self):
        sessions = _MemorySessions()
        retriever = _CarryoverRetriever()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=retriever,
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )

        first = service.chat(
            RagV2ChatRequest(
                query="Ziraat Katilim konut finansmani tutari nedir?"
            )
        )
        follow = service.chat(
            RagV2ChatRequest(
                session_id=first.session_id,
                query="Peki vadesi ne kadar?",
            )
        )

        self.assertEqual(first.status, "verified")
        self.assertEqual(retriever.retrieve_calls, 2)
        self.assertTrue(retriever.hydrate_references[-1])
        self.assertEqual(follow.status, "verified")
        self.assertEqual(follow.evidence[0].source_id, "S1")
        self.assertEqual(follow.diagnostics["fresh_evidence_records"], 0)
        self.assertEqual(follow.diagnostics["session_evidence_selected"], 1)
        self.assertIn("session_evidence_reused", follow.issues)
        self.assertNotIn("no_matching_evidence", follow.issues)

    def test_insufficient_turn_preserves_last_verified_financial_state(self):
        sessions = _MemorySessions()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=_CountingRetriever(),
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )
        first = service.chat(
            RagV2ChatRequest(
                query="Ziraat Katilim konut finansmani tutari nedir?"
            )
        )
        self.assertEqual(first.status, "verified")
        self.assertIsNotNone(sessions.session)
        before = sessions.session.state.model_copy(deep=True)
        service.evren = _InsufficientEvren()

        failed = service.chat(
            RagV2ChatRequest(
                session_id=first.session_id,
                query=(
                    "Ziraat Katilim konut finansmani kosullari nelerdir?"
                ),
            )
        )

        self.assertEqual(failed.status, "insufficient_evidence")
        after = sessions.session.state
        self.assertEqual(after.active_banks, before.active_banks)
        self.assertEqual(after.active_products, before.active_products)
        self.assertEqual(after.active_offer_ids, before.active_offer_ids)
        self.assertEqual(after.ranked_offers, before.ranked_offers)
        self.assertGreater(
            after.conversation_turn_count,
            before.conversation_turn_count,
        )

    def test_first_insufficient_turn_still_preserves_query_context(self):
        sessions = _MemorySessions()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=_CountingRetriever(),
                evren=_InsufficientEvren(),
                qdrant=None,
            )
        )

        first = service.chat(
            RagV2ChatRequest(
                query="Vakif Katilim tasit finansmani kosullari nelerdir?"
            )
        )
        follow = service.chat(
            RagV2ChatRequest(
                session_id=first.session_id,
                query="Peki vadesi ne kadar?",
            )
        )

        self.assertEqual(first.status, "insufficient_evidence")
        self.assertEqual(follow.route.banks, ["Vakif Katilim"])
        self.assertEqual(follow.route.product_types, ["TASIT_FINANSMANI"])
        self.assertIn("banks", follow.route.inherited_fields)
        self.assertIn("product_types", follow.route.inherited_fields)

    def test_unsupported_calculation_fails_closed_without_model_math(self):
        sessions = _MemorySessions()
        retriever = _CountingRetriever()
        evren = _GroundedEvren()
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=retriever,
                evren=evren,
                qdrant=None,
            )
        )
        response = service.chat(
            RagV2ChatRequest(
                query=(
                    "Ziraat Katilim konut finansmani tutar farkini hesapla."
                )
            )
        )
        self.assertEqual(response.status, "insufficient_evidence")
        self.assertIn("deterministic_calculation_unavailable", response.issues)
        self.assertEqual(response.answer, "Yeterli doğrulanabilir kaynak bulunamadı.")

    def test_request_constraints_replace_inherited_period_without_conflict(self):
        sessions = _MemorySessions()
        session = sessions.create()
        session.state = SessionState(
            active_banks=["Ziraat Katilim"],
            active_products=["KONUT_FINANSMANI"],
            active_scope="historical",
            active_year=2025,
            active_date_from=datetime(2025, 1, 1).date(),
            active_date_to=datetime(2025, 12, 31).date(),
            active_offer_ids=["old-offer"],
            last_standalone_query="initial",
        )
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=sessions,
                router=QueryRouter(),
                retriever=_CountingRetriever(),
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )
        route = service._route(
            RagV2ChatRequest(query="Peki vadesi?", scope="current"),
            session,
            [],
            [],
        )
        self.assertEqual(route.scope, "current")
        self.assertIsNone(route.year)
        self.assertIsNone(route.date_from)
        self.assertIsNone(route.date_to)
        self.assertEqual(route.offer_ids, [])

    def test_invalid_request_date_order_is_rejected(self):
        with self.assertRaises(ValidationError):
            RagV2ChatRequest(
                query="kampanyalar",
                date_from=datetime(2026, 2, 1).date(),
                date_to=datetime(2026, 1, 1).date(),
            )

    def test_review_product_label_is_available_in_default_session_state(self):
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(),
                pool=None,
                sessions=_MemorySessions(),
                router=QueryRouter(),
                retriever=_CountingRetriever(),
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )
        review = accepted_record().model_copy(
            update={
                "classification_confidence": 0.72,
                "classification_status": "review",
            }
        )
        state = service._next_state(
            QueryRoute(standalone_query="kosullar"),
            [review],
            [],
        )
        self.assertEqual(state.active_products, ["KONUT_FINANSMANI"])
        self.assertEqual(
            state.ranked_offers[0].product_types,
            ["KONUT_FINANSMANI"],
        )

    def test_opt_in_policy_keeps_review_product_out_of_session_state(self):
        service = RagV2Service(
            RagV2Components(
                settings=base_settings(enforce_classification_policy=True),
                pool=None,
                sessions=_MemorySessions(),
                router=QueryRouter(),
                retriever=_CountingRetriever(),
                evren=_GroundedEvren(),
                qdrant=None,
            )
        )
        review = accepted_record().model_copy(
            update={
                "classification_confidence": 0.72,
                "classification_status": "review",
            }
        )
        state = service._next_state(
            QueryRoute(standalone_query="kosullar"),
            [review],
            [],
        )
        self.assertEqual(state.active_products, [])
        self.assertEqual(state.ranked_offers[0].product_types, [])

    def test_inherited_date_range_is_persisted_as_structured_context(self):
        route = QueryRoute(
            standalone_query="same period",
            scope="historical",
            date_from=datetime(2025, 1, 1).date(),
            date_to=datetime(2025, 6, 30).date(),
            inherited_fields=["scope", "date_range"],
        )
        inherited = inherited_context_for_route(route)
        self.assertEqual(
            inherited["date_range"],
            {"from": "2025-01-01", "to": "2025-06-30"},
        )


class _MissingSchemaPool:
    instance = None

    def __init__(self, _settings) -> None:
        self.opened = False
        self.closed = False
        self.relations = None
        type(self).instance = self

    def open(self) -> None:
        self.opened = True

    def require_relations(self, relations) -> None:
        self.relations = relations
        raise RuntimeError("unit-test-database-password")

    def close(self) -> None:
        self.closed = True


class ServiceReadinessTest(unittest.TestCase):
    def test_missing_schema_fails_closed_and_closes_pool_without_secret(self):
        with patch("rag_v2.service.RagDatabasePool", _MissingSchemaPool):
            with self.assertRaises(RuntimeError) as raised:
                RagV2Service.create(base_settings())
        pool = _MissingSchemaPool.instance
        self.assertTrue(pool.opened)
        self.assertTrue(pool.closed)
        self.assertIn("rag_chunks", pool.relations)
        self.assertNotIn(
            "unit-test-database-password",
            str(raised.exception),
        )

    def test_unready_service_returns_503_instead_of_endpoint_500(self):
        app = FastAPI()
        app.state.rag_v2_service = None
        app.include_router(rag_v2_router)
        response = TestClient(app).post(
            "/rag/v2/chat",
            json={"query": "konut finansmani"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "RAG V2 is not ready."})


class PostgresLexicalExpressionTest(unittest.TestCase):
    def test_generated_or_prefix_query_hits_real_postgresql_tsvector(self):
        load_dotenv(ROOT / ".env", override=False)
        pool = None
        try:
            configured = replace(
                RagV2Settings.from_env(),
                db_pool_min_size=1,
                db_pool_max_size=1,
                route_with_llm=False,
            )
            pool = RagDatabasePool(configured)
            pool.open()
            query = build_lexical_tsquery(
                "guncel konut finansmani tutari vade suresi nedir"
            )
            with pool.connection() as connection:
                row = connection.execute(
                    """
                    SELECT to_tsvector(
                        'simple',
                        %s
                    ) @@ to_tsquery('simple', %s) AS matched
                    """,
                    (
                        "Konut finansmani azami tutar ve vade kosullari",
                        query,
                    ),
                ).fetchone()
            self.assertTrue(row["matched"])
        except Exception as error:
            if isinstance(error, AssertionError):
                raise
            raise unittest.SkipTest(
                "PostgreSQL lexical expression unavailable: "
                + type(error).__name__
            ) from error
        finally:
            if pool is not None:
                pool.close()


class _ApiSessions:
    def __init__(self, session: SessionData) -> None:
        self.session = session
        self.calls: list[tuple[str, str, str | None]] = []
        self.get_error: Exception | None = None

    def get(self, token, owner):
        self.calls.append(("get", token, owner))
        if self.get_error:
            raise self.get_error
        return self.session

    def clear(self, token, owner):
        self.calls.append(("clear", token, owner))
        if self.get_error:
            raise self.get_error
        return self.session

    def revoke(self, token, owner):
        self.calls.append(("revoke", token, owner))
        if self.get_error:
            raise self.get_error

    def conversation_transcript(self, session):
        self.calls.append(("transcript", session.token, None))
        return [
            RagV2ChatResponse(
                session_id=session.token,
                query="Merhaba",
                standalone_query="Merhaba",
                answer="Merhaba!",
                status="conversational",
                route=QueryRoute(
                    standalone_query="Merhaba",
                    intent="chat",
                ),
            )
        ]


class SessionApiTest(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.session = SessionData(
            token=secrets.token_urlsafe(32),
            internal_id=uuid.uuid4(),
            created_at=now,
            expires_at=now + timedelta(hours=1),
            state=SessionState(),
            version=1,
        )
        self.sessions = _ApiSessions(self.session)
        app = FastAPI()
        app.include_router(rag_v2_router)
        app.state.rag_v2_service = SimpleNamespace(sessions=self.sessions)
        self.client = TestClient(app)
        self.owner = "owner-token-with-enough-entropy"
        self.headers = {
            "X-RAG-Session-Id": self.session.token,
            "X-RAG-Client-Id": self.owner,
        }

    def test_header_session_endpoints_do_not_put_secret_in_url(self):
        response = self.client.post(
            "/rag/v2/session/clear", headers=self.headers, json={}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.sessions.calls[-1],
            ("clear", self.session.token, self.owner),
        )
        self.assertNotIn(self.session.token, str(response.request.url))

        response = self.client.delete("/rag/v2/session", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.sessions.calls[-1],
            ("revoke", self.session.token, self.owner),
        )
        self.assertNotIn(self.session.token, str(response.request.url))

    def test_owner_bound_transcript_uses_header_session(self):
        response = self.client.get(
            "/rag/v2/session/messages",
            headers=self.headers,
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["query"], "Merhaba")
        self.assertEqual(
            self.sessions.calls[-2:],
            [
                ("get", self.session.token, self.owner),
                ("transcript", self.session.token, None),
            ],
        )
        self.assertNotIn(self.session.token, str(response.request.url))

    def test_expired_and_denied_sessions_are_not_disclosed(self):
        self.sessions.get_error = SessionExpired("secret detail")
        expired = self.client.get("/rag/v2/session", headers=self.headers)
        self.assertEqual(expired.status_code, 410)
        self.assertNotIn(self.session.token, expired.text)

        self.sessions.get_error = SessionNotFound("secret detail")
        missing = self.client.get("/rag/v2/session", headers=self.headers)
        self.assertEqual(missing.status_code, 404)
        self.assertNotIn("secret detail", missing.text)

    def test_concurrent_session_conflict_maps_to_409(self):
        service = SimpleNamespace(
            sessions=self.sessions,
            chat=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                SessionConflict("internal version")
            ),
        )
        self.client.app.state.rag_v2_service = service
        response = self.client.post(
            "/rag/v2/chat",
            headers={"X-RAG-Client-Id": self.owner},
            json={"query": "konut finansmani", "session_id": self.session.token},
        )
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("internal version", response.text)


class PostgresSessionIntegrationTest(unittest.TestCase):
    pool: RagDatabasePool
    store: SessionStore
    created_ids: list[uuid.UUID]

    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv(ROOT / ".env", override=False)
        try:
            settings = RagV2Settings.from_env()
            settings = replace(
                settings,
                db_pool_min_size=1,
                db_pool_max_size=2,
                route_with_llm=False,
            )
            cls.pool = RagDatabasePool(settings)
            cls.pool.open()
            with cls.pool.connection() as connection:
                migrated = connection.execute(
                    "SELECT to_regclass('public.rag_sessions') IS NOT NULL"
                ).fetchone()
            if not migrated or not next(iter(migrated.values())):
                cls.pool.close()
                raise unittest.SkipTest("RAG V2 migration 0003 is not applied")
            cls.store = SessionStore(cls.pool, settings)
            cls.created_ids = []
        except unittest.SkipTest:
            raise
        except Exception as error:
            raise unittest.SkipTest(
                f"PostgreSQL RAG session integration unavailable: {type(error).__name__}"
            ) from error

    @classmethod
    def tearDownClass(cls) -> None:
        if not hasattr(cls, "pool"):
            return
        try:
            if getattr(cls, "created_ids", None):
                with cls.pool.connection() as connection:
                    connection.execute(
                        "DELETE FROM rag_sessions WHERE id = ANY(%s)",
                        (cls.created_ids,),
                    )
                    connection.commit()
        finally:
            cls.pool.close()

    def _create(self, owner: str) -> SessionData:
        session = self.store.create(owner)
        self.created_ids.append(session.internal_id)
        return session

    def test_two_sessions_are_owner_bound_and_do_not_leak_state(self):
        first_owner = secrets.token_urlsafe(24)
        second_owner = secrets.token_urlsafe(24)
        first = self._create(first_owner)
        second = self._create(second_owner)
        route = QueryRoute(
            standalone_query="Ziraat Katilim guncel konut finansmani nedir?",
            banks=["Ziraat Katilim"],
            product_types=["KONUT_FINANSMANI"],
        )
        state = SessionState(
            active_banks=["Ziraat Katilim"],
            active_products=["KONUT_FINANSMANI"],
            last_standalone_query=route.standalone_query,
        )
        turn_id, _ = self.store.add_user_message(first.internal_id, "ilk soru")
        self.store.finish_turn(
            first,
            turn_id,
            "Yeterli dogrulanabilir kaynak bulunamadi.",
            "insufficient_evidence",
            route,
            [],
            state,
        )

        loaded_first = self.store.get(first.token, first_owner)
        loaded_second = self.store.get(second.token, second_owner)
        self.assertEqual(loaded_first.state.active_banks, ["Ziraat Katilim"])
        self.assertEqual(loaded_second.state.active_banks, [])
        self.assertEqual(
            self.store.recent_user_messages(first.internal_id), ["ilk soru"]
        )
        self.assertEqual(self.store.recent_user_messages(second.internal_id), [])
        conversation = self.store.recent_conversation_messages(first.internal_id)
        self.assertEqual(
            [item["role"] for item in conversation],
            ["user", "assistant"],
        )
        self.assertEqual(conversation[0]["content"], "ilk soru")
        self.assertIn("Yeterli", conversation[1]["content"])
        self.assertEqual(
            self.store.recent_conversation_messages(second.internal_id),
            [],
        )
        transcript = self.store.conversation_transcript(loaded_first)
        self.assertEqual(len(transcript), 1)
        self.assertEqual(transcript[0].query, "ilk soru")
        self.assertEqual(transcript[0].status, "insufficient_evidence")
        self.assertEqual(
            self.store.conversation_transcript(loaded_second),
            [],
        )
        with self.assertRaises(SessionAccessDenied):
            self.store.get(first.token, second_owner)

        with self.pool.connection() as connection:
            stored_hash = connection.execute(
                "SELECT token_hash FROM rag_sessions WHERE id = %s",
                (first.internal_id,),
            ).fetchone()["token_hash"]
        self.assertNotEqual(stored_hash, first.token)
        self.assertEqual(
            stored_hash.strip(),
            hashlib.sha256(first.token.encode("utf-8")).hexdigest(),
        )

    def test_expired_session_is_rejected(self):
        owner = secrets.token_urlsafe(24)
        session = self._create(owner)
        now = datetime.now(timezone.utc)
        with self.pool.connection() as connection:
            connection.execute(
                """
                UPDATE rag_sessions
                SET created_at = %s, expires_at = %s
                WHERE id = %s
                """,
                (
                    now - timedelta(hours=2),
                    now - timedelta(minutes=1),
                    session.internal_id,
                ),
            )
            connection.commit()
        with self.assertRaises(SessionExpired):
            self.store.get(session.token, owner)

    def test_clear_removes_messages_and_structured_state(self):
        owner = secrets.token_urlsafe(24)
        session = self._create(owner)
        turn_id, _ = self.store.add_user_message(session.internal_id, "test soru")
        self.store.finish_turn(
            session,
            turn_id,
            "Netlestirir misiniz?",
            "needs_clarification",
            QueryRoute(
                standalone_query="test soru",
                needs_clarification=True,
                clarification_question="Netlestirir misiniz?",
            ),
            [],
            SessionState(active_banks=["Vakif Katilim"]),
        )
        cleared = self.store.clear(session.token, owner)
        self.assertEqual(cleared.state, SessionState())
        self.assertEqual(self.store.recent_user_messages(session.internal_id), [])
        self.assertEqual(
            self.store.recent_conversation_messages(session.internal_id),
            [],
        )


if __name__ == "__main__":
    unittest.main()
