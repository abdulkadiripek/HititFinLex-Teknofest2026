from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from psycopg import Error as DatabaseError
from psycopg_pool import PoolTimeout

from .database import RagDatabasePool
from .evidence import (
    NUMERIC_ANSWER_FIELDS,
    build_answer_messages,
    deterministic_order,
    deterministic_numeric_answer,
    select_evidence_records,
    to_evidence,
)
from .models import (
    OfferReference,
    RagV2ChatRequest,
    RagV2ChatResponse,
    SessionState,
)
from .providers import (
    EvrenClient,
    ProviderProtocolError,
    ProviderUnavailable,
    QdrantRestClient,
    build_conversation_messages,
)
from .retrieval import HybridRetriever
from .identity import normalize_text
from .routing import (
    BANK_KEYS,
    QueryRouter,
    build_standalone_query,
    has_explicit_product,
)
from .sessions import SessionData, SessionStore, inherited_context_for_route
from .settings import RagV2Settings
from .validation import CITATION_PATTERN, validate_answer


FAIL_CLOSED_ANSWER = "Yeterli doğrulanabilir kaynak bulunamadı."
REQUIRED_RELATIONS = (
    "rag_chunks",
    "rag_messages",
    "rag_session_state",
    "rag_sessions",
    "rag_turn_evidence",
)


def _is_fail_closed_answer(value: str) -> bool:
    normalized_value = normalize_text(value).rstrip(" .")
    normalized_expected = normalize_text(FAIL_CLOSED_ANSWER).rstrip(" .")
    return normalized_value == normalized_expected


@dataclass(slots=True)
class RagV2Components:
    settings: RagV2Settings
    pool: RagDatabasePool
    sessions: SessionStore
    router: QueryRouter
    retriever: HybridRetriever
    evren: EvrenClient | None
    qdrant: QdrantRestClient | None


class RagV2Service:
    def __init__(self, components: RagV2Components) -> None:
        self.settings = components.settings
        self.pool = components.pool
        self.sessions = components.sessions
        self.router = components.router
        self.retriever = components.retriever
        self.evren = components.evren
        self.qdrant = components.qdrant

    @classmethod
    def create(cls, settings: RagV2Settings) -> "RagV2Service":
        pool = RagDatabasePool(settings)
        evren: EvrenClient | None = None
        qdrant: QdrantRestClient | None = None
        try:
            pool.open()
            pool.require_relations(REQUIRED_RELATIONS)
            evren = EvrenClient(settings) if settings.evren_ready else None
            qdrant = (
                QdrantRestClient(settings) if settings.qdrant_ready else None
            )
            components = RagV2Components(
                settings=settings,
                pool=pool,
                sessions=SessionStore(pool, settings),
                router=QueryRouter(),
                retriever=HybridRetriever(pool, settings, evren, qdrant),
                evren=evren,
                qdrant=qdrant,
            )
            return cls(components)
        except Exception:
            if evren is not None:
                evren.close()
            if qdrant is not None:
                qdrant.close()
            try:
                pool.close()
            except Exception:
                pass
            raise RuntimeError("RAG V2 initialization failed") from None

    def close(self) -> None:
        if self.evren is not None:
            self.evren.close()
        if self.qdrant is not None:
            self.qdrant.close()
        self.pool.close()

    def _route(
        self,
        payload: RagV2ChatRequest,
        session: SessionData,
        recent_messages: list[dict[str, Any]],
        issues: list[str],
    ):
        model_route: dict[str, Any] | None = None
        if self.settings.route_with_llm and self.evren is not None:
            try:
                model_route = self.evren.route_query(
                    payload.query,
                    session.state,
                    recent_messages,
                )
            except (ProviderUnavailable, ProviderProtocolError, ValueError):
                issues.append("route_model_fallback_used")
        route = self.router.resolve(payload.query, session.state, model_route)
        query_text = normalize_text(payload.query)
        constraints_changed = False
        if payload.product_types and not has_explicit_product(payload.query):
            requested_products = list(payload.product_types)
            if route.product_types != requested_products:
                route.product_types = requested_products
                route.offer_ids = []
                route.inherited_fields = [
                    item
                    for item in route.inherited_fields
                    if item not in {"product_types", "offer_ids"}
                ]
                constraints_changed = True
        explicit_scope = any(
            marker in query_text
            for marker in (
                "guncel",
                "bugun",
                "simdi",
                "gecmis",
                "arsiv",
                "tarihsel",
                "tum donem",
                "guncel ve gecmis",
            )
        )
        explicit_year = bool(
            re.search(r"(?<!\d)(?:19|20)\d{2}(?!\d)", query_text)
        )
        if payload.scope is not None and not explicit_scope and not explicit_year:
            if route.scope != payload.scope:
                route.scope = payload.scope
                route.year = None
                route.date_from = None
                route.date_to = None
                route.offer_ids = []
                route.inherited_fields = [
                    item
                    for item in route.inherited_fields
                    if item not in {"scope", "year", "date_range", "offer_ids"}
                ]
                constraints_changed = True
        query_has_date = explicit_year
        supplied_date_filter = (
            payload.date_from is not None or payload.date_to is not None
        )
        if not query_has_date and supplied_date_filter:
            if (
                route.date_from != payload.date_from
                or route.date_to != payload.date_to
            ):
                route.date_from = payload.date_from
                route.date_to = payload.date_to
                route.year = None
                route.offer_ids = []
                route.inherited_fields = [
                    item
                    for item in route.inherited_fields
                    if item not in {"year", "date_range", "offer_ids"}
                ]
                constraints_changed = True
        if constraints_changed:
            route.offer_ids = []
            route.inherited_fields = [
                item
                for item in route.inherited_fields
                if item not in {"offer_ids", "date_range"}
            ]
            route.standalone_query = build_standalone_query(payload.query, route)
        return route

    def _inherited_context(self, route) -> dict[str, Any]:
        return inherited_context_for_route(route)

    def _recent_messages(self, session_id: uuid.UUID) -> list[dict[str, Any]]:
        loader = getattr(self.sessions, "recent_conversation_messages", None)
        if callable(loader):
            return loader(session_id)
        return [
            {"role": "user", "content": content, "status": None}
            for content in self.sessions.recent_user_messages(session_id)
        ]

    def _recent_verified_evidence_refs(
        self,
        session_id: uuid.UUID,
        limit: int,
    ) -> list[dict[str, Any]]:
        loader = getattr(self.sessions, "recent_verified_evidence_refs", None)
        if not callable(loader):
            return []
        return loader(session_id, limit)

    @staticmethod
    def _merge_session_records(records, session_records):
        merged = list(records)
        chunk_ids = {item.chunk_id for item in merged}
        offer_counts: dict[str, int] = {}
        document_counts: dict[str, int] = {}
        for item in merged:
            offer_counts[item.offer_id] = offer_counts.get(item.offer_id, 0) + 1
            document_counts[item.document_id] = (
                document_counts.get(item.document_id, 0) + 1
            )
        added = []
        for item in session_records:
            if item.chunk_id in chunk_ids:
                continue
            if offer_counts.get(item.offer_id, 0) >= 2:
                continue
            if document_counts.get(item.document_id, 0) >= 2:
                continue
            merged.append(item)
            added.append(item)
            chunk_ids.add(item.chunk_id)
            offer_counts[item.offer_id] = offer_counts.get(item.offer_id, 0) + 1
            document_counts[item.document_id] = (
                document_counts.get(item.document_id, 0) + 1
            )
        return merged, added

    def _remember_turn(
        self,
        state: SessionState,
        query: str,
        answer: str,
        status: str,
        route,
    ) -> SessionState:
        next_state = state.model_copy(deep=True)
        next_state.conversation_turn_count += 1
        entry = json.dumps(
            {
                "turn": next_state.conversation_turn_count,
                "user": query[:2000],
                "assistant": answer[:5000],
                "status": status,
                "route": {
                    "intent": route.intent,
                    "banks": route.banks,
                    "product_types": route.product_types,
                    "field_types": route.field_types,
                    "scope": route.scope,
                    "year": route.year,
                },
                "authority": "conversation_only_not_financial_evidence",
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        combined = "\n".join(
            value
            for value in (next_state.conversation_summary, entry)
            if value
        )
        limit = self.settings.conversation_summary_max_chars
        if len(combined) > limit:
            lines = combined.splitlines()
            while len("\n".join(lines)) > limit and len(lines) > 1:
                lines.pop(0)
            combined = "\n".join(lines)
            if len(combined) > limit:
                combined = combined[-limit:]
        next_state.conversation_summary = combined
        return next_state

    def _next_state(
        self,
        route,
        records,
        evidence,
        *,
        context_records=None,
        context_evidence=None,
        base_state: SessionState | None = None,
    ) -> SessionState:
        context_records = records if context_records is None else context_records
        context_evidence = evidence if context_evidence is None else context_evidence
        display_by_key = {value: key for key, value in BANK_KEYS.items()}

        def context_eligible(record) -> bool:
            if not self.settings.enforce_classification_policy:
                return True
            return record.classification_status != "required" and (
                not record.classification_conflict
                or record.classification_status == "verified"
            )

        def record_products(record) -> list[str]:
            if not context_eligible(record):
                return []
            if (
                self.settings.enforce_classification_policy
                and record.classification_status not in {"accepted", "verified"}
            ):
                return []
            return record.product_types

        references = [
            OfferReference(
                offer_id=record.offer_id,
                bank=display_by_key.get(record.bank_key, record.bank_name),
                product_types=record_products(record),
                document_id=record.document_id,
                rank=index,
            )
            for index, record in enumerate(records, start=1)
            if context_eligible(record)
        ]
        context_banks = list(
            dict.fromkeys(
                display_by_key.get(record.bank_key, record.bank_name)
                for record in context_records
                if context_eligible(record)
            )
        )
        context_products = list(
            dict.fromkeys(
                product
                for record in context_records
                for product in record_products(record)
            )
        )
        broad_bank_context = (
            not route.banks
            and route.intent in {"lookup", "list", "compare"}
            and bool(route.product_types or route.field_types)
            and bool(context_banks)
        )
        next_state = (base_state or SessionState()).model_copy(deep=True)
        next_state.active_banks = route.banks or context_banks
        next_state.broad_bank_context = broad_bank_context
        next_state.active_products = route.product_types or context_products
        next_state.active_scope = route.scope
        next_state.active_year = route.year
        next_state.active_date_from = route.date_from
        next_state.active_date_to = route.date_to
        next_state.active_offer_ids = (
            []
            if broad_bank_context
            else list(dict.fromkeys(item.offer_id for item in context_evidence))
        )
        next_state.ranked_offers = references
        next_state.last_intent = route.intent
        next_state.last_field_types = route.field_types
        next_state.last_source_ids = [
            item.source_id for item in context_evidence
        ]
        next_state.last_document_ids = [
            item.document_id for item in context_evidence
        ]
        next_state.last_standalone_query = route.standalone_query
        return next_state

    def _state_after_unverified_route(
        self,
        state: SessionState,
        route,
    ) -> SessionState:
        next_state = state.model_copy(deep=True)
        bank_changed = bool(route.banks) and set(route.banks) != set(
            state.active_banks
        )
        product_changed = bool(route.product_types) and set(
            route.product_types
        ) != set(state.active_products)
        period_changed = (
            route.scope != state.active_scope
            or route.year != state.active_year
            or route.date_from != state.active_date_from
            or route.date_to != state.active_date_to
        )
        broad_context = (
            not route.banks
            and route.intent in {"lookup", "list", "compare", "calculate"}
            and bool(route.product_types or route.field_types)
        )

        if route.banks:
            next_state.active_banks = list(route.banks)
        elif product_changed or broad_context:
            next_state.active_banks = []
        if route.product_types:
            next_state.active_products = list(route.product_types)
        elif bank_changed:
            next_state.active_products = []
        next_state.broad_bank_context = broad_context
        next_state.active_scope = route.scope
        next_state.active_year = route.year
        next_state.active_date_from = route.date_from
        next_state.active_date_to = route.date_to

        if bank_changed or product_changed or period_changed or broad_context:
            next_state.active_offer_ids = []
            next_state.ranked_offers = []
            next_state.last_source_ids = []
            next_state.last_document_ids = []
        elif route.offer_ids:
            allowed = set(route.offer_ids)
            next_state.active_offer_ids = list(route.offer_ids)
            next_state.ranked_offers = [
                item for item in next_state.ranked_offers if item.offer_id in allowed
            ]

        next_state.last_intent = route.intent
        next_state.last_field_types = list(route.field_types)
        next_state.last_standalone_query = route.standalone_query
        return next_state

    def chat(
        self,
        payload: RagV2ChatRequest,
        owner_token: str | None = None,
    ) -> RagV2ChatResponse:
        started = time.perf_counter()
        session = (
            self.sessions.get(payload.session_id, owner_token)
            if payload.session_id
            else self.sessions.create(owner_token)
        )
        recent = self._recent_messages(session.internal_id)
        turn_id = uuid.uuid4()
        issues: list[str] = []
        route_started = time.perf_counter()
        route = self._route(payload, session, recent, issues)
        route_ms = (time.perf_counter() - route_started) * 1000.0
        inherited = self._inherited_context(route)

        if route.intent == "chat":
            generation_started = time.perf_counter()
            answer = (
                "Şu anda EVREN sohbet servisine ulaşamıyorum. "
                "Yine de katılım finansı sorularını ürün veya bilgi alanıyla "
                "yazarsanız kaynak kayıtlarında arayabilirim."
            )
            answer_strategy = "conversation_provider_fallback"
            if self.evren is None:
                issues.append("conversation_provider_unavailable")
            else:
                try:
                    candidate = self.evren.chat(
                        build_conversation_messages(
                            payload.query,
                            session.state,
                            recent,
                        )
                    )
                    candidate = CITATION_PATTERN.sub("", candidate)
                    candidate = re.sub(r"[ \t]+", " ", candidate).strip()
                    if candidate:
                        answer = candidate
                        answer_strategy = "evren_conversation"
                    else:
                        issues.append("conversation_provider_empty_answer")
                except (ProviderUnavailable, ProviderProtocolError, ValueError):
                    issues.append("conversation_provider_unavailable")
            generation_ms = (
                time.perf_counter() - generation_started
            ) * 1000.0
            new_state = self._remember_turn(
                session.state,
                payload.query,
                answer,
                "conversational",
                route,
            )
            self.sessions.finish_turn(
                session,
                turn_id,
                answer,
                "conversational",
                route,
                [],
                new_state,
                user_content=payload.query,
            )
            return RagV2ChatResponse(
                session_id=session.token,
                query=payload.query,
                standalone_query=route.standalone_query,
                answer=answer,
                status="conversational",
                inherited_context=inherited,
                route=route,
                evidence=[],
                issues=list(dict.fromkeys(issues)),
                diagnostics={
                    "route_ms": round(route_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "total_ms": round(
                        (time.perf_counter() - started) * 1000.0,
                        2,
                    ),
                    "retrieval_performed": False,
                    "answer_validation_passed": None,
                    "answer_strategy": answer_strategy,
                    "conversation_messages_used": len(recent),
                    "assistant_context_used": any(
                        item.get("role") == "assistant" for item in recent
                    ),
                    "conversation_turn_count": (
                        new_state.conversation_turn_count
                    ),
                },
            )

        if route.needs_clarification:
            answer = route.clarification_question or (
                "Sorunuzu banka veya ürün adıyla netleştirir misiniz?"
            )
            new_state = self._remember_turn(
                session.state,
                payload.query,
                answer,
                "needs_clarification",
                route,
            )
            self.sessions.finish_turn(
                session,
                turn_id,
                answer,
                "needs_clarification",
                route,
                [],
                new_state,
                user_content=payload.query,
            )
            return RagV2ChatResponse(
                session_id=session.token,
                query=payload.query,
                standalone_query=route.standalone_query,
                answer=answer,
                status="needs_clarification",
                inherited_context=inherited,
                route=route,
                evidence=[],
                issues=issues,
                diagnostics={
                    "route_ms": round(route_ms, 2),
                    "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "retrieval_performed": False,
                    "conversation_messages_used": len(recent),
                    "conversation_turn_count": (
                        new_state.conversation_turn_count
                    ),
                },
            )

        retrieval_started = time.perf_counter()
        fresh_records, retrieval_diagnostics, retrieval_issues = self.retriever.retrieve(
            route,
            payload.top_k,
            use_reranker=payload.use_reranker,
        )
        issues.extend(retrieval_issues)
        session_evidence_refs: list[dict[str, Any]] = []
        hydrated_session_records = []
        session_records = []
        hydrate_context = getattr(
            self.retriever,
            "hydrate_context_records",
            None,
        )
        if callable(hydrate_context):
            try:
                session_evidence_refs = self._recent_verified_evidence_refs(
                    session.internal_id,
                    max(payload.top_k * 4, self.settings.max_bank_evidence * 2),
                )
                hydrated_session_records = hydrate_context(
                    route,
                    session_evidence_refs,
                )
            except (DatabaseError, PoolTimeout):
                issues.append("session_evidence_unavailable")
        records, session_records = self._merge_session_records(
            fresh_records,
            hydrated_session_records,
        )
        if self.settings.enforce_classification_policy:
            eligible_records = [
                item
                for item in records
                if item.classification_status != "required"
                and (
                    not item.classification_conflict
                    or item.classification_status == "verified"
                )
            ]
        else:
            eligible_records = records
        bank_coverage_mode = (
            not route.banks
            and route.intent in {"lookup", "list", "compare"}
            and bool(route.product_types or route.field_types)
        )
        evidence_limit = (
            min(payload.top_k, self.settings.max_bank_evidence)
            if bank_coverage_mode
            else min(payload.top_k, self.settings.max_evidence)
        )
        selection_records = (
            deterministic_order(eligible_records, route)
            if session_records
            else eligible_records
        )
        answer_records = select_evidence_records(
            selection_records,
            route,
            evidence_limit,
            require_textual_product_confirmation=(
                self.settings.require_textual_product_confirmation
            ),
        )
        answer_records = deterministic_order(answer_records, route)
        evidence = to_evidence(answer_records, evidence_limit)
        session_chunk_ids = {item.chunk_id for item in session_records}
        selected_session_evidence = sum(
            item.chunk_id in session_chunk_ids for item in answer_records
        )
        if selected_session_evidence:
            issues = [item for item in issues if item != "no_matching_evidence"]
            issues.append("session_evidence_reused")
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

        status = "insufficient_evidence"
        answer = FAIL_CLOSED_ANSWER
        validation_issues: list[str] = []
        generation_ms = 0.0
        answer_strategy = "fail_closed"
        low_confidence_withheld = (
            sum(item.classification_status == "required" for item in records)
            if self.settings.enforce_classification_policy
            else 0
        )
        conflict_withheld = (
            sum(
                item.classification_conflict
                and item.classification_status != "verified"
                for item in records
            )
            if self.settings.enforce_classification_policy
            else 0
        )
        withheld_evidence = len(records) - len(eligible_records)
        if low_confidence_withheld:
            issues.append("low_confidence_evidence_withheld")
        if conflict_withheld:
            issues.append("classification_conflict_evidence_withheld")
        if route.intent == "calculate" and (evidence or eligible_records):
            issues.append("deterministic_calculation_unavailable")
        elif (
            evidence
            and route.intent == "lookup"
            and (
                bool(route.offer_ids)
                or not route.banks
                or len({item.offer_id for item in evidence}) == 1
            )
            and set(route.field_types).intersection(NUMERIC_ANSWER_FIELDS)
            and (candidate := deterministic_numeric_answer(route, evidence))
            is not None
        ):
            generation_started = time.perf_counter()
            validation = validate_answer(candidate, evidence)
            validation_issues = validation.issues
            if validation.valid:
                answer = candidate
                status = "verified"
                answer_strategy = "deterministic_numeric"
            else:
                status = "rejected"
                issues.append("answer_validation_failed")
            generation_ms = (time.perf_counter() - generation_started) * 1000.0
        elif evidence and route.intent in {"compare", "list"} and set(
            route.field_types
        ).intersection(NUMERIC_ANSWER_FIELDS):
            generation_started = time.perf_counter()
            candidate = deterministic_numeric_answer(route, evidence)
            if candidate is None:
                issues.append("insufficient_numeric_comparison_evidence")
            else:
                validation = validate_answer(candidate, evidence)
                validation_issues = validation.issues
                if validation.valid:
                    answer = candidate
                    status = "verified"
                    answer_strategy = "deterministic_numeric"
                else:
                    status = "rejected"
                    issues.append("answer_validation_failed")
            generation_ms = (time.perf_counter() - generation_started) * 1000.0
        elif evidence and self.evren is None:
            issues.append("answer_provider_unavailable")
        elif evidence:
            generation_started = time.perf_counter()
            try:
                candidate = self.evren.chat(
                    build_answer_messages(
                        route,
                        evidence,
                        original_query=payload.query,
                        conversation_history=recent,
                        conversation_summary=session.state.conversation_summary,
                    )
                )
                if _is_fail_closed_answer(candidate):
                    answer = FAIL_CLOSED_ANSWER
                    status = "insufficient_evidence"
                    issues.append("model_reported_insufficient_evidence")
                else:
                    validation = validate_answer(candidate, evidence)
                    validation_issues = validation.issues
                    if validation.valid:
                        answer = candidate
                        status = "verified"
                        answer_strategy = "evren_llm"
                    else:
                        status = "rejected"
                        issues.append("answer_validation_failed")
            except (ProviderUnavailable, ProviderProtocolError, ValueError):
                issues.append("answer_provider_unavailable")
            generation_ms = (time.perf_counter() - generation_started) * 1000.0

        if validation_issues:
            issues.extend(validation_issues)
        verified_evidence = []
        verified_records = []
        if status == "verified":
            cited_source_ids = {
                f"S{value}" for value in CITATION_PATTERN.findall(answer)
            }
            verified_evidence = [
                item for item in evidence if item.source_id in cited_source_ids
            ]
            record_by_chunk = {item.chunk_id: item for item in records}
            verified_records = [
                record_by_chunk[item.chunk_id]
                for item in verified_evidence
                if item.chunk_id in record_by_chunk
            ]
        if status == "verified":
            state_base = self._next_state(
                route,
                verified_records,
                verified_evidence,
                context_records=answer_records,
                context_evidence=evidence,
                base_state=session.state,
            )
        else:
            state_base = self._state_after_unverified_route(
                session.state,
                route,
            )
        new_state = self._remember_turn(
            state_base,
            payload.query,
            answer,
            status,
            route,
        )
        self.sessions.finish_turn(
            session,
            turn_id,
            answer,
            status,
            route,
            evidence,
            new_state,
            user_content=payload.query,
        )
        display_by_key = {value: key for key, value in BANK_KEYS.items()}
        covered_banks = list(
            dict.fromkeys(
                display_by_key.get(record.bank_key, record.bank_name)
                for record in answer_records
            )
        )
        missing_evidence_banks = (
            [name for name in BANK_KEYS if name not in covered_banks]
            if bank_coverage_mode
            else []
        )
        if bank_coverage_mode and evidence and missing_evidence_banks:
            issues.append("partial_bank_coverage")
        diagnostics = {
            **retrieval_diagnostics,
            "route_ms": round(route_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "retrieval_performed": True,
            "answer_validation_passed": status == "verified",
            "answer_strategy": answer_strategy,
            "classification_policy_enforced": (
                self.settings.enforce_classification_policy
            ),
            "classification_status_counts": {
                status_name: sum(
                    item.classification_status == status_name
                    for item in records
                )
                for status_name in ("accepted", "review", "required", "verified")
            },
            "withheld_candidates": withheld_evidence,
            "low_confidence_candidates_withheld": low_confidence_withheld,
            "classification_conflict_candidates_withheld": conflict_withheld,
            "selected_evidence_records": len(answer_records),
            "fresh_evidence_records": len(fresh_records),
            "session_evidence_refs": len(session_evidence_refs),
            "session_evidence_hydrated": len(hydrated_session_records),
            "session_evidence_selected": selected_session_evidence,
            "bank_coverage_mode": bank_coverage_mode,
            "covered_banks": covered_banks if bank_coverage_mode else [],
            "missing_evidence_banks": missing_evidence_banks,
            "conversation_messages_used": len(recent),
            "assistant_context_used": any(
                item.get("role") == "assistant" for item in recent
            ),
            "conversation_turn_count": new_state.conversation_turn_count,
        }
        return RagV2ChatResponse(
            session_id=session.token,
            query=payload.query,
            standalone_query=route.standalone_query,
            answer=answer,
            status=status,
            inherited_context=inherited,
            route=route,
            evidence=evidence,
            issues=list(dict.fromkeys(issues)),
            diagnostics=diagnostics,
        )
