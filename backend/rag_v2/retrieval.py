from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime, time, timezone
from typing import Any

from psycopg import Error as DatabaseError
from psycopg_pool import PoolTimeout

from .database import RagDatabasePool
from .evidence import FIELD_TO_FACT_TYPES
from .models import QueryRoute, SearchRecord, StructuredFact
from .providers import (
    EvrenClient,
    ProviderProtocolError,
    ProviderUnavailable,
    QdrantRestClient,
)
from .identity import normalize_text
from .routing import bank_keys
from .settings import RagV2Settings


NUMERIC_FIELDS = {
    "amount",
    "rate",
    "maturity",
    "fee",
    "reward",
    "spending_threshold",
}

PRODUCT_FAMILIES = {
    "KART": "card",
    "KART_KAMPANYASI": "card",
    "KART_URUNU": "card",
    "ALISVERIS_PUANI": "card",
    "KONUT_FINANSMANI": "housing",
    "TASIT_FINANSMANI": "vehicle",
    "IHTIYAC_FINANSMANI": "personal",
    "TICARI_FINANSMAN": "commercial",
    "KATILMA_HESABI": "investment",
    "CARI_HESAP": "account",
    "YATIRIM_URUNU": "investment",
    "SIGORTA_TEKAFUL": "insurance",
    "SIGORTA_TEKAFUL_URUNU": "insurance",
    "ODEME_TRANSFER": "payment",
    "ODEME_TRANSFER_HIZMETI": "payment",
    "DIGER_KAMPANYA": "campaign",
    "MOBIL_UYGULAMA_KAMPANYASI": "campaign",
}

LEXICAL_STOP_WORDS = {
    "acaba",
    "guncel",
    "gecmis",
    "icin",
    "kadar",
    "midir",
    "nedir",
    "olan",
    "olarak",
    "peki",
}

LEXICAL_ROOTS = {
    "esigi": "esik",
    "finansmani": "finansman",
    "orani": "oran",
    "odulu": "odul",
    "suresi": "sure",
    "tarihi": "tarih",
    "tutari": "tutar",
    "ucreti": "ucret",
    "vadesi": "vade",
}


def build_lexical_tsquery(query: str) -> str:
    tokens: list[str] = []
    for raw in re.findall(r"[a-z0-9]+", normalize_text(query)):
        token = LEXICAL_ROOTS.get(raw, raw)
        if token in LEXICAL_STOP_WORDS or len(token) < 3:
            continue
        if token not in tokens:
            tokens.append(token)
        if len(tokens) == 32:
            break
    if not tokens:
        return "ragv2nomatch"
    return " | ".join(f"{token}:*" for token in tokens)


def product_family(value: str) -> str:
    normalized = str(value or "").strip().upper()
    return PRODUCT_FAMILIES.get(normalized, normalized)


def rrf_score(
    dense_rank: int | None,
    lexical_rank: int | None,
    *,
    dense_weight: float = 1.0,
    lexical_weight: float = 0.5,
    rrf_k: int = 60,
) -> float:
    score = 0.0
    if dense_rank is not None:
        score += dense_weight / (rrf_k + dense_rank)
    if lexical_rank is not None:
        score += lexical_weight / (rrf_k + lexical_rank)
    return score


def _product_matches(record: SearchRecord, products: list[str]) -> bool:
    if not products:
        return False
    record_products = {product_family(item) for item in record.product_types}
    if record.primary_product:
        record_products.add(product_family(record.primary_product))
    query_products = {product_family(item) for item in products}
    return bool(record_products.intersection(query_products))


def _requested_fact_types(field_types: list[str]) -> set[str]:
    return set().union(
        *(FIELD_TO_FACT_TYPES.get(field_type, set()) for field_type in field_types)
    )


def _field_matches(record: SearchRecord, field_types: list[str]) -> bool:
    return _field_coverage(record, field_types) > 0


def _field_coverage(record: SearchRecord, field_types: list[str]) -> int:
    record_fact_types = {fact.fact_type for fact in record.facts}
    return sum(
        bool(FIELD_TO_FACT_TYPES.get(field_type, set()).intersection(record_fact_types))
        for field_type in dict.fromkeys(field_types)
    )


def _relevance_tier(record: SearchRecord, route: QueryRoute) -> int:
    product_requested = bool(route.product_types)
    field_requested = bool(route.field_types)
    product_match = (
        _product_matches(record, route.product_types) if product_requested else True
    )
    field_match = (
        _field_matches(record, route.field_types) if field_requested else True
    )
    if product_match and field_match:
        return 0
    if field_requested and field_match:
        return 1
    if product_requested and product_match:
        return 2
    return 3


def _numeric_value(
    record: SearchRecord,
    field_types: list[str],
    *,
    descending: bool,
) -> float | None:
    requested = _requested_fact_types(field_types)
    if not requested:
        return None
    values: list[float] = []
    for fact in record.facts:
        if fact.fact_type not in requested:
            continue
        normalized = fact.normalized_value or {}
        try:
            value = float(normalized.get("value"))
        except (TypeError, ValueError):
            continue
        unit = str(normalized.get("unit") or "").strip().lower()
        if fact.fact_type == "VADE_SURESI" and unit == "year":
            value *= 12.0
        elif fact.fact_type == "VADE_SURESI" and unit == "day":
            value /= 30.0
        values.append(value)
    if not values:
        return None
    return max(values) if descending else min(values)


def _numeric_descending(route: QueryRoute) -> bool:
    normalized_query = normalize_text(route.standalone_query)
    if "en dusuk" in normalized_query:
        return False
    if "en yuksek" in normalized_query:
        return True
    return bool(set(route.field_types).intersection({"amount", "maturity", "reward"}))


def rank_relevant_records(
    records: list[SearchRecord],
    route: QueryRoute,
    *,
    enforce_classification_policy: bool = False,
) -> list[SearchRecord]:
    numeric_order = route.intent in {"compare", "list", "calculate"} and bool(
        route.field_types
    )
    descending = _numeric_descending(route)

    def key(record: SearchRecord) -> tuple[Any, ...]:
        trust_tier = (
            0
            if not enforce_classification_policy
            or _is_primary_evidence_candidate(record)
            else 1
        )
        relevance_tier = _relevance_tier(record, route)
        field_coverage = _field_coverage(record, route.field_types)
        stable_tail = (
            -record.rrf_score,
            record.bank_key,
            record.offer_id,
            record.document_id,
            record.chunk_id,
        )
        if not numeric_order:
            return (trust_tier, relevance_tier, -field_coverage, *stable_tail)
        value = _numeric_value(
            record,
            route.field_types,
            descending=descending,
        )
        numeric_key = -value if value is not None and descending else value
        return (
            trust_tier,
            relevance_tier,
            -field_coverage,
            value is None,
            numeric_key if numeric_key is not None else 0.0,
            *stable_tail,
        )

    return sorted(records, key=key)


def _diversify_relevance_tiers(
    records: list[SearchRecord],
    route: QueryRoute,
    top_k: int,
) -> list[SearchRecord]:
    selected: list[SearchRecord] = []
    for tier in range(4):
        tier_records = [
            record for record in records if _relevance_tier(record, route) == tier
        ]
        if not tier_records:
            continue
        selected.extend(diversify_banks(tier_records, top_k - len(selected)))
        if len(selected) >= top_k:
            break
    return selected


def fuse_results(
    dense: list[SearchRecord],
    lexical: list[SearchRecord],
    *,
    settings: RagV2Settings,
    product_types: list[str] | None = None,
) -> list[SearchRecord]:
    combined: dict[str, SearchRecord] = {}
    for rank, source in enumerate(dense, start=1):
        record = source.model_copy(deep=True)
        record.dense_rank = rank
        combined[record.chunk_id] = record
    for rank, source in enumerate(lexical, start=1):
        existing = combined.get(source.chunk_id)
        if existing is None:
            existing = source.model_copy(deep=True)
            combined[source.chunk_id] = existing
        existing.lexical_rank = rank
        existing.lexical_score = source.lexical_score
    products = product_types or []
    for record in combined.values():
        record.rrf_score = rrf_score(
            record.dense_rank,
            record.lexical_rank,
            dense_weight=settings.dense_weight,
            lexical_weight=settings.lexical_weight,
            rrf_k=settings.rrf_k,
        )
        if _product_matches(record, products):
            record.product_boost = settings.product_soft_boost
            record.rrf_score += settings.product_soft_boost
    return sorted(
        combined.values(),
        key=lambda item: (
            -item.rrf_score,
            item.offer_id,
            item.document_id,
            item.chunk_id,
        ),
    )


def deduplicate_results(records: list[SearchRecord]) -> list[SearchRecord]:
    seen_offers: set[str] = set()
    offer_unique: list[SearchRecord] = []
    for record in records:
        if record.offer_id in seen_offers:
            continue
        seen_offers.add(record.offer_id)
        offer_unique.append(record)

    seen_documents: set[str] = set()
    output: list[SearchRecord] = []
    for record in offer_unique:
        if record.document_id in seen_documents:
            continue
        seen_documents.add(record.document_id)
        output.append(record)
    return output


def diversify_banks(
    records: list[SearchRecord],
    top_k: int,
) -> list[SearchRecord]:
    if top_k <= 0:
        return []
    by_bank: dict[str, list[SearchRecord]] = defaultdict(list)
    bank_order: list[str] = []
    for record in records:
        if record.bank_key not in by_bank:
            bank_order.append(record.bank_key)
        by_bank[record.bank_key].append(record)
    output: list[SearchRecord] = []
    offset = 0
    while len(output) < top_k:
        added = False
        for bank in bank_order:
            values = by_bank[bank]
            if offset < len(values):
                output.append(values[offset])
                added = True
                if len(output) == top_k:
                    break
        if not added:
            break
        offset += 1
    return output


def _is_primary_evidence_candidate(record: SearchRecord) -> bool:
    return record.classification_status != "required" and (
        not record.classification_conflict
        or record.classification_status == "verified"
    )


def _facts(value: Any) -> list[StructuredFact]:
    if not isinstance(value, list):
        return []
    output: list[StructuredFact] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            output.append(StructuredFact.model_validate(item))
        except ValueError:
            continue
    return output


def record_from_mapping(
    value: dict[str, Any],
    *,
    dense_score: float | None = None,
    lexical_score: float | None = None,
) -> SearchRecord:
    confidence = value.get("classification_confidence")
    try:
        confidence_value = float(confidence if confidence is not None else 0.0)
    except (TypeError, ValueError):
        confidence_value = 0.0
    confidence_value = min(max(confidence_value, 0.0), 1.0)
    status = str(value.get("classification_status") or "required").lower()
    if status not in {"accepted", "review", "required", "verified"}:
        status = "required"
    scope = str(value.get("scope") or "current").lower()
    if scope not in {"current", "historical"}:
        scope = "current"
    return SearchRecord(
        chunk_id=str(value.get("chunk_id") or ""),
        offer_id=str(value.get("offer_id") or ""),
        document_id=str(value.get("document_id") or ""),
        bank_key=str(value.get("bank_key") or ""),
        bank_name=str(value.get("bank_name") or ""),
        primary_product=value.get("primary_product"),
        product_types=list(value.get("product_types") or []),
        page_title=value.get("page_title"),
        section_heading=value.get("section_heading"),
        source_url=value.get("source_url"),
        scope=scope,
        effective_date=value.get("effective_date"),
        campaign_start=value.get("campaign_start"),
        campaign_end=value.get("campaign_end"),
        content=str(value.get("content") or ""),
        facts=_facts(value.get("facts")),
        classification_confidence=confidence_value,
        classification_status=status,
        classification_conflict=bool(value.get("classification_conflict", False)),
        product_scores=dict(value.get("product_scores") or {}),
        dense_score=dense_score,
        lexical_score=lexical_score,
    )


def _iso_datetime(value: date, end: bool = False) -> str:
    boundary = time.max if end else time.min
    return datetime.combine(value, boundary, tzinfo=timezone.utc).isoformat()


def _route_date_bounds(route: QueryRoute) -> tuple[date | None, date | None]:
    if route.date_from is not None or route.date_to is not None:
        return route.date_from, route.date_to
    if route.year is not None:
        return date(route.year, 1, 1), date(route.year, 12, 31)
    return None, None


def _uses_campaign_period(route: QueryRoute) -> bool:
    return (
        bool(
            {"reward", "spending_threshold", "campaign_date"}.intersection(
                route.field_types
            )
        )
        or any("KAMPANYA" in item.upper() for item in route.product_types)
        or "ALISVERIS_PUANI" in {
            item.upper() for item in route.product_types
        }
        or "kampanya" in normalize_text(route.standalone_query)
    )


def build_qdrant_filter(
    route: QueryRoute,
    *,
    min_confidence: float | None,
) -> dict[str, Any] | None:
    must: list[dict[str, Any]] = []
    date_from, date_to = _route_date_bounds(route)
    if route.scope != "all":
        must.append({"key": "scope", "match": {"value": route.scope}})
    keys = bank_keys(route.banks)
    if keys:
        must.append({"key": "bank_key", "match": {"any": keys}})
    if route.offer_ids:
        must.append(
            {"key": "offer_id", "match": {"any": route.offer_ids}}
        )
    if date_from or date_to:
        campaign_period = _uses_campaign_period(route)
        range_filter: dict[str, Any] = {}
        if date_from:
            range_filter["gte"] = _iso_datetime(date_from)
        if date_to:
            range_filter["lte"] = _iso_datetime(date_to, end=True)
        if campaign_period:
            must.append(
                {
                    "should": [
                        {
                            "must_not": [
                                {"is_null": {"key": "campaign_start"}}
                            ]
                        },
                        {
                            "must_not": [
                                {"is_null": {"key": "campaign_end"}}
                            ]
                        },
                    ]
                }
            )
            if date_from:
                must.append(
                    {
                        "should": [
                            {
                                "key": "campaign_end",
                                "range": {"gte": _iso_datetime(date_from)},
                            },
                            {
                                "is_null": {"key": "campaign_end"}
                            },
                        ]
                    }
                )
            if date_to:
                must.append(
                    {
                        "should": [
                            {
                                "key": "campaign_start",
                                "range": {
                                    "lte": _iso_datetime(date_to, end=True)
                                },
                            },
                            {
                                "is_null": {"key": "campaign_start"}
                            },
                        ]
                    }
                )
        else:
            must.append({"key": "effective_date", "range": range_filter})
    if min_confidence is not None:
        must.append(
            {
                "should": [
                    {
                        "key": "classification_status",
                        "match": {"value": "verified"},
                    },
                    {
                        "must": [
                            {
                                "key": "classification_confidence",
                                "range": {"gte": min_confidence},
                            },
                            {
                                "key": "classification_status",
                                "match": {"any": ["accepted", "review"]},
                            },
                            {
                                "key": "classification_conflict",
                                "match": {"value": False},
                            },
                        ]
                    },
                ]
            }
        )
    return {"must": must} if must else None


LEXICAL_SQL = """
WITH query_input AS (
    SELECT to_tsquery(
        'simple',
        %(lexical_query)s
    ) AS text_query
)
SELECT
    chunks.chunk_id,
    chunks.offer_id,
    chunks.scope,
    chunks.document_id,
    chunks.bank_key,
    chunks.bank_name,
    chunks.primary_product,
    chunks.product_types,
    chunks.product_scores,
    chunks.classification_confidence,
    chunks.classification_status,
    chunks.classification_conflict,
    chunks.page_title,
    chunks.section_heading,
    chunks.source_url,
    chunks.effective_date,
    chunks.campaign_start,
    chunks.campaign_end,
    chunks.content,
    chunks.facts,
    ts_rank_cd(chunks.search_vector, query_input.text_query) AS lexical_score
FROM rag_chunks AS chunks
CROSS JOIN query_input
WHERE chunks.search_vector @@ query_input.text_query
  AND (%(scope)s = 'all' OR chunks.scope = %(scope)s)
  AND (%(bank_keys)s::text[] IS NULL OR chunks.bank_key = ANY(%(bank_keys)s))
  AND (%(offer_ids)s::text[] IS NULL OR chunks.offer_id = ANY(%(offer_ids)s))
  AND (
      (%(date_from)s::date IS NULL AND %(date_to)s::date IS NULL)
      OR %(date_mode)s <> 'campaign'
      OR chunks.campaign_start IS NOT NULL
      OR chunks.campaign_end IS NOT NULL
  )
  AND (
      %(date_from)s::date IS NULL
      OR (
          %(date_mode)s = 'campaign'
          AND (
              chunks.campaign_end IS NULL
              OR chunks.campaign_end >= %(date_from)s
          )
      )
      OR (
          %(date_mode)s = 'effective'
          AND chunks.effective_date >= %(date_from)s
      )
  )
  AND (
      %(date_to)s::date IS NULL
      OR (
          %(date_mode)s = 'campaign'
          AND (
              chunks.campaign_start IS NULL
              OR chunks.campaign_start <= %(date_to)s
          )
      )
      OR (
          %(date_mode)s = 'effective'
          AND chunks.effective_date <= %(date_to)s
      )
  )
  AND (
      %(min_confidence)s::double precision IS NULL
      OR chunks.classification_status = 'verified'
      OR (
          chunks.classification_confidence >= %(min_confidence)s
          AND chunks.classification_status IN ('accepted', 'review')
          AND NOT chunks.classification_conflict
      )
  )
ORDER BY lexical_score DESC, chunks.chunk_id
LIMIT %(limit)s
"""


DENSE_HYDRATE_SQL = """
SELECT
    chunks.chunk_id,
    chunks.offer_id,
    chunks.scope,
    chunks.document_id,
    chunks.bank_key,
    chunks.bank_name,
    chunks.primary_product,
    chunks.product_types,
    chunks.product_scores,
    chunks.classification_confidence,
    chunks.classification_status,
    chunks.classification_conflict,
    chunks.page_title,
    chunks.section_heading,
    chunks.source_url,
    chunks.effective_date,
    chunks.campaign_start,
    chunks.campaign_end,
    chunks.content,
    chunks.facts
FROM rag_chunks AS chunks
WHERE chunks.chunk_id::TEXT = ANY(%(chunk_ids)s::TEXT[])
  AND (%(scope)s = 'all' OR chunks.scope = %(scope)s)
  AND (%(bank_keys)s::TEXT[] IS NULL OR chunks.bank_key = ANY(%(bank_keys)s))
  AND (%(offer_ids)s::TEXT[] IS NULL OR chunks.offer_id = ANY(%(offer_ids)s))
  AND (
      (%(date_from)s::DATE IS NULL AND %(date_to)s::DATE IS NULL)
      OR %(date_mode)s <> 'campaign'
      OR chunks.campaign_start IS NOT NULL
      OR chunks.campaign_end IS NOT NULL
  )
  AND (
      %(date_from)s::DATE IS NULL
      OR (
          %(date_mode)s = 'campaign'
          AND (
              chunks.campaign_end IS NULL
              OR chunks.campaign_end >= %(date_from)s
          )
      )
      OR (
          %(date_mode)s = 'effective'
          AND chunks.effective_date >= %(date_from)s
      )
  )
  AND (
      %(date_to)s::DATE IS NULL
      OR (
          %(date_mode)s = 'campaign'
          AND (
              chunks.campaign_start IS NULL
              OR chunks.campaign_start <= %(date_to)s
          )
      )
      OR (
          %(date_mode)s = 'effective'
          AND chunks.effective_date <= %(date_to)s
      )
  )
  AND (
      %(min_confidence)s::DOUBLE PRECISION IS NULL
      OR chunks.classification_status = 'verified'
      OR (
          chunks.classification_confidence >= %(min_confidence)s
          AND chunks.classification_status IN ('accepted', 'review')
          AND NOT chunks.classification_conflict
      )
  )
"""


class HybridRetriever:
    def __init__(
        self,
        pool: RagDatabasePool,
        settings: RagV2Settings,
        evren: EvrenClient | None,
        qdrant: QdrantRestClient | None,
    ) -> None:
        self.pool = pool
        self.settings = settings
        self.evren = evren
        self.qdrant = qdrant

    def _requires_trusted_numeric(self, route: QueryRoute) -> bool:
        return self.settings.enforce_classification_policy and (
            route.intent in {"compare", "calculate"}
            or bool(NUMERIC_FIELDS.intersection(route.field_types))
        )

    def _filter_parameters(
        self,
        route: QueryRoute,
        min_confidence: float | None,
    ) -> dict[str, Any]:
        date_from, date_to = _route_date_bounds(route)
        return {
            "scope": route.scope,
            "bank_keys": bank_keys(route.banks) or None,
            "offer_ids": route.offer_ids or None,
            "date_from": date_from,
            "date_to": date_to,
            "date_mode": "campaign" if _uses_campaign_period(route) else "effective",
            "min_confidence": min_confidence,
        }

    def _execute(self, query: str, parameters: dict[str, Any]):
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(self.settings.db_statement_timeout_ms),),
                )
                return connection.execute(query, parameters).fetchall()

    def _lexical(
        self,
        route: QueryRoute,
        candidate_count: int,
        min_confidence: float | None,
    ) -> list[SearchRecord]:
        parameters = {
            **self._filter_parameters(route, min_confidence),
            "lexical_query": build_lexical_tsquery(route.standalone_query),
            "limit": candidate_count,
        }
        rows = self._execute(LEXICAL_SQL, parameters)
        return [
            record_from_mapping(row, lexical_score=float(row["lexical_score"]))
            for row in rows
        ]

    def _hydrate_dense(
        self,
        route: QueryRoute,
        chunk_ids: list[str],
        scores: dict[str, float],
        min_confidence: float | None,
    ) -> list[SearchRecord]:
        if not chunk_ids:
            return []
        parameters = {
            **self._filter_parameters(route, min_confidence),
            "chunk_ids": chunk_ids,
        }
        rows = self._execute(DENSE_HYDRATE_SQL, parameters)
        by_chunk = {
            str(row["chunk_id"]).strip(): record_from_mapping(
                row,
                dense_score=scores.get(str(row["chunk_id"]).strip(), 0.0),
            )
            for row in rows
        }
        return [by_chunk[item] for item in chunk_ids if item in by_chunk]

    def hydrate_context_records(
        self,
        route: QueryRoute,
        references: list[dict[str, Any]],
    ) -> list[SearchRecord]:
        reference_by_chunk: dict[str, dict[str, Any]] = {}
        chunk_ids: list[str] = []
        for reference in references:
            chunk_id = str(reference.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in reference_by_chunk:
                continue
            reference_by_chunk[chunk_id] = reference
            chunk_ids.append(chunk_id)
        if not chunk_ids:
            return []
        min_confidence = (
            self.settings.review_confidence
            if self._requires_trusted_numeric(route)
            else None
        )
        try:
            hydrated = self._hydrate_dense(
                route,
                chunk_ids,
                {},
                min_confidence,
            )
        except (DatabaseError, PoolTimeout):
            return []

        requested_products = set(route.product_types)
        output: list[SearchRecord] = []
        for record in hydrated:
            reference = reference_by_chunk.get(record.chunk_id)
            if reference is None:
                continue
            if (
                record.offer_id != str(reference.get("offer_id") or "")
                or record.document_id
                != str(reference.get("document_id") or "")
            ):
                continue
            record_products = set(record.product_types)
            if record.primary_product:
                record_products.add(record.primary_product)
            if requested_products and requested_products.isdisjoint(record_products):
                continue
            output.append(record)
        return output

    def _dense(
        self,
        route: QueryRoute,
        candidate_count: int,
        min_confidence: float | None,
    ) -> list[SearchRecord]:
        if self.evren is None or self.qdrant is None:
            return []
        vector = self.evren.embed([route.standalone_query])[0]
        query_filter = build_qdrant_filter(
            route, min_confidence=min_confidence
        )
        points = self.qdrant.query(
            vector,
            query_filter=query_filter,
            limit=candidate_count,
        )
        chunk_ids: list[str] = []
        scores: dict[str, float] = {}
        for point in points:
            payload = point.get("payload")
            if not isinstance(payload, dict):
                continue
            try:
                chunk_id = str(payload.get("chunk_id") or "").strip()
                if len(chunk_id) != 64:
                    continue
                scores[chunk_id] = float(point.get("score") or 0.0)
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
            except (TypeError, ValueError):
                continue
        return self._hydrate_dense(
            route,
            chunk_ids,
            scores,
            min_confidence,
        )

    def retrieve(
        self,
        route: QueryRoute,
        top_k: int,
        *,
        use_reranker: bool = False,
    ) -> tuple[list[SearchRecord], dict[str, Any], list[str]]:
        candidate_count = min(
            max(top_k * self.settings.candidate_multiplier, 40), 500
        )
        issues: list[str] = []
        min_confidence = (
            self.settings.review_confidence
            if self._requires_trusted_numeric(route)
            else None
        )

        dense: list[SearchRecord] = []
        try:
            dense = self._dense(route, candidate_count, min_confidence)
        except (ProviderUnavailable, ProviderProtocolError):
            issues.append("dense_provider_unavailable")
        except (DatabaseError, PoolTimeout):
            issues.append("dense_database_unavailable")

        try:
            lexical = self._lexical(route, candidate_count, min_confidence)
        except (DatabaseError, PoolTimeout):
            lexical = []
            issues.append("lexical_database_unavailable")
        fused = fuse_results(
            dense,
            lexical,
            settings=self.settings,
            product_types=route.product_types,
        )
        enforce_policy = self.settings.enforce_classification_policy
        deduplicated = deduplicate_results(
            rank_relevant_records(
                fused,
                route,
                enforce_classification_policy=enforce_policy,
            )
        )
        if enforce_policy:
            primary = rank_relevant_records(
                [
                    item
                    for item in deduplicated
                    if _is_primary_evidence_candidate(item)
                ],
                route,
                enforce_classification_policy=True,
            )
            fallback = rank_relevant_records(
                [
                    item
                    for item in deduplicated
                    if not _is_primary_evidence_candidate(item)
                ],
                route,
                enforce_classification_policy=True,
            )
        else:
            primary = rank_relevant_records(
                deduplicated,
                route,
                enforce_classification_policy=False,
            )
            fallback = []
        bank_diversity_applied = route.intent == "compare" or (
            not route.banks
            and route.intent in {"lookup", "list"}
            and bool(route.product_types or route.field_types)
        )
        if bank_diversity_applied:
            selected = _diversify_relevance_tiers(primary, route, top_k)
            if len(selected) < top_k:
                remaining = [
                    item
                    for item in fallback
                    if item.chunk_id
                    not in {record.chunk_id for record in selected}
                ]
                selected.extend(
                    _diversify_relevance_tiers(
                        remaining,
                        route,
                        top_k - len(selected),
                    )
                )
        else:
            selected = (primary + fallback)[:top_k]

        if not selected and min_confidence is None:
            issues.append("no_matching_evidence")
        if use_reranker:
            if not (
                self.settings.reranker_enabled
                and self.settings.reranker_validated
            ):
                issues.append("reranker_disabled_until_baseline_is_beaten")
            else:
                issues.append("reranker_not_implemented")

        diagnostics = {
            "dense_candidates": len(dense),
            "lexical_candidates": len(lexical),
            "fused_candidates": len(fused),
            "deduplicated_candidates": len(deduplicated),
            "primary_candidates": len(primary),
            "fallback_candidates": len(fallback),
            "returned_candidates": len(selected),
            "product_matched_candidates": sum(
                _product_matches(item, route.product_types)
                for item in deduplicated
            ),
            "field_matched_candidates": sum(
                _field_matches(item, route.field_types)
                for item in deduplicated
            ),
            "returned_product_matches": sum(
                _product_matches(item, route.product_types) for item in selected
            ),
            "returned_field_matches": sum(
                _field_matches(item, route.field_types) for item in selected
            ),
            "dense_weight": self.settings.dense_weight,
            "lexical_weight": self.settings.lexical_weight,
            "rrf_k": self.settings.rrf_k,
            "product_filter_mode": "soft",
            "classification_policy_enforced": enforce_policy,
            "bank_diversity_applied": bank_diversity_applied,
            "min_confidence": min_confidence,
            "reranker_used": False,
        }
        return selected, diagnostics, issues
