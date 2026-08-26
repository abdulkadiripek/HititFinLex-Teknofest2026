from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from fact_context_rules import (
    AUTO_THRESHOLDS,
    campaign_amount_roles,
    campaign_date_context_pass,
    excluded_context_reason,
)
from fact_surface_rules import validate_entity_surface
from hybrid_search import get_connection
from intake_service import ALLOWED_ENTITY_LABELS_BY_PRODUCT, PRODUCT_TITLES


PIPELINE_VERSION = "ner_v4_rules_v3_0"
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_CAMPAIGN_TYPE = "KONUT_FINANSMANI"
DEFAULT_LIMIT = 50
DEFAULT_CHUNK_CHARS = 1000
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_REVIEW_THRESHOLD = 0.60
DEFAULT_CLASSIFICATION_THRESHOLD = 0.80
API_PREDICTION_THRESHOLD = 0.40


NUMERIC_LABELS = {
    "EKSPERTIZ_UCRETI",
    "FINANSMAN_TUTARI",
    "HARCAMA_ESIGI",
    "HARCAMA_UST_LIMITI",
    "INDIRIM_ORANI",
    "INDIRIM_TUTARI",
    "IPOTEK_TESIS_UCRETI",
    "ISLEM_ALT_LIMITI",
    "ISLEM_UST_LIMITI",
    "KAR_PAYI_ORANI",
    "KAR_PAYLASIM_ORANI",
    "MINIMUM_BAKIYE",
    "ODUL_TUTARI",
    "TAHSIS_UCRETI",
    "TAKSIT_SAYISI",
    "VADE_SURESI",
}


STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ner_document_state (
    document_id BIGINT PRIMARY KEY
        REFERENCES documents(id) ON DELETE CASCADE,
    content_hash CHAR(64) NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    pipeline_version VARCHAR(64) NOT NULL,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    review_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


REVIEW_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS comparison_fact_review_queue (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL
        REFERENCES documents(id) ON DELETE CASCADE,
    fact_type VARCHAR(64) NOT NULL,
    fact_text TEXT NOT NULL,
    normalized_value JSONB,
    evidence_text TEXT NOT NULL,
    extraction_method VARCHAR(64) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL
        CHECK (confidence >= 0 AND confidence <= 1),
    source_chunk INTEGER NOT NULL,
    fact_key CHAR(64) NOT NULL,
    review_reason VARCHAR(128) NOT NULL,
    review_status VARCHAR(16) NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, fact_key)
)
"""


REVIEW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS comparison_fact_review_status_idx
ON comparison_fact_review_queue (review_status, confidence DESC)
"""


FACT_UPSERT_SQL = """
INSERT INTO comparison_facts (
    document_id,
    fact_type,
    fact_text,
    normalized_value,
    evidence_text,
    extraction_method,
    confidence,
    source_chunk,
    fact_key
)
SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s
WHERE NOT EXISTS (
    SELECT 1
    FROM comparison_facts existing
    WHERE existing.document_id = %s
      AND existing.fact_type = %s
      AND LOWER(BTRIM(existing.fact_text)) = LOWER(BTRIM(%s))
)
AND NOT EXISTS (
    SELECT 1
    FROM passages p
    JOIN entities e ON e.passage_id = p.id
    WHERE p.document_id = %s
      AND e.entity_label = %s
      AND LOWER(BTRIM(e.entity_text)) = LOWER(BTRIM(%s))
)
ON CONFLICT (document_id, fact_key) DO UPDATE SET
    normalized_value = EXCLUDED.normalized_value,
    evidence_text = EXCLUDED.evidence_text,
    extraction_method = EXCLUDED.extraction_method,
    confidence = EXCLUDED.confidence,
    source_chunk = EXCLUDED.source_chunk,
    updated_at = NOW()
"""


REVIEW_UPSERT_SQL = """
INSERT INTO comparison_fact_review_queue (
    document_id,
    fact_type,
    fact_text,
    normalized_value,
    evidence_text,
    extraction_method,
    confidence,
    source_chunk,
    fact_key,
    review_reason
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (document_id, fact_key) DO UPDATE SET
    normalized_value = EXCLUDED.normalized_value,
    evidence_text = EXCLUDED.evidence_text,
    confidence = EXCLUDED.confidence,
    source_chunk = EXCLUDED.source_chunk,
    review_reason = EXCLUDED.review_reason,
    updated_at = NOW()
WHERE comparison_fact_review_queue.review_status = 'pending'
"""


STATE_UPSERT_SQL = """
INSERT INTO ner_document_state (
    document_id,
    content_hash,
    model_name,
    pipeline_version,
    accepted_count,
    review_count,
    rejected_count
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (document_id) DO UPDATE SET
    content_hash = EXCLUDED.content_hash,
    model_name = EXCLUDED.model_name,
    pipeline_version = EXCLUDED.pipeline_version,
    accepted_count = EXCLUDED.accepted_count,
    review_count = EXCLUDED.review_count,
    rejected_count = EXCLUDED.rejected_count,
    processed_at = NOW()
"""


@dataclass
class Document:
    document_id: int
    bank_name: str
    page_title: str
    source_url: str
    campaign_type_code: str
    raw_text: str
    verified: bool


@dataclass
class Candidate:
    fact_type: str
    fact_text: str
    normalized_value: dict[str, Any] | None
    evidence_text: str
    confidence: float
    source_chunk: int
    decision: str
    reason: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract NER facts from new or changed documents and route them "
            "to automatic acceptance, review, or rejection."
        )
    )
    parser.add_argument(
        "--campaign-type",
        default=DEFAULT_CAMPAIGN_TYPE,
        help=f"Campaign type code. Default: {DEFAULT_CAMPAIGN_TYPE}",
    )
    parser.add_argument(
        "--document-id",
        action="append",
        type=int,
        default=[],
        help="Process one document id. Repeat for multiple documents.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=DEFAULT_REVIEW_THRESHOLD,
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess documents even when their content hash is unchanged.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write accepted classifications, facts, and review candidates "
            "to PostgreSQL."
        ),
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Print the evidence text for every extracted candidate.",
    )
    args = parser.parse_args()

    args.campaign_type = args.campaign_type.strip().upper()
    args.api_url = args.api_url.rstrip("/")
    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    if any(value < 1 for value in args.document_id):
        parser.error("--document-id values must be positive.")
    if args.chunk_chars < 400 or args.chunk_chars > 8000:
        parser.error("--chunk-chars must be between 400 and 8000.")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_chars:
        parser.error("--chunk-overlap must be smaller than --chunk-chars.")
    if not 0.0 <= args.review_threshold <= 1.0:
        parser.error("--review-threshold must be between 0 and 1.")
    if not 0.0 <= args.classification_threshold <= 1.0:
        parser.error("--classification-threshold must be between 0 and 1.")
    return args


def fold_text(value: str) -> str:
    translated = value.translate(str.maketrans({"ı": "i", "İ": "I"}))
    decomposed = unicodedata.normalize("NFKD", translated)
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip().casefold()


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_fact_key(document_id: int, candidate: Candidate) -> str:
    value = "|".join(
        [
            str(document_id),
            PIPELINE_VERSION,
            candidate.fact_type,
            fold_text(candidate.fact_text),
            fold_text(candidate.evidence_text),
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def table_exists(cursor, table_name: str) -> bool:
    cursor.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{table_name}",))
    return bool(cursor.fetchone()[0])


def load_state() -> dict[int, tuple[str, str, str]]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            if not table_exists(cursor, "ner_document_state"):
                return {}
            cursor.execute(
                """
                SELECT document_id, content_hash, model_name, pipeline_version
                FROM ner_document_state
                """
            )
            return {
                row[0]: (row[1].strip(), row[2], row[3])
                for row in cursor.fetchall()
            }


def load_documents(
    campaign_type: str,
    limit: int,
    document_ids: list[int],
) -> list[Document]:
    filters = [
        "COALESCE(NULLIF(BTRIM(d.raw_text), ''), "
        "NULLIF(BTRIM(d.summary_text), ''), "
        "NULLIF(BTRIM(d.page_title), '')) IS NOT NULL"
    ]
    parameters: list[Any] = []
    if campaign_type:
        filters.append("d.campaign_type_code = %s")
        parameters.append(campaign_type)
    if document_ids:
        filters.append("d.id = ANY(%s)")
        parameters.append(document_ids)
    parameters.append(limit)

    query = f"""
        SELECT
            d.id,
            b.bank_name,
            COALESCE(d.page_title, ''),
            COALESCE(d.source_url, ''),
            COALESCE(d.campaign_type_code, ''),
            COALESCE(
                NULLIF(BTRIM(d.raw_text), ''),
                NULLIF(BTRIM(d.summary_text), ''),
                NULLIF(BTRIM(d.page_title), ''),
                ''
            ),
            d.verified
        FROM documents d
        JOIN banks b ON b.id = d.bank_id
        WHERE {' AND '.join(filters)}
        ORDER BY d.id
        LIMIT %s
    """
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            return [Document(*row) for row in cursor.fetchall()]


def split_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            minimum_end = start + int(chunk_chars * 0.65)
            candidates = [
                text.rfind("\n", minimum_end, hard_end),
                text.rfind(". ", minimum_end, hard_end),
                text.rfind("; ", minimum_end, hard_end),
            ]
            breakpoint = max(candidates)
            if breakpoint >= minimum_end:
                end = breakpoint + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)
        whitespace = text.find(" ", next_start, min(end + 1, len(text)))
        start = whitespace + 1 if whitespace != -1 else next_start
    return chunks


def evidence_window(text: str, start: int, end: int) -> str:
    left_floor = max(0, start - 180)
    left_candidates = [
        text.rfind(marker, left_floor, start)
        for marker in (". ", "! ", "? ", "; ", "\n")
    ]
    left = max(left_candidates)
    left = left + 2 if left >= 0 else left_floor

    right_ceiling = min(len(text), end + 180)
    right_candidates = [
        position
        for marker in (". ", "! ", "? ", "; ", "\n")
        for position in [text.find(marker, end, right_ceiling)]
        if position >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else right_ceiling
    raw_evidence = text[left:right]
    if len(raw_evidence) > 360:
        local_entity_start = start - left
        local_entity_end = end - left
        local_start = max(0, local_entity_start - 150)
        local_end = min(len(raw_evidence), local_entity_end + 150)
        raw_evidence = raw_evidence[local_start:local_end]
    return re.sub(r"\s+", " ", raw_evidence).strip()


def has_amount(text: str) -> bool:
    return bool(
        re.search(
            r"\d[\d., ]*\s*(?:tl|try|turk lirasi|₺|usd|eur)\b",
            fold_text(text),
        )
    )


def has_percent(text: str) -> bool:
    folded = fold_text(text)
    return "%" in folded or "yuzde" in folded


def has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def has_suspicious_characters(text: str) -> bool:
    return any(
        character == "�"
        or unicodedata.category(character) in {"Co", "Cs"}
        or character in {"²", "³"}
        for character in text
    )


def context_rule_pass(label: str, evidence: str, value: str = "") -> bool:
    folded = fold_text(evidence)
    amount = has_amount(evidence)
    percent = has_percent(evidence)

    if label == "FINANSMAN_TUTARI":
        return amount and any(
            cue in folded
            for cue in (
                "finansman",
                "inansman tutar",
                "kredi tutari",
                "kullanim tutari",
                "finansman limiti",
                "kredi limiti",
                "tutari",
            )
        )
    if label == "VADE_SURESI":
        duration_range = bool(
            re.search(
                r"\b\d{1,3}\s*(?:-|\u2013|\u2014)\s*"
                r"\d{1,3}\s*(?:ay|yil|gun)\b",
                folded,
            )
        )
        return duration_range or (
            "vade" in folded or "geri odeme suresi" in folded
        ) and bool(
            re.search(
                r"\d+\s*(?:ay|yil|gun)(?:a|i|e|dir|lik|lik)?\b",
                folded,
            )
        )
    if label == "TAKSIT_SAYISI":
        return "taksit" in folded and has_number(folded)
    if label == "KAR_PAYI_ORANI":
        return (
            "kar payi" in folded
            or "kar orani" in folded
            or "finansman orani" in folded
        ) and percent
    if label == "KAR_PAYLASIM_ORANI":
        return "paylasim" in folded and (percent or has_number(folded))
    if label == "TAHSIS_UCRETI":
        return "tahsis" in folded and amount
    if label == "EKSPERTIZ_UCRETI":
        return (
            "ekspertiz" in folded or "degerleme" in folded
        ) and amount
    if label == "IPOTEK_TESIS_UCRETI":
        return "ipotek" in folded and "tesis" in folded and amount
    if label == "HARCAMA_ESIGI":
        roles = campaign_amount_roles(value, evidence) if value else set()
        return (
            "harcama" in folded or "alisveris" in folded
        ) and amount and (
            "spend_threshold" in roles
            if value
            else any(
                cue in folded
                for cue in (
                    "en az",
                    "minimum",
                    "alt limit",
                    "esik",
                    "ve uzeri",
                )
            )
        )
    if label == "HARCAMA_UST_LIMITI":
        return (
            "harcama" in folded or "alisveris" in folded
        ) and amount and any(
            cue in folded
            for cue in ("en fazla", "azami", "ust limit", "tavan", "kadar")
        )
    if label == "INDIRIM_ORANI":
        return ("indirim" in folded or "iade" in folded) and percent
    if label == "INDIRIM_TUTARI":
        return "indirim" in folded and amount
    if label == "ODUL_TUTARI":
        roles = campaign_amount_roles(value, evidence) if value else set()
        return amount and (
            "reward_amount" in roles
            if value
            else any(
                cue in folded
                for cue in ("odul", "iade", "bonus", "worldpuan")
            )
        )
    if label == "KAMPANYA_TARIH_ARALIGI":
        return campaign_date_context_pass(evidence)
    if label == "MINIMUM_BAKIYE":
        return (
            "bakiye" in folded or "hesap acilis" in folded
        ) and amount and any(
            cue in folded for cue in ("minimum", "asgari", "en az", "alt")
        )
    if label in {"ISLEM_ALT_LIMITI", "ISLEM_UST_LIMITI"}:
        operation_context = any(
            cue in folded
            for cue in ("islem", "eft", "fast", "havale", "transfer", "atm")
        )
        if label == "ISLEM_ALT_LIMITI":
            limit_context = any(
                cue in folded for cue in ("minimum", "asgari", "en az", "alt limit")
            )
        else:
            limit_context = any(
                cue in folded for cue in ("maksimum", "azami", "en fazla", "ust limit")
            )
        return operation_context and limit_context and amount
    return False


def parse_turkish_number(raw_value: str) -> float | int | None:
    value = re.sub(r"[^0-9.,-]", "", raw_value)
    if not value or value == "-":
        return None

    if "," in value and "." in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    elif "," in value:
        decimals = len(value) - value.rfind(",") - 1
        value = value.replace(",", "" if decimals == 3 else ".")
    elif "." in value:
        decimals = len(value) - value.rfind(".") - 1
        if decimals == 3:
            value = value.replace(".", "")

    try:
        number = float(value)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def normalize_entity(label: str, value: str) -> dict[str, Any] | None:
    number = parse_turkish_number(value)
    if label in {
        "FINANSMAN_TUTARI",
        "HARCAMA_ESIGI",
        "HARCAMA_UST_LIMITI",
        "INDIRIM_TUTARI",
        "ISLEM_ALT_LIMITI",
        "ISLEM_UST_LIMITI",
        "MINIMUM_BAKIYE",
        "ODUL_TUTARI",
        "TAHSIS_UCRETI",
        "EKSPERTIZ_UCRETI",
        "IPOTEK_TESIS_UCRETI",
    }:
        if number is None:
            return None
        folded = fold_text(value)
        currency = (
            "USD" if "usd" in folded or "dolar" in folded
            else "EUR" if "eur" in folded or "euro" in folded or "avro" in folded
            else "TRY"
        )
        return {"value": number, "currency": currency}
    if label in {"KAR_PAYI_ORANI", "KAR_PAYLASIM_ORANI", "INDIRIM_ORANI"}:
        return {"value": number, "unit": "percent"} if number is not None else None
    if label == "VADE_SURESI":
        folded = fold_text(value)
        unit = "month" if "ay" in folded else "year" if "yil" in folded else "day"
        return {"value": number, "unit": unit} if number is not None else None
    if label == "TAKSIT_SAYISI":
        return {"value": number, "unit": "count"} if number is not None else None
    if label == "KAMPANYA_TARIH_ARALIGI":
        return {"raw": value}
    return None


def decide_candidate(
    label: str,
    value: str,
    evidence: str,
    confidence: float,
    source_chunk: int,
    review_threshold: float,
) -> Candidate:
    rule_passed = context_rule_pass(label, evidence, value)
    auto_threshold = AUTO_THRESHOLDS.get(label, 0.95)
    folded_evidence = fold_text(evidence)
    fact_in_evidence = fold_text(value) in folded_evidence
    surface_error = validate_entity_surface(label, value)
    context_exclusion = excluded_context_reason(label, evidence, value)

    if not fact_in_evidence:
        decision = "rejected"
        reason = "fact_not_in_evidence"
    elif label in NUMERIC_LABELS and not has_number(value):
        decision = "rejected"
        reason = "entity_missing_number"
    elif has_suspicious_characters(value):
        decision = "review"
        reason = "suspicious_text_encoding"
    elif surface_error is not None:
        decision = "rejected"
        reason = surface_error
    elif context_exclusion is not None:
        decision = "rejected"
        reason = context_exclusion
    elif rule_passed and confidence >= auto_threshold:
        decision = "accepted"
        reason = "rule_and_confidence_passed"
    elif confidence >= (
        0.50
        if label in {
            "TAHSIS_UCRETI",
            "EKSPERTIZ_UCRETI",
            "IPOTEK_TESIS_UCRETI",
        }
        and rule_passed
        else review_threshold
    ):
        decision = "review"
        reason = (
            "below_auto_threshold"
            if rule_passed
            else "context_rule_failed"
        )
    else:
        decision = "rejected"
        reason = "below_review_threshold"

    return Candidate(
        fact_type=label,
        fact_text=value.strip(),
        normalized_value=normalize_entity(label, value),
        evidence_text=evidence,
        confidence=confidence,
        source_chunk=source_chunk,
        decision=decision,
        reason=reason,
    )


def call_ner(client: httpx.Client, text: str) -> tuple[str, list[dict]]:
    response = client.post(
        "/ner",
        json={
            "text": text,
            "threshold": API_PREDICTION_THRESHOLD,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return payload["model"], payload["entities"]


def call_classifier(
    client: httpx.Client,
    document: Document,
    threshold: float,
) -> dict[str, Any]:
    classification_text = "\n".join(
        part
        for part in (document.page_title, document.raw_text)
        if part.strip()
    )[:10000]
    response = client.post(
        "/classify",
        json={
            "text": classification_text,
            "page_title": document.page_title,
            "source_url": document.source_url,
            "threshold": threshold,
        },
    )
    response.raise_for_status()
    return response.json()


def deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    best: dict[tuple[str, str], Candidate] = {}
    decision_rank = {"accepted": 2, "review": 1, "rejected": 0}
    for candidate in candidates:
        key = (
            candidate.fact_type,
            fold_text(candidate.fact_text),
        )
        current = best.get(key)
        if current is None or (
            decision_rank[candidate.decision],
            candidate.confidence,
        ) > (
            decision_rank[current.decision],
            current.confidence,
        ):
            best[key] = candidate
    return list(best.values())


def save_document(
    document: Document,
    digest: str,
    model_name: str,
    candidates: list[Candidate],
    classification: dict[str, Any],
) -> None:
    accepted = [item for item in candidates if item.decision == "accepted"]
    review = [item for item in candidates if item.decision == "review"]
    rejected = [item for item in candidates if item.decision == "rejected"]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(STATE_TABLE_SQL)
            cursor.execute(REVIEW_TABLE_SQL)
            cursor.execute(REVIEW_INDEX_SQL)
            if not document.verified:
                product_type = str(classification["product_type"]["label"])
                cursor.execute(
                    """
                    UPDATE documents
                    SET campaign_type_code = %s,
                        campaign_type = %s,
                        confidence = %s,
                        label_source = %s,
                        rationale = %s,
                        auto_accepted = TRUE,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        product_type,
                        PRODUCT_TITLES.get(product_type, product_type),
                        float(classification["product_type"]["score"]),
                        PIPELINE_VERSION,
                        json.dumps(
                            {
                                "decision": classification["decision"],
                                "decision_basis": classification["decision_basis"],
                                "strong_rule": classification["strong_rule"],
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                        document.document_id,
                    ),
                )
            cursor.execute(
                """
                DELETE FROM comparison_facts
                WHERE document_id = %s AND extraction_method = %s
                """,
                (document.document_id, PIPELINE_VERSION),
            )
            cursor.execute(
                """
                DELETE FROM comparison_fact_review_queue
                WHERE document_id = %s
                  AND extraction_method = %s
                  AND review_status = 'pending'
                """,
                (document.document_id, PIPELINE_VERSION),
            )

            for candidate in accepted:
                key = make_fact_key(document.document_id, candidate)
                base = (
                    document.document_id,
                    candidate.fact_type,
                    candidate.fact_text,
                    Jsonb(candidate.normalized_value)
                    if candidate.normalized_value is not None
                    else None,
                    candidate.evidence_text,
                    PIPELINE_VERSION,
                    candidate.confidence,
                    candidate.source_chunk,
                    key,
                )
                duplicate_checks = (
                    document.document_id,
                    candidate.fact_type,
                    candidate.fact_text,
                    document.document_id,
                    candidate.fact_type,
                    candidate.fact_text,
                )
                cursor.execute(FACT_UPSERT_SQL, base + duplicate_checks)

            for candidate in review:
                cursor.execute(
                    REVIEW_UPSERT_SQL,
                    (
                        document.document_id,
                        candidate.fact_type,
                        candidate.fact_text,
                        Jsonb(candidate.normalized_value)
                        if candidate.normalized_value is not None
                        else None,
                        candidate.evidence_text,
                        PIPELINE_VERSION,
                        candidate.confidence,
                        candidate.source_chunk,
                        make_fact_key(document.document_id, candidate),
                        candidate.reason,
                    ),
                )

            cursor.execute(
                STATE_UPSERT_SQL,
                (
                    document.document_id,
                    digest,
                    model_name,
                    PIPELINE_VERSION,
                    len(accepted),
                    len(review),
                    len(rejected),
                ),
            )


def main() -> None:
    load_dotenv()
    args = parse_args()
    documents = load_documents(
        args.campaign_type,
        args.limit,
        args.document_id,
    )
    states = load_state()
    selected = []
    skipped = 0
    for document in documents:
        digest = content_hash(document.raw_text)
        state = states.get(document.document_id)
        unchanged = (
            state is not None
            and state[0] == digest
            and state[2] == PIPELINE_VERSION
        )
        if unchanged and not args.force:
            skipped += 1
            continue
        selected.append((document, digest))

    print("Pipeline:", PIPELINE_VERSION)
    print("Mode:", "DATABASE_WRITE" if args.write else "DRY_RUN")
    print("Selected documents:", len(selected))
    print("Unchanged documents skipped:", skipped)
    if not selected:
        print("Nothing to process.")
        return

    totals = {
        "accepted": 0,
        "review": 0,
        "rejected": 0,
        "failed": 0,
        "classification_review": 0,
        "ner_skipped": 0,
        "filtered_out": 0,
    }
    timeout = httpx.Timeout(120.0, connect=10.0)
    with httpx.Client(base_url=args.api_url, timeout=timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        health_payload = health.json()
        if not health_payload.get("ner_model_ready"):
            raise RuntimeError("NER model is not ready in the API.")
        if not health_payload.get("classifier_ready"):
            raise RuntimeError("Classification models are not ready in the API.")

        for index, (document, digest) in enumerate(selected, start=1):
            try:
                classification = call_classifier(
                    client,
                    document,
                    args.classification_threshold,
                )
            except Exception as error:
                print(
                    f"[{index}/{len(selected)}] document={document.document_id} "
                    f"classification ERROR: {error}"
                )
                totals["failed"] += 1
                continue

            product_type = str(classification["product_type"]["label"])
            product_score = float(classification["product_type"]["score"])
            classification_decision = str(classification["decision"])
            allowed_labels = ALLOWED_ENTITY_LABELS_BY_PRODUCT.get(product_type)
            print(
                f"[{index}/{len(selected)}] document={document.document_id} "
                f"old_type={document.campaign_type_code or 'NULL'} "
                f"classified={product_type} score={product_score:.4f} "
                f"decision={classification_decision}"
            )

            if classification_decision == "REVIEW":
                totals["classification_review"] += 1
                totals["ner_skipped"] += 1
                reasons = ",".join(classification.get("review_reasons", []))
                print(
                    "  NER_SKIPPED | classification_requires_review "
                    f"| {reasons or 'unspecified'}"
                )
                continue

            if not allowed_labels:
                totals["ner_skipped"] += 1
                print("  NER_SKIPPED | product_type_not_extraction_eligible")
                if args.write:
                    save_document(
                        document=document,
                        digest=digest,
                        model_name=str(
                            health_payload.get("ner_model", "ner_skipped")
                        ),
                        candidates=[],
                        classification=classification,
                    )
                continue

            chunks = split_text(
                document.raw_text,
                args.chunk_chars,
                args.chunk_overlap,
            )
            candidates = []
            model_name = "unknown"
            failed = False
            for chunk_index, chunk in enumerate(chunks, start=1):
                try:
                    model_name, entities = call_ner(client, chunk)
                except Exception as error:
                    print(
                        f"[{index}/{len(selected)}] document={document.document_id} "
                        f"chunk={chunk_index} ERROR: {error}"
                    )
                    failed = True
                    break

                for entity in entities:
                    if str(entity["label"]) not in allowed_labels:
                        totals["filtered_out"] += 1
                        continue
                    evidence = evidence_window(
                        chunk,
                        int(entity["start"]),
                        int(entity["end"]),
                    )
                    candidates.append(
                        decide_candidate(
                            label=str(entity["label"]),
                            value=str(entity["text"]),
                            evidence=evidence,
                            confidence=float(entity["score"]),
                            source_chunk=chunk_index,
                            review_threshold=args.review_threshold,
                        )
                    )

            if failed:
                totals["failed"] += 1
                continue

            candidates = deduplicate_candidates(candidates)
            counts = {
                decision: sum(
                    item.decision == decision for item in candidates
                )
                for decision in ("accepted", "review", "rejected")
            }
            for decision, count in counts.items():
                totals[decision] += count

            print(
                f"  NER_RESULT bank={document.bank_name} chunks={len(chunks)} "
                f"accepted={counts['accepted']} review={counts['review']} "
                f"rejected={counts['rejected']}"
            )
            for candidate in candidates:
                print(
                    f"  {candidate.decision.upper():8} "
                    f"{candidate.fact_type:28} "
                    f"{candidate.confidence:.4f} | {candidate.fact_text} "
                    f"| {candidate.reason}"
                )
                if args.show_evidence:
                    print(f"    EVIDENCE | {candidate.evidence_text}")

            if args.write:
                save_document(
                    document=document,
                    digest=digest,
                    model_name=model_name,
                    candidates=candidates,
                    classification=classification,
                )

    print("Summary:", json.dumps(totals, ensure_ascii=False, sort_keys=True))
    print("Completed at:", datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
