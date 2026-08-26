from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from psycopg.types.json import Jsonb

from classifier_service import classify_text
from fact_surface_rules import validate_entity_surface
from hybrid_search import MODEL_NAME, get_connection
from ner_service import predict_entities


PIPELINE_VERSION = "classifier_v2_ner_v4_intake_v2_1"
FACT_EXTRACTION_METHOD = "ner_v4_rules_v2_1"
DEFAULT_NER_CHUNK_CHARS = 900
DEFAULT_NER_CHUNK_OVERLAP = 100
DEFAULT_EMBED_MAX_TOKENS = 384
DEFAULT_EMBED_OVERLAP_TOKENS = 64


FINANCE_ENTITY_LABELS = {
    "FINANSMAN_TUTARI",
    "VADE_SURESI",
    "KAR_PAYI_ORANI",
    "TAHSIS_UCRETI",
}
CAMPAIGN_ENTITY_LABELS = {
    "HARCAMA_ESIGI",
    "HARCAMA_UST_LIMITI",
    "INDIRIM_ORANI",
    "INDIRIM_TUTARI",
    "ODUL_TUTARI",
    "TAKSIT_SAYISI",
    "KAMPANYA_TARIH_ARALIGI",
    "ISLEM_ALT_LIMITI",
    "ISLEM_UST_LIMITI",
}
ALLOWED_ENTITY_LABELS_BY_PRODUCT = {
    "KONUT_FINANSMANI": FINANCE_ENTITY_LABELS
    | {"EKSPERTIZ_UCRETI", "IPOTEK_TESIS_UCRETI"},
    "TASIT_FINANSMANI": FINANCE_ENTITY_LABELS,
    "IHTIYAC_FINANSMANI": FINANCE_ENTITY_LABELS,
    "TICARI_FINANSMAN": FINANCE_ENTITY_LABELS
    | {"ISLEM_ALT_LIMITI", "ISLEM_UST_LIMITI"},
    "DIGER_FINANSMAN": FINANCE_ENTITY_LABELS,
    "KART_KAMPANYASI": CAMPAIGN_ENTITY_LABELS,
    "DIGER_KAMPANYA": CAMPAIGN_ENTITY_LABELS
    | FINANCE_ENTITY_LABELS,
    "KART_URUNU": {
        "HARCAMA_ESIGI",
        "HARCAMA_UST_LIMITI",
        "TAKSIT_SAYISI",
        "ISLEM_ALT_LIMITI",
        "ISLEM_UST_LIMITI",
    },
    "KATILMA_HESABI": {
        "MINIMUM_BAKIYE",
        "KAR_PAYLASIM_ORANI",
        "VADE_SURESI",
    },
    "YATIRIM_URUNU": {
        "MINIMUM_BAKIYE",
        "KAR_PAYLASIM_ORANI",
        "ISLEM_ALT_LIMITI",
        "ISLEM_UST_LIMITI",
    },
    "ODEME_TRANSFER_HIZMETI": {
        "ISLEM_ALT_LIMITI",
        "ISLEM_UST_LIMITI",
    },
}


PRODUCT_TITLES = {
    "DIGER": "Diger",
    "DIGER_FINANSMAN": "Diger Finansman",
    "DIGER_KAMPANYA": "Diger Kampanya",
    "IHTIYAC_FINANSMANI": "Ihtiyac Finansmani",
    "KART_KAMPANYASI": "Kart Kampanyasi",
    "KART_URUNU": "Kart Urunu",
    "KATILMA_HESABI": "Katilma Hesabi",
    "KONUT_FINANSMANI": "Konut Finansmani",
    "ODEME_TRANSFER_HIZMETI": "Odeme ve Transfer Hizmeti",
    "SIGORTA_TEKAFUL_URUNU": "Sigorta ve Tekaful Urunu",
    "TASIT_FINANSMANI": "Tasit Finansmani",
    "TICARI_FINANSMAN": "Ticari Finansman",
    "YATIRIM_URUNU": "Yatirim Urunu",
}


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


AUTO_THRESHOLDS = {
    "EKSPERTIZ_UCRETI": 0.55,
    "FINANSMAN_TUTARI": 0.88,
    "HARCAMA_ESIGI": 0.85,
    "HARCAMA_UST_LIMITI": 0.85,
    "INDIRIM_ORANI": 0.90,
    "INDIRIM_TUTARI": 0.90,
    "IPOTEK_TESIS_UCRETI": 0.55,
    "ISLEM_ALT_LIMITI": 0.85,
    "ISLEM_UST_LIMITI": 0.85,
    "KAMPANYA_TARIH_ARALIGI": 0.90,
    "KAR_PAYI_ORANI": 0.85,
    "KAR_PAYLASIM_ORANI": 0.95,
    "MINIMUM_BAKIYE": 0.85,
    "ODUL_TUTARI": 0.85,
    "TAHSIS_UCRETI": 0.55,
    "TAKSIT_SAYISI": 0.90,
    "VADE_SURESI": 0.85,
}


DOCUMENT_REVIEW_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_intake_review_queue (
    id BIGSERIAL PRIMARY KEY,
    record_key VARCHAR(255) NOT NULL,
    bank_key VARCHAR(128) NOT NULL,
    bank_name VARCHAR(255) NOT NULL,
    source_url TEXT NOT NULL,
    page_title TEXT,
    raw_text TEXT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    classification JSONB NOT NULL,
    review_reason VARCHAR(128) NOT NULL,
    review_status VARCHAR(16) NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (record_key, content_hash)
)
"""


DOCUMENT_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_intake_state (
    document_id BIGINT PRIMARY KEY
        REFERENCES documents(id) ON DELETE CASCADE,
    bank_id BIGINT NOT NULL
        REFERENCES banks(id) ON DELETE CASCADE,
    record_key VARCHAR(255) NOT NULL,
    content_hash CHAR(64) NOT NULL,
    pipeline_version VARCHAR(64) NOT NULL,
    classification JSONB NOT NULL,
    accepted_fact_count INTEGER NOT NULL DEFAULT 0,
    review_fact_count INTEGER NOT NULL DEFAULT 0,
    rejected_fact_count INTEGER NOT NULL DEFAULT 0,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (bank_id, content_hash)
)
"""


FACT_REVIEW_TABLE_SQL = """
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


@dataclass
class FactCandidate:
    label: str
    text: str
    normalized_value: dict[str, Any] | None
    evidence_text: str
    confidence: float
    source_chunk: int
    decision: str
    reason: str


@dataclass
class EmbeddingChunk:
    chunk_index: int
    content: str
    token_count: int
    content_hash: str


def fold_text(value: str) -> str:
    translated = value.translate(str.maketrans({"ı": "i", "İ": "I"}))
    decomposed = unicodedata.normalize("NFKD", translated)
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip().casefold()


def normalized_content(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def content_digest(value: str) -> str:
    return hashlib.sha256(
        normalized_content(value).encode("utf-8")
    ).hexdigest()


def default_record_key(bank_key: str, source_url: str) -> str:
    stable_value = f"{fold_text(bank_key)}|{source_url.strip().casefold()}"
    digest = hashlib.sha256(stable_value.encode("utf-8")).hexdigest()[:24]
    safe_bank = re.sub(r"[^a-z0-9]+", "-", fold_text(bank_key)).strip("-")
    return f"live-{safe_bank or 'bank'}-{digest}"


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
            breakpoints = [
                text.rfind("\n", minimum_end, hard_end),
                text.rfind(". ", minimum_end, hard_end),
                text.rfind("; ", minimum_end, hard_end),
            ]
            best_breakpoint = max(breakpoints)
            if best_breakpoint >= minimum_end:
                end = best_breakpoint + 1

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
    return normalized_content(text[left:right])


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


def context_rule_pass(label: str, evidence: str) -> bool:
    folded = fold_text(evidence)
    amount = has_amount(evidence)
    percent = has_percent(evidence)

    if label == "FINANSMAN_TUTARI":
        return amount and any(
            cue in folded
            for cue in (
                "finansman",
                "kredi tutari",
                "kullanim tutari",
                "finansman limiti",
                "kredi limiti",
                "tutari",
            )
        )
    if label == "VADE_SURESI":
        return (
            "vade" in folded or "geri odeme suresi" in folded
        ) and bool(
            re.search(
                r"\d+\s*(?:ay|yil|gun)(?:a|i|e|dir|lik)?\b",
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
        return ("ekspertiz" in folded or "degerleme" in folded) and amount
    if label == "IPOTEK_TESIS_UCRETI":
        return "ipotek" in folded and "tesis" in folded and amount
    if label == "HARCAMA_ESIGI":
        return (
            "harcama" in folded or "alisveris" in folded
        ) and amount and any(
            cue in folded
            for cue in ("en az", "minimum", "alt limit", "esik", "ve uzeri")
        )
    if label == "HARCAMA_UST_LIMITI":
        return (
            "harcama" in folded or "alisveris" in folded
        ) and amount and any(
            cue in folded
            for cue in ("en fazla", "azami", "ust limit", "tavan", "kadar")
        )
    if label == "INDIRIM_ORANI":
        return "indirim" in folded and percent
    if label == "INDIRIM_TUTARI":
        return "indirim" in folded and amount
    if label == "ODUL_TUTARI":
        return any(cue in folded for cue in ("odul", "iade", "bonus")) and amount
    if label == "KAMPANYA_TARIH_ARALIGI":
        return any(
            cue in folded for cue in ("kampanya", "tarih", "gecerli")
        ) and has_number(folded)
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
            "USD"
            if "usd" in folded or "dolar" in folded
            else "EUR"
            if "eur" in folded or "euro" in folded or "avro" in folded
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
) -> FactCandidate:
    rule_passed = context_rule_pass(label, evidence)
    auto_threshold = AUTO_THRESHOLDS.get(label, 0.95)
    folded_evidence = fold_text(evidence)
    fact_in_evidence = fold_text(value) in folded_evidence
    surface_error = validate_entity_surface(label, value)
    early_payment_context = label == "VADE_SURESI" and any(
        cue in folded_evidence
        for cue in ("erken odeme", "tazminat", "kalan vade")
    )
    first_payment_context = label == "VADE_SURESI" and any(
        cue in folded_evidence
        for cue in ("ilk taksit", "ilk odeme", "odeme tarihi")
    )
    conflicting_finance_context = (
        label == "FINANSMAN_TUTARI"
        and any(
            cue in folded_evidence
            for cue in (
                "deger",
                "bedel",
                "fiyat",
                "rayic",
                "satis tutari",
                "fatura tutari",
            )
        )
    )

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
    elif early_payment_context:
        decision = "rejected"
        reason = "excluded_early_payment_context"
    elif first_payment_context:
        decision = "rejected"
        reason = "excluded_first_payment_timing_context"
    elif conflicting_finance_context:
        decision = "rejected"
        reason = "conflicting_property_value_context"
    elif rule_passed and confidence >= auto_threshold:
        decision = "accepted"
        reason = "rule_and_confidence_passed"
    elif confidence >= (
        0.50
        if label
        in {"TAHSIS_UCRETI", "EKSPERTIZ_UCRETI", "IPOTEK_TESIS_UCRETI"}
        and rule_passed
        else review_threshold
    ):
        decision = "review"
        reason = "below_auto_threshold" if rule_passed else "context_rule_failed"
    else:
        decision = "rejected"
        reason = "below_review_threshold"

    return FactCandidate(
        label=label,
        text=value.strip(),
        normalized_value=normalize_entity(label, value),
        evidence_text=evidence,
        confidence=confidence,
        source_chunk=source_chunk,
        decision=decision,
        reason=reason,
    )


def deduplicate_candidates(candidates: list[FactCandidate]) -> list[FactCandidate]:
    best: dict[tuple[str, str], FactCandidate] = {}
    decision_rank = {"accepted": 2, "review": 1, "rejected": 0}
    for candidate in candidates:
        key = (candidate.label, fold_text(candidate.text))
        current = best.get(key)
        if current is None or (
            decision_rank[candidate.decision],
            candidate.confidence,
        ) > (
            decision_rank[current.decision],
            current.confidence,
        ):
            best[key] = candidate
    return sorted(
        best.values(),
        key=lambda item: (item.source_chunk, item.label, item.text),
    )


def candidate_to_dict(candidate: FactCandidate) -> dict[str, Any]:
    return asdict(candidate)


def analyze_intake(
    *,
    page_title: str,
    raw_text: str,
    classifier_bundle,
    classifier_lock,
    ner_bundle,
    ner_lock,
    classification_threshold: float,
    ner_threshold: float,
    review_threshold: float,
) -> dict[str, Any]:
    classification_input = normalized_content(
        "\n".join(part for part in (page_title, raw_text) if part)
    )[:10000]
    with classifier_lock:
        classification = classify_text(
            text=classification_input,
            bundle=classifier_bundle,
            threshold=classification_threshold,
            page_title=page_title,
        )

    return analyze_classified_intake(
        classification=classification,
        raw_text=raw_text,
        ner_bundle=ner_bundle,
        ner_lock=ner_lock,
        ner_threshold=ner_threshold,
        review_threshold=review_threshold,
    )


def analyze_reviewed_intake(
    *,
    original_classification: dict[str, Any],
    product_type: str,
    raw_text: str,
    ner_bundle,
    ner_lock,
    ner_threshold: float,
    review_threshold: float,
) -> dict[str, Any]:
    classification = json.loads(json.dumps(original_classification))
    original_top3 = classification.get("product_top3", [])
    classification["product_type"] = {
        "label": product_type,
        "score": 1.0,
    }
    classification["product_top3"] = [
        {"label": product_type, "score": 1.0},
        *[
            item
            for item in original_top3
            if item.get("label") != product_type
        ][:2],
    ]
    classification["decision"] = "ACCEPTED"
    classification["decision_basis"] = "human_review"
    classification["review_reasons"] = []
    return analyze_classified_intake(
        classification=classification,
        raw_text=raw_text,
        ner_bundle=ner_bundle,
        ner_lock=ner_lock,
        ner_threshold=ner_threshold,
        review_threshold=review_threshold,
    )


def analyze_classified_intake(
    *,
    classification: dict[str, Any],
    raw_text: str,
    ner_bundle,
    ner_lock,
    ner_threshold: float,
    review_threshold: float,
) -> dict[str, Any]:

    product_type = str(classification["product_type"]["label"])
    allowed_labels = ALLOWED_ENTITY_LABELS_BY_PRODUCT.get(product_type)
    if classification["decision"] == "REVIEW":
        return {
            "status": "REVIEW",
            "classification": classification,
            "ner": {
                "executed": False,
                "skip_reason": "classification_requires_review",
                "model": ner_bundle.model_dir.name,
                "raw_count": 0,
                "filtered_out_count": 0,
                "allowed_labels": sorted(allowed_labels or []),
                "accepted_count": 0,
                "review_count": 0,
                "rejected_count": 0,
                "candidates": [],
            },
        }

    if not allowed_labels:
        return {
            "status": "ACCEPTED",
            "classification": classification,
            "ner": {
                "executed": False,
                "skip_reason": "product_type_not_extraction_eligible",
                "model": ner_bundle.model_dir.name,
                "raw_count": 0,
                "filtered_out_count": 0,
                "allowed_labels": [],
                "accepted_count": 0,
                "review_count": 0,
                "rejected_count": 0,
                "candidates": [],
            },
        }

    chunks = split_text(
        raw_text,
        DEFAULT_NER_CHUNK_CHARS,
        DEFAULT_NER_CHUNK_OVERLAP,
    )
    candidates: list[FactCandidate] = []
    raw_count = 0
    filtered_out_count = 0
    with ner_lock:
        for chunk_index, chunk in enumerate(chunks):
            raw_entities = predict_entities(
                text=chunk,
                bundle=ner_bundle,
                threshold=ner_threshold,
            )
            raw_count += len(raw_entities)
            for entity in raw_entities:
                label = str(entity["label"])
                if label not in allowed_labels:
                    filtered_out_count += 1
                    continue
                evidence = evidence_window(
                    chunk,
                    int(entity["start"]),
                    int(entity["end"]),
                )
                candidates.append(
                    decide_candidate(
                        label=label,
                        value=str(entity["text"]),
                        evidence=evidence,
                        confidence=float(entity["score"]),
                        source_chunk=chunk_index,
                        review_threshold=review_threshold,
                    )
                )

    candidates = deduplicate_candidates(candidates)
    counts = {
        decision: sum(item.decision == decision for item in candidates)
        for decision in ("accepted", "review", "rejected")
    }
    return {
        "status": "ACCEPTED",
        "classification": classification,
        "ner": {
            "executed": True,
            "skip_reason": None,
            "model": ner_bundle.model_dir.name,
            "raw_count": raw_count,
            "filtered_out_count": filtered_out_count,
            "allowed_labels": sorted(allowed_labels),
            "accepted_count": counts["accepted"],
            "review_count": counts["review"],
            "rejected_count": counts["rejected"],
            "candidates": [candidate_to_dict(item) for item in candidates],
        },
    }


def build_embedding_chunks(
    *,
    bank_name: str,
    page_title: str,
    raw_text: str,
    tokenizer,
    max_tokens: int = DEFAULT_EMBED_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_EMBED_OVERLAP_TOKENS,
) -> list[EmbeddingChunk]:
    header_parts = []
    if bank_name.strip():
        header_parts.append(f"Banka: {bank_name.strip()}")
    if page_title.strip():
        header_parts.append(f"Baslik: {page_title.strip()}")
    header = "\n".join(header_parts)
    header_token_ids = tokenizer.encode(
        header,
        add_special_tokens=False,
    )[:64]
    header = tokenizer.decode(
        header_token_ids,
        skip_special_tokens=True,
    ).strip()

    body_token_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    body_capacity = max_tokens - len(header_token_ids)
    if body_capacity <= overlap_tokens:
        raise RuntimeError("Document header leaves no room for chunk content.")
    step = body_capacity - overlap_tokens
    chunks = []
    for chunk_index, start in enumerate(range(0, len(body_token_ids), step)):
        window = body_token_ids[start : start + body_capacity]
        if not window:
            break
        body = tokenizer.decode(window, skip_special_tokens=True).strip()
        content = "\n".join(part for part in (header, body) if part).strip()
        if content:
            chunks.append(
                EmbeddingChunk(
                    chunk_index=chunk_index,
                    content=content,
                    token_count=len(header_token_ids) + len(window),
                    content_hash=content_digest(content),
                )
            )
        if start + body_capacity >= len(body_token_ids):
            break
    if not chunks:
        raise RuntimeError("No embedding chunks were generated.")
    return chunks


def encode_embedding_chunks(model, model_lock, chunks: list[EmbeddingChunk]):
    contents = [chunk.content for chunk in chunks]
    with model_lock:
        embeddings = model.encode(
            contents,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise RuntimeError(
            f"Expected embedding shape (n, 1024), got {embeddings.shape}."
        )
    return embeddings


def make_fact_key(document_id: int, candidate: dict[str, Any]) -> str:
    value = "|".join(
        [
            str(document_id),
            FACT_EXTRACTION_METHOD,
            str(candidate["label"]),
            fold_text(str(candidate["text"])),
            fold_text(str(candidate["evidence_text"])),
        ]
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def queue_document_review(
    cursor,
    *,
    record_key: str,
    bank_key: str,
    bank_name: str,
    source_url: str,
    page_title: str,
    raw_text: str,
    digest: str,
    classification: dict[str, Any],
    reason: str,
) -> int:
    cursor.execute(DOCUMENT_REVIEW_TABLE_SQL)
    cursor.execute(
        """
        INSERT INTO document_intake_review_queue (
            record_key,
            bank_key,
            bank_name,
            source_url,
            page_title,
            raw_text,
            content_hash,
            classification,
            review_reason
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (record_key, content_hash) DO UPDATE SET
            bank_key = EXCLUDED.bank_key,
            bank_name = EXCLUDED.bank_name,
            source_url = EXCLUDED.source_url,
            page_title = EXCLUDED.page_title,
            raw_text = EXCLUDED.raw_text,
            classification = EXCLUDED.classification,
            review_reason = EXCLUDED.review_reason,
            updated_at = NOW()
        WHERE document_intake_review_queue.review_status = 'pending'
        RETURNING id
        """,
        (
            record_key,
            bank_key,
            bank_name,
            source_url,
            page_title or None,
            raw_text,
            digest,
            Jsonb(classification),
            reason,
        ),
    )
    row = cursor.fetchone()
    if row is not None:
        return int(row[0])
    cursor.execute(
        """
        SELECT id
        FROM document_intake_review_queue
        WHERE record_key = %s AND content_hash = %s
        """,
        (record_key, digest),
    )
    return int(cursor.fetchone()[0])


def inspect_existing_document(cursor, record_key: str, bank_key: str, digest: str):
    cursor.execute(
        """
        SELECT d.id, d.raw_text, b.bank_key
        FROM documents d
        JOIN banks b ON b.id = d.bank_id
        WHERE d.record_key = %s
        """,
        (record_key,),
    )
    existing = cursor.fetchone()
    if existing is not None:
        existing_digest = content_digest(existing[1])
        return {
            "kind": "same_record",
            "document_id": int(existing[0]),
            "same_content": existing_digest == digest,
            "bank_key": existing[2],
        }

    cursor.execute(
        """
        SELECT d.id, d.raw_text
        FROM documents d
        JOIN banks b ON b.id = d.bank_id
        WHERE b.bank_key = %s
        """,
        (bank_key,),
    )
    for document_id, raw_text in cursor.fetchall():
        if content_digest(raw_text) == digest:
            return {
                "kind": "duplicate_content",
                "document_id": int(document_id),
                "same_content": True,
                "bank_key": bank_key,
            }
    return None


def preflight_existing_action(
    record_key: str,
    bank_key: str,
    digest: str,
) -> dict[str, Any] | None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            existing = inspect_existing_document(
                cursor,
                record_key,
                bank_key,
                digest,
            )
    if existing is None:
        return None
    if existing["kind"] == "duplicate_content":
        return {
            "mode": "DATABASE_WRITE",
            "action": "duplicate_content_skipped",
            "document_id": existing["document_id"],
            "document_review_id": None,
            "chunks_written": 0,
            "facts_written": 0,
            "fact_reviews_queued": 0,
        }
    if existing["same_content"]:
        return {
            "mode": "DATABASE_WRITE",
            "action": "unchanged_skipped",
            "document_id": existing["document_id"],
            "document_review_id": None,
            "chunks_written": 0,
            "facts_written": 0,
            "fact_reviews_queued": 0,
        }
    return None


def insert_chunks(
    cursor,
    document_id: int,
    record_key: str,
    chunks: list[EmbeddingChunk],
    embeddings,
) -> None:
    cursor.execute(
        """
        SELECT is_generated
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'document_chunks'
          AND column_name = 'search_vector'
        """
    )
    row = cursor.fetchone()
    search_vector_is_generated = row is not None and row[0] != "NEVER"

    cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
    rows = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        rows.append(
            (
                document_id,
                chunk.chunk_index,
                chunk.content,
                chunk.token_count,
                chunk.content_hash,
                MODEL_NAME,
                embedding,
                Jsonb(
                    {
                        "source": PIPELINE_VERSION,
                        "record_key": record_key,
                    }
                ),
            )
        )
    cursor.executemany(
        """
        INSERT INTO document_chunks (
            document_id,
            chunk_index,
            content,
            token_count,
            content_hash,
            embedding_model,
            embedding,
            metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows,
    )
    if not search_vector_is_generated:
        cursor.execute(
            """
            UPDATE document_chunks
            SET search_vector =
                to_tsvector('simple', content)
                || to_tsvector('turkish', content)
            WHERE document_id = %s
            """,
            (document_id,),
        )


def insert_fact_candidates(cursor, document_id: int, candidates: list[dict[str, Any]]):
    cursor.execute(FACT_REVIEW_TABLE_SQL)
    cursor.execute(
        """
        DELETE FROM comparison_facts
        WHERE document_id = %s AND extraction_method = %s
        """,
        (document_id, FACT_EXTRACTION_METHOD),
    )
    cursor.execute(
        """
        DELETE FROM comparison_fact_review_queue
        WHERE document_id = %s
          AND extraction_method = %s
          AND review_status = 'pending'
        """,
        (document_id, FACT_EXTRACTION_METHOD),
    )

    facts_written = 0
    reviews_queued = 0
    for candidate in candidates:
        key = make_fact_key(document_id, candidate)
        if candidate["decision"] == "accepted":
            cursor.execute(
                """
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
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, fact_key) DO UPDATE SET
                    normalized_value = EXCLUDED.normalized_value,
                    evidence_text = EXCLUDED.evidence_text,
                    confidence = EXCLUDED.confidence,
                    source_chunk = EXCLUDED.source_chunk,
                    updated_at = NOW()
                """,
                (
                    document_id,
                    candidate["label"],
                    candidate["text"],
                    Jsonb(candidate["normalized_value"])
                    if candidate["normalized_value"] is not None
                    else None,
                    candidate["evidence_text"],
                    FACT_EXTRACTION_METHOD,
                    candidate["confidence"],
                    candidate["source_chunk"],
                    key,
                ),
            )
            facts_written += 1
        elif candidate["decision"] == "review":
            cursor.execute(
                """
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
                """,
                (
                    document_id,
                    candidate["label"],
                    candidate["text"],
                    Jsonb(candidate["normalized_value"])
                    if candidate["normalized_value"] is not None
                    else None,
                    candidate["evidence_text"],
                    FACT_EXTRACTION_METHOD,
                    candidate["confidence"],
                    candidate["source_chunk"],
                    key,
                    candidate["reason"],
                ),
            )
            reviews_queued += 1
    return facts_written, reviews_queued


def persist_intake(
    *,
    record_key: str,
    bank_key: str,
    bank_name: str,
    source_url: str,
    page_title: str,
    raw_text: str,
    digest: str,
    analysis: dict[str, Any],
    embedding_model,
    embedding_lock,
    allow_update: bool,
    human_verified: bool = False,
) -> dict[str, Any]:
    classification = analysis["classification"]
    if analysis["status"] == "REVIEW":
        reason = ",".join(classification["review_reasons"]) or "classification_review"
        with get_connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    review_id = queue_document_review(
                        cursor,
                        record_key=record_key,
                        bank_key=bank_key,
                        bank_name=bank_name,
                        source_url=source_url,
                        page_title=page_title,
                        raw_text=raw_text,
                        digest=digest,
                        classification=classification,
                        reason=reason,
                    )
        return {
            "mode": "DATABASE_WRITE",
            "action": "review_queued",
            "document_id": None,
            "document_review_id": review_id,
            "chunks_written": 0,
            "facts_written": 0,
            "fact_reviews_queued": 0,
        }

    with get_connection() as connection:
        with connection.cursor() as cursor:
            existing = inspect_existing_document(
                cursor,
                record_key,
                bank_key,
                digest,
            )
    if existing is not None:
        if existing["kind"] == "duplicate_content":
            return {
                "mode": "DATABASE_WRITE",
                "action": "duplicate_content_skipped",
                "document_id": existing["document_id"],
                "document_review_id": None,
                "chunks_written": 0,
                "facts_written": 0,
                "fact_reviews_queued": 0,
            }
        if existing["same_content"]:
            return {
                "mode": "DATABASE_WRITE",
                "action": "unchanged_skipped",
                "document_id": existing["document_id"],
                "document_review_id": None,
                "chunks_written": 0,
                "facts_written": 0,
                "fact_reviews_queued": 0,
            }
        if not allow_update:
            with get_connection() as connection:
                with connection.transaction():
                    with connection.cursor() as cursor:
                        review_id = queue_document_review(
                            cursor,
                            record_key=record_key,
                            bank_key=bank_key,
                            bank_name=bank_name,
                            source_url=source_url,
                            page_title=page_title,
                            raw_text=raw_text,
                            digest=digest,
                            classification=classification,
                            reason="existing_document_changed",
                        )
            return {
                "mode": "DATABASE_WRITE",
                "action": "changed_document_review_queued",
                "document_id": existing["document_id"],
                "document_review_id": review_id,
                "chunks_written": 0,
                "facts_written": 0,
                "fact_reviews_queued": 0,
            }

    chunks = build_embedding_chunks(
        bank_name=bank_name,
        page_title=page_title,
        raw_text=raw_text,
        tokenizer=embedding_model.tokenizer,
    )
    embeddings = encode_embedding_chunks(
        embedding_model,
        embedding_lock,
        chunks,
    )
    product_type = str(classification["product_type"]["label"])
    candidates = analysis["ner"]["candidates"]
    rationale = json.dumps(
        {
            "pipeline": PIPELINE_VERSION,
            "decision": classification["decision"],
            "decision_basis": classification["decision_basis"],
            "strong_rule": classification["strong_rule"],
        },
        ensure_ascii=True,
        sort_keys=True,
    )

    with get_connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(DOCUMENT_REVIEW_TABLE_SQL)
                cursor.execute(DOCUMENT_STATE_TABLE_SQL)
                cursor.execute(FACT_REVIEW_TABLE_SQL)
                cursor.execute(
                    "SELECT to_regclass('public.comparison_facts') IS NOT NULL"
                )
                if not cursor.fetchone()[0]:
                    raise RuntimeError("Table public.comparison_facts was not found.")

                cursor.execute(
                    """
                    INSERT INTO banks (bank_key, bank_name)
                    VALUES (%s, %s)
                    ON CONFLICT (bank_key) DO UPDATE SET
                        bank_name = EXCLUDED.bank_name
                    RETURNING id
                    """,
                    (bank_key, bank_name),
                )
                bank_id = int(cursor.fetchone()[0])

                cursor.execute(
                    """
                    INSERT INTO documents (
                        record_key,
                        bank_id,
                        source_url,
                        page_title,
                        raw_text,
                        summary_text,
                        campaign_type_code,
                        campaign_type,
                        confidence,
                        label_source,
                        rationale,
                        verified,
                        auto_accepted,
                        updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, NULL, %s, %s,
                        %s, %s, %s, %s, %s, NOW()
                    )
                    ON CONFLICT (record_key) DO UPDATE SET
                        bank_id = EXCLUDED.bank_id,
                        source_url = EXCLUDED.source_url,
                        page_title = EXCLUDED.page_title,
                        raw_text = EXCLUDED.raw_text,
                        campaign_type_code = EXCLUDED.campaign_type_code,
                        campaign_type = EXCLUDED.campaign_type,
                        confidence = EXCLUDED.confidence,
                        label_source = EXCLUDED.label_source,
                        rationale = EXCLUDED.rationale,
                        verified = EXCLUDED.verified,
                        auto_accepted = EXCLUDED.auto_accepted,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (
                        record_key,
                        bank_id,
                        source_url,
                        page_title or None,
                        raw_text,
                        product_type,
                        PRODUCT_TITLES.get(product_type, product_type),
                        float(classification["product_type"]["score"]),
                        PIPELINE_VERSION,
                        rationale,
                        human_verified,
                        not human_verified,
                    ),
                )
                document_id = int(cursor.fetchone()[0])
                insert_chunks(
                    cursor,
                    document_id,
                    record_key,
                    chunks,
                    embeddings,
                )
                facts_written, reviews_queued = insert_fact_candidates(
                    cursor,
                    document_id,
                    candidates,
                )
                cursor.execute(
                    """
                    INSERT INTO document_intake_state (
                        document_id,
                        bank_id,
                        record_key,
                        content_hash,
                        pipeline_version,
                        classification,
                        accepted_fact_count,
                        review_fact_count,
                        rejected_fact_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        bank_id = EXCLUDED.bank_id,
                        record_key = EXCLUDED.record_key,
                        content_hash = EXCLUDED.content_hash,
                        pipeline_version = EXCLUDED.pipeline_version,
                        classification = EXCLUDED.classification,
                        accepted_fact_count = EXCLUDED.accepted_fact_count,
                        review_fact_count = EXCLUDED.review_fact_count,
                        rejected_fact_count = EXCLUDED.rejected_fact_count,
                        processed_at = NOW()
                    """,
                    (
                        document_id,
                        bank_id,
                        record_key,
                        digest,
                        PIPELINE_VERSION,
                        Jsonb(classification),
                        analysis["ner"]["accepted_count"],
                        analysis["ner"]["review_count"],
                        analysis["ner"]["rejected_count"],
                    ),
                )
                cursor.execute(
                    """
                    UPDATE document_intake_review_queue
                    SET review_status = 'approved', updated_at = NOW()
                    WHERE record_key = %s
                      AND review_status = 'pending'
                    """,
                    (record_key,),
                )

    return {
        "mode": "DATABASE_WRITE",
        "action": "updated" if existing is not None else "inserted",
        "document_id": document_id,
        "document_review_id": None,
        "chunks_written": len(chunks),
        "facts_written": facts_written,
        "fact_reviews_queued": reviews_queued,
    }
