from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote

import httpx

from .chunking import attach_facts, build_embedding_context, chunk_document
from .identity import (
    canonicalize_url,
    normalize_text,
    qdrant_point_id,
    stable_chunk_id,
    stable_offer_id,
)
from .settings import RagV2Settings


Scope = Literal["current", "historical"]
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)"
)
_ISO_DATE_PATTERN = re.compile(
    r"(?<!\d)((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})(?!\d)"
)
_WORD_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})\s+"
    r"(ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik)"
    r"(?:\s+(\d{4}))?",
    re.IGNORECASE,
)
_SHORT_WORD_DATE_RANGE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})\s*[-–—]\s*(\d{1,2})\s+"
    r"(ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik)"
    r"\s+((?:19|20)\d{2})(?!\d)",
    re.IGNORECASE,
)
_REWARD_PATTERN = re.compile(
    r"(?<!\d)(\d+(?:[.\s]\d{3})*(?:,\d+)?)\s*tl\s*"
    r"(?:['’]?\s*(?:ye|ya)\s+varan\s+)?"
    r"(world\s*puan|worldpuan|altin\s*puan|bonus\s*puan|alisveris\s*puani)"
    r"(?!\w)",
    re.IGNORECASE,
)
_MONTHS = {
    "ocak": 1,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "agustos": 8,
    "eylul": 9,
    "ekim": 10,
    "kasim": 11,
    "aralik": 12,
}
_PRODUCT_SIGNALS = {
    "housing": ("konut finansmani", "ev finansmani", "mortgage"),
    "vehicle": ("tasit finansmani", "arac finansmani", "otomobil finansmani"),
    "personal": ("ihtiyac finansmani", "bireysel finansman"),
    "card": ("kart kampanyasi", "kredi karti", "banka karti", "kart"),
    "investment": ("katilma hesabi", "yatirim urunu", "kar payi hesabi"),
    "commercial": ("ticari finansman", "kobi finansmani", "isletme finansmani"),
    "insurance": ("tekaful", "sigorta"),
}
_PRODUCT_FAMILY = {
    "KONUT_FINANSMANI": "housing",
    "TASIT_FINANSMANI": "vehicle",
    "IHTIYAC_FINANSMANI": "personal",
    "KART": "card",
    "KART_KAMPANYASI": "card",
    "KART_URUNU": "card",
    "ALISVERIS_PUANI": "card",
    "YATIRIM_URUNU": "investment",
    "KATILMA_HESABI": "investment",
    "TICARI_FINANSMAN": "commercial",
    "SIGORTA_TEKAFUL_URUNU": "insurance",
}
_IDENTITY_FACT_MARKERS = (
    "DONEM",
    "HARCAMA",
    "LIMIT",
    "ORAN",
    "ODUL",
    "TAKSIT",
    "TARIH",
    "TUTAR",
    "UCRET",
    "VADE",
)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    scope: Scope
    source_id: int
    bank_id: int
    bank_key: str
    bank_name: str
    source_url: str | None
    canonical_url: str
    page_title: str | None
    raw_text: str
    content_hash: str
    primary_product: str | None
    classification_confidence: float
    classification_decision: str | None
    classification_payload: dict[str, Any]
    verified: bool
    effective_date: date | None
    facts: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    @property
    def document_id(self) -> str:
        return f"{self.scope}:{self.source_id}"


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    chunk_id: str
    offer_id: str
    scope: Scope
    document_id: str
    current_source_id: int | None
    historical_source_id: int | None
    chunk_index: int
    content_hash: str
    bank_key: str
    bank_name: str
    primary_product: str | None
    product_types: tuple[str, ...]
    product_scores: dict[str, float]
    classification_confidence: float
    classification_status: str
    classification_conflict: bool
    page_title: str | None
    section_heading: str | None
    source_url: str | None
    canonical_url: str
    effective_date: date | None
    campaign_start: date | None
    campaign_end: date | None
    content: str
    facts: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    embedding_context: str
    embedding_model: str
    embedding_dimension: int
    qdrant_point_id: str

    def qdrant_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "offer_id": self.offer_id,
            "scope": self.scope,
            "document_id": self.document_id,
            "bank_key": self.bank_key,
            "bank_name": self.bank_name,
            "primary_product": self.primary_product,
            "product_types": list(self.product_types),
            "product_scores": self.product_scores,
            "classification_confidence": self.classification_confidence,
            "classification_status": self.classification_status,
            "classification_conflict": self.classification_conflict,
            "page_title": self.page_title,
            "section_heading": self.section_heading,
            "source_url": self.source_url,
            "effective_date": _iso_date(self.effective_date),
            "campaign_start": _iso_date(self.campaign_start),
            "campaign_end": _iso_date(self.campaign_end),
            "content": self.content,
            "facts": list(self.facts),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class IndexReport:
    documents: int = 0
    chunks: int = 0
    facts_attached: int = 0
    stale_points_removed: int = 0

    def add(self, other: "IndexReport") -> None:
        self.documents += other.documents
        self.chunks += other.chunks
        self.facts_attached += other.facts_attached
        self.stale_points_removed += other.stale_points_removed


class EmbeddingProvider(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class VectorIndex(Protocol):
    def ensure_collection(self, dimension: int) -> None:
        ...

    def upsert(
        self,
        chunks: Sequence[PreparedChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        ...

    def delete_points(self, point_ids: Sequence[str]) -> None:
        ...


def _iso_date(value: date | None) -> str | None:
    return f"{value.isoformat()}T00:00:00Z" if value else None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _clamp_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def extract_product_scores(payload: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    candidates = payload.get("product_top3")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            label = str(candidate.get("label") or "").strip()
            if label:
                output[label] = _clamp_confidence(candidate.get("score"))
    product = payload.get("product_type")
    if isinstance(product, Mapping):
        label = str(product.get("label") or "").strip()
        if label:
            output[label] = _clamp_confidence(product.get("score"))
    return output


def has_classification_conflict(
    *,
    page_title: str | None,
    source_url: str | None,
    primary_product: str | None,
    payload: Mapping[str, Any] | None = None,
) -> bool:
    review_reasons = (payload or {}).get("review_reasons")
    if isinstance(review_reasons, list) and any(
        "conflict" in str(item).lower() for item in review_reasons
    ):
        return True

    surface = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalize_text(f"{page_title or ''} {source_url or ''}"),
    )
    observed = {
        family
        for family, signals in _PRODUCT_SIGNALS.items()
        if any(signal in surface for signal in signals)
    }
    expected = _PRODUCT_FAMILY.get(str(primary_product or "").upper())
    if expected == "housing" and any(
        signal in surface
        for signal in (
            "arsa finansmani",
            "is yeri finansmani",
            "isyeri finansmani",
            "ihracat finansmani",
            "ihracat finansmanlari",
            "motosiklet finansmani",
            "tarim finansmani",
            "tarim finansmanlari",
            "tasit finansmani",
            "ticari finansman",
        )
    ):
        return True
    if not observed:
        return False
    if expected:
        return observed != {expected}
    return str(primary_product or "").upper() in {
        "DIGER",
        "DIGER_FINANSMAN",
        "DIGER_KAMPANYA",
    }


def classification_status(
    document: SourceDocument,
    *,
    accepted_confidence: float,
    review_confidence: float,
    conflict: bool,
) -> str:
    if document.verified:
        return "verified"
    decision = str(document.classification_decision or "").upper()
    quality = str(document.metadata.get("quality_status") or "").lower()
    if decision == "FAILED" or quality == "failed":
        return "required"
    if document.classification_confidence < review_confidence:
        return "required"
    if decision == "REVIEW" or conflict:
        return "review"
    if document.classification_confidence >= accepted_confidence:
        return "accepted"
    return "review"


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _dates_in_text(
    text: str,
    fallback_year: int | None,
    reference_date: date | None = None,
) -> list[date]:
    normalized = normalize_text(text)
    found: list[tuple[int, int, int, int | None]] = []
    occupied: list[tuple[int, int]] = []
    for match in _SHORT_WORD_DATE_RANGE_PATTERN.finditer(normalized):
        month = _MONTHS[match.group(3).lower()]
        year = int(match.group(4))
        found.extend(
            (
                (match.start(1), int(match.group(1)), month, year),
                (match.start(2), int(match.group(2)), month, year),
            )
        )
        occupied.append(match.span())
    for match in _ISO_DATE_PATTERN.finditer(normalized):
        found.append(
            (
                match.start(),
                int(match.group(3)),
                int(match.group(2)),
                int(match.group(1)),
            )
        )
    for match in _NUMERIC_DATE_PATTERN.finditer(normalized):
        found.append(
            (
                match.start(),
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
        )
    for match in _WORD_DATE_PATTERN.finditer(normalized):
        if any(
            start <= match.start() and match.end() <= end
            for start, end in occupied
        ):
            continue
        found.append(
            (
                match.start(),
                int(match.group(1)),
                _MONTHS[match.group(2).lower()],
                int(match.group(3)) if match.group(3) else None,
            )
        )
    found.sort(key=lambda item: item[0])
    if not found:
        return []

    years = [item[3] for item in found]
    for index, year in enumerate(years):
        if year is not None:
            continue
        later = next((item for item in years[index + 1 :] if item is not None), None)
        earlier = next(
            (item for item in reversed(years[:index]) if item is not None),
            None,
        )
        years[index] = later or earlier or fallback_year

    output: list[date] = []
    explicit_years = [item[3] for item in found]
    for (_, day, month, _), year in zip(found, years, strict=True):
        if year is None:
            continue
        parsed = _safe_date(int(year), month, day)
        if parsed:
            output.append(parsed)
    if len(output) != 2 or output[1] >= output[0]:
        return output

    first_explicit, second_explicit = explicit_years
    if first_explicit is not None and second_explicit is None:
        adjusted = _safe_date(
            output[1].year + 1,
            output[1].month,
            output[1].day,
        )
        return [output[0], adjusted] if adjusted else output
    if first_explicit is None and second_explicit is not None:
        adjusted = _safe_date(
            output[0].year - 1,
            output[0].month,
            output[0].day,
        )
        return [adjusted, output[1]] if adjusted else output
    if first_explicit is not None or second_explicit is not None:
        return output
    if reference_date is None:
        return output

    previous_start = _safe_date(
        output[0].year - 1,
        output[0].month,
        output[0].day,
    )
    previous_end = output[1]
    next_start = output[0]
    next_end = _safe_date(
        output[1].year + 1,
        output[1].month,
        output[1].day,
    )
    candidates = [
        (start, end)
        for start, end in (
            (previous_start, previous_end),
            (next_start, next_end),
        )
        if start is not None
        and end is not None
        and start <= reference_date <= end
    ]
    if len(candidates) == 1:
        return [candidates[0][0], candidates[0][1]]
    return output


def _fact_date_texts(fact: Mapping[str, Any]) -> tuple[str, ...]:
    normalized_value = fact.get("normalized_value")
    normalized_text = (
        _json(normalized_value) if normalized_value is not None else ""
    )
    return (
        str(fact.get("fact_text") or ""),
        str(fact.get("evidence_text") or ""),
        normalized_text,
    )


def _date_token_count(text: str) -> int:
    normalized = normalize_text(text)
    short_spans = {
        match.span() for match in _SHORT_WORD_DATE_RANGE_PATTERN.finditer(normalized)
    }
    spans = {
        match.span()
        for pattern in (
            _ISO_DATE_PATTERN,
            _NUMERIC_DATE_PATTERN,
            _WORD_DATE_PATTERN,
        )
        for match in pattern.finditer(normalized)
        if not any(
            start <= match.start() and match.end() <= end
            for start, end in short_spans
        )
    }
    return len(spans) + (2 * len(short_spans))


def _sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", text)
        if item.strip()
    ]


def _decimal_value(value: str) -> int | float | None:
    cleaned = value.replace(" ", "")
    if "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif cleaned.count(".") > 1 or re.fullmatch(r"\d{1,3}\.\d{3}", cleaned):
        cleaned = cleaned.replace(".", "")
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def extract_high_confidence_facts(
    text: str,
    *,
    effective_date: date | None,
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    full_text = normalize_text(text)
    for sentence in _sentences(text):
        normalized = normalize_text(sentence)
        for match in _REWARD_PATTERN.finditer(normalized):
            value = _decimal_value(match.group(1))
            if value is None:
                continue
            reward_name = (
                "Worldpuan"
                if "world" in match.group(2)
                else "Altin Puan"
                if "altin" in match.group(2)
                else "Bonus Puan"
                if "bonus" in match.group(2)
                else "Alisveris Puani"
            )
            key = ("ALISVERIS_PUANI", f"{value}:{reward_name}")
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "fact_type": "ALISVERIS_PUANI",
                    "fact_text": f"{match.group(1)} TL {reward_name}",
                    "normalized_value": {
                        "value": value,
                        "unit": "currency",
                        "currency": "TRY",
                    },
                    "evidence_text": sentence,
                    "confidence": 0.99,
                }
            )

        if "kampanya" not in normalized:
            continue
        parsed_dates = _dates_in_text(
            sentence,
            effective_date.year if effective_date else None,
            effective_date,
        )
        is_validity_range = any(
            marker in normalized
            for marker in (
                "kampanya araligi",
                "baslangic ve bitis",
                "tarihleri arasinda",
                "tarihlerinde gecerli",
                "tarihleri boyunca gecerli",
            )
        )
        if (
            len(parsed_dates) >= 2
            and parsed_dates[-1] >= parsed_dates[0]
            and is_validity_range
        ):
            start_date = parsed_dates[0]
            end_date = parsed_dates[-1]
            key = (
                "KAMPANYA_TARIH_ARALIGI",
                f"{start_date.isoformat()}:{end_date.isoformat()}",
            )
            if key not in seen:
                seen.add(key)
                output.append(
                    {
                        "fact_type": "KAMPANYA_TARIH_ARALIGI",
                        "fact_text": (
                            f"{start_date.isoformat()} - {end_date.isoformat()}"
                        ),
                        "normalized_value": {
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                        },
                        "evidence_text": sentence,
                        "confidence": 0.99,
                    }
                )
        elif len(parsed_dates) == 1 and any(
            marker in normalized
            for marker in (
                "tarihine kadar gecerli",
                "tarihine dek gecerli",
                "kampanya bitis",
                "son katilim tarihi",
            )
        ):
            end_date = parsed_dates[0]
            key = ("BASVURU_SON_TARIHI", end_date.isoformat())
            if key not in seen:
                seen.add(key)
                output.append(
                    {
                        "fact_type": "BASVURU_SON_TARIHI",
                        "fact_text": end_date.isoformat(),
                        "normalized_value": {"end_date": end_date.isoformat()},
                        "evidence_text": sentence,
                        "confidence": 0.99,
                    }
                )
    date_facts = [
        item
        for item in output
        if item["fact_type"]
        in {"KAMPANYA_TARIH_ARALIGI", "BASVURU_SON_TARIHI"}
    ]
    is_campaign_index = (
        full_text.count("kampanya araligi") > 1
        or "tum kampanyalar" in full_text
    )
    if date_facts and not is_campaign_index:
        full_ranges = [
            item
            for item in date_facts
            if item["fact_type"] == "KAMPANYA_TARIH_ARALIGI"
        ]
        selected_dates = full_ranges[:1] or date_facts[:1]
        output = [
            item
            for item in output
            if item["fact_type"]
            not in {"KAMPANYA_TARIH_ARALIGI", "BASVURU_SON_TARIHI"}
        ] + selected_dates
    return tuple(output)


def merge_document_facts(document: SourceDocument) -> SourceDocument:
    derived = extract_high_confidence_facts(
        document.raw_text,
        effective_date=document.effective_date,
    )
    if not derived:
        return document
    merged: list[dict[str, Any]] = list(document.facts)
    keys = {
        (
            str(item.get("fact_type") or "").upper(),
            _json(item.get("normalized_value")),
            normalize_text(str(item.get("fact_text") or "")),
        )
        for item in merged
    }
    for item in derived:
        key = (
            str(item.get("fact_type") or "").upper(),
            _json(item.get("normalized_value")),
            normalize_text(str(item.get("fact_text") or "")),
        )
        if key not in keys:
            keys.add(key)
            merged.append(item)
    return replace(document, facts=tuple(merged))


def _fact_dates(
    fact: Mapping[str, Any],
    effective_date: date | None,
) -> list[date]:
    fallback: list[date] = []
    for candidate in _fact_date_texts(fact):
        parsed = _dates_in_text(
            candidate,
            effective_date.year if effective_date else None,
            effective_date,
        )
        if len(parsed) >= 2:
            return parsed
        if len(parsed) > len(fallback):
            fallback = parsed
    return fallback


def campaign_bounds(
    facts: Iterable[Mapping[str, Any]],
    *,
    effective_date: date | None,
) -> tuple[date | None, date | None]:
    starts: list[date] = []
    ends: list[date] = []
    for fact in facts:
        fact_type = str(fact.get("fact_type") or "").upper()
        if fact_type not in {"KAMPANYA_TARIH_ARALIGI", "BASVURU_SON_TARIHI"}:
            continue
        parsed = _fact_dates(fact, effective_date)
        if not parsed:
            continue
        if fact_type == "BASVURU_SON_TARIHI":
            ends.append(parsed[-1])
        elif len(parsed) == 1:
            starts.append(parsed[0])
            ends.append(parsed[0])
        elif parsed[-1] < parsed[0]:
            continue
        else:
            starts.append(parsed[0])
            ends.append(parsed[-1])
    return (min(starts) if starts else None, max(ends) if ends else None)


def has_ambiguous_campaign_period(
    facts: Iterable[Mapping[str, Any]],
    *,
    effective_date: date | None,
) -> bool:
    for fact in facts:
        if str(fact.get("fact_type") or "").upper() != "KAMPANYA_TARIH_ARALIGI":
            continue
        parsed = _fact_dates(fact, effective_date)
        if len(parsed) >= 2 and parsed[-1] < parsed[0]:
            return True
        if len(parsed) < 2 and any(
            _date_token_count(candidate) >= 2
            for candidate in _fact_date_texts(fact)
        ):
            return True
    return False


def has_multiple_campaign_periods(
    facts: Iterable[Mapping[str, Any]],
    *,
    effective_date: date | None,
) -> bool:
    periods: set[tuple[date | None, date | None]] = set()
    for fact in facts:
        fact_type = str(fact.get("fact_type") or "").upper()
        if fact_type not in {"KAMPANYA_TARIH_ARALIGI", "BASVURU_SON_TARIHI"}:
            continue
        bounds = campaign_bounds((fact,), effective_date=effective_date)
        if any(bounds):
            periods.add(bounds)
    return len(periods) > 1


def _offer_identity_boundary(
    document: SourceDocument,
    start_date: date | None,
    end_date: date | None,
) -> str:
    if start_date is None and end_date is None:
        return document.content_hash
    identity_facts: list[tuple[str, str, str]] = []
    for fact in document.facts:
        fact_type = str(fact.get("fact_type") or "").upper()
        if not any(marker in fact_type for marker in _IDENTITY_FACT_MARKERS):
            continue
        identity_facts.append(
            (
                fact_type,
                normalize_text(str(fact.get("fact_text") or "")),
                _json(fact.get("normalized_value")),
            )
        )
    encoded = _json(sorted(identity_facts))
    fact_digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"campaign:{start_date or ''}:{end_date or ''}:{fact_digest}"


def prepare_document(
    document: SourceDocument,
    settings: RagV2Settings,
    *,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[PreparedChunk]:
    document = merge_document_facts(document)
    multiple_periods = has_multiple_campaign_periods(
        document.facts,
        effective_date=document.effective_date,
    )
    ambiguous_period = has_ambiguous_campaign_period(
        document.facts,
        effective_date=document.effective_date,
    )
    conflict = has_classification_conflict(
        page_title=document.page_title,
        source_url=document.source_url,
        primary_product=document.primary_product,
        payload=document.classification_payload,
    ) or multiple_periods or ambiguous_period
    status = classification_status(
        document,
        accepted_confidence=settings.accepted_confidence,
        review_confidence=settings.review_confidence,
        conflict=conflict,
    )
    drafts = chunk_document(
        document.raw_text,
        page_title=document.page_title,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    if not drafts:
        drafts = chunk_document(
            document.raw_text,
            page_title=document.page_title,
            max_words=max_words,
            overlap_words=overlap_words,
            include_navigation=True,
        )
    drafts = attach_facts(drafts, document.facts)
    start_date, end_date = campaign_bounds(
        document.facts,
        effective_date=document.effective_date,
    )
    product_scores = extract_product_scores(document.classification_payload)
    product_types = (document.primary_product,) if document.primary_product else ()
    identity_boundary = _offer_identity_boundary(
        document,
        start_date,
        end_date,
    )
    offer_id = stable_offer_id(
        bank=document.bank_key or document.bank_name,
        product=document.primary_product,
        source_url=document.canonical_url or document.source_url,
        title=document.page_title,
        content_boundary=identity_boundary,
        campaign_start=start_date,
        campaign_end=end_date,
    )

    output: list[PreparedChunk] = []
    for draft in drafts:
        chunk_id = stable_chunk_id(
            document.scope,
            document.source_id,
            draft.chunk_index,
            draft.content,
        )
        metadata = {
            **document.metadata,
            "token_count": draft.token_count,
            "is_navigation": draft.is_navigation,
            "multiple_campaign_periods": multiple_periods,
            "ambiguous_campaign_period": ambiguous_period,
        }
        output.append(
            PreparedChunk(
                chunk_id=chunk_id,
                offer_id=offer_id,
                scope=document.scope,
                document_id=document.document_id,
                current_source_id=(
                    document.source_id if document.scope == "current" else None
                ),
                historical_source_id=(
                    document.source_id if document.scope == "historical" else None
                ),
                chunk_index=draft.chunk_index,
                content_hash=draft.content_hash,
                bank_key=document.bank_key,
                bank_name=document.bank_name,
                primary_product=document.primary_product,
                product_types=product_types,
                product_scores=product_scores,
                classification_confidence=document.classification_confidence,
                classification_status=status,
                classification_conflict=conflict,
                page_title=document.page_title,
                section_heading=draft.section_heading,
                source_url=document.source_url,
                canonical_url=document.canonical_url,
                effective_date=document.effective_date,
                campaign_start=start_date,
                campaign_end=end_date,
                content=draft.content,
                facts=draft.facts,
                metadata=metadata,
                embedding_context=build_embedding_context(
                    bank_name=document.bank_name,
                    primary_product=document.primary_product,
                    page_title=document.page_title,
                    section_heading=draft.section_heading,
                    content=draft.content,
                ),
                embedding_model=settings.evren_embedding_model,
                embedding_dimension=settings.embedding_dimension,
                qdrant_point_id=qdrant_point_id(chunk_id),
            )
        )
    return output


class EvrenEmbeddingProvider:
    def __init__(
        self,
        settings: RagV2Settings,
        *,
        client: httpx.Client | None = None,
        batch_size: int = 64,
    ) -> None:
        if not settings.evren_ready:
            raise RuntimeError("EVREN embedding service is not configured")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = settings.evren_embedding_model
        self.expected_dimension = settings.embedding_dimension
        self.batch_size = batch_size
        self._base_url = settings.evren_base_url
        self._api_key = settings.evren_api_key
        self._max_retries = settings.evren_max_retries
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                settings.evren_timeout_seconds,
                connect=settings.evren_connect_timeout_seconds,
            ),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "EvrenEmbeddingProvider":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _request(self, payload: dict[str, Any]) -> httpx.Response:
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except httpx.TransportError as error:
                if attempt >= self._max_retries:
                    raise RuntimeError("EVREN embedding request failed") from error
                time.sleep(min(0.25 * (2**attempt), 2.0))
                continue
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt >= self._max_retries:
                break
            time.sleep(min(0.25 * (2**attempt), 2.0))
        if response is None or response.is_error:
            status = response.status_code if response is not None else "transport"
            raise RuntimeError(f"EVREN embedding request failed with status {status}")
        return response

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        output: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = list(texts[offset : offset + self.batch_size])
            response = self._request(
                {
                    "model": self.model_name,
                    "input": batch,
                    "encoding_format": "float",
                }
            )
            payload = response.json()
            data = payload.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise RuntimeError("EVREN embedding response has an invalid item count")
            if not all(isinstance(item, Mapping) for item in data):
                raise RuntimeError("EVREN embedding response contains an invalid item")
            try:
                indices = [int(item.get("index", 0)) for item in data]
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "EVREN embedding response has invalid indexes"
                ) from error
            if sorted(indices) != list(range(len(batch))):
                raise RuntimeError("EVREN embedding response indexes are incomplete")
            ordered = [
                item
                for _, item in sorted(
                    zip(indices, data),
                    key=lambda pair: pair[0],
                )
            ]
            for item in ordered:
                vector = item.get("embedding") if isinstance(item, Mapping) else None
                if not isinstance(vector, list) or not vector:
                    raise RuntimeError(
                        "EVREN embedding response contains an invalid vector"
                    )
                parsed = [float(value) for value in vector]
                if not all(math.isfinite(value) for value in parsed):
                    raise RuntimeError(
                        "EVREN embedding response contains a non-finite value"
                    )
                if len(parsed) != self.expected_dimension:
                    raise RuntimeError(
                        "EVREN embedding dimension does not match configuration: "
                        f"expected {self.expected_dimension}, received {len(parsed)}"
                    )
                output.append(parsed)
        return output


class QdrantRestIndex:
    _PAYLOAD_INDEXES = {
        "chunk_id": "keyword",
        "offer_id": "keyword",
        "scope": "keyword",
        "document_id": "keyword",
        "bank_key": "keyword",
        "primary_product": "keyword",
        "product_types": "keyword",
        "classification_confidence": "float",
        "classification_status": "keyword",
        "classification_conflict": "bool",
        "effective_date": "datetime",
        "campaign_start": "datetime",
        "campaign_end": "datetime",
    }

    def __init__(
        self,
        settings: RagV2Settings,
        *,
        client: httpx.Client | None = None,
        batch_size: int = 64,
    ) -> None:
        if not settings.qdrant_ready:
            raise RuntimeError("Qdrant is not configured")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._base_url = settings.qdrant_url
        self._collection = settings.qdrant_collection
        self._api_key = settings.qdrant_api_key
        self._max_retries = settings.qdrant_max_retries
        self._batch_size = batch_size
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(settings.qdrant_timeout_seconds),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        collection = quote(self._collection, safe="")
        self._collection_url = f"{self._base_url}/collections/{collection}"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "QdrantRestIndex":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        accepted: set[int] | None = None,
    ) -> httpx.Response:
        accepted = accepted or {200}
        response: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    headers={
                        "api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            except httpx.TransportError as error:
                if attempt >= self._max_retries:
                    raise RuntimeError("Qdrant request failed") from error
                time.sleep(min(0.25 * (2**attempt), 2.0))
                continue
            if response.status_code in accepted:
                return response
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt < self._max_retries:
                time.sleep(min(0.25 * (2**attempt), 2.0))
        status = response.status_code if response is not None else "transport"
        raise RuntimeError(f"Qdrant request failed with status {status}")

    @staticmethod
    def _collection_dimension(payload: Mapping[str, Any]) -> int | None:
        result = payload.get("result")
        if not isinstance(result, Mapping):
            return None
        config = result.get("config")
        params = config.get("params") if isinstance(config, Mapping) else None
        vectors = params.get("vectors") if isinstance(params, Mapping) else None
        if not isinstance(vectors, Mapping):
            return None
        if "size" in vectors:
            return int(vectors["size"])
        for vector_config in vectors.values():
            if isinstance(vector_config, Mapping) and "size" in vector_config:
                return int(vector_config["size"])
        return None

    def ensure_collection(self, dimension: int) -> None:
        response = self._request(
            "GET",
            self._collection_url,
            accepted={200, 404},
        )
        if response.status_code == 404:
            self._request(
                "PUT",
                self._collection_url,
                payload={"vectors": {"size": dimension, "distance": "Cosine"}},
                accepted={200, 201, 409},
            )
            response = self._request(
                "GET",
                self._collection_url,
                accepted={200},
            )
        existing = self._collection_dimension(response.json())
        if existing is None:
            raise RuntimeError("Qdrant collection vector size could not be verified")
        if existing != dimension:
            raise RuntimeError(
                "Qdrant collection vector size mismatch: "
                f"expected {dimension}, received {existing}"
            )

        for field_name, field_schema in self._PAYLOAD_INDEXES.items():
            self._request(
                "PUT",
                f"{self._collection_url}/index?wait=true",
                payload={
                    "field_name": field_name,
                    "field_schema": field_schema,
                },
                accepted={200, 201, 409},
            )

    def upsert(
        self,
        chunks: Sequence[PreparedChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts must match")
        if not chunks:
            return
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("All vectors must have the same dimension")
        for offset in range(0, len(chunks), self._batch_size):
            batch_chunks = chunks[offset : offset + self._batch_size]
            batch_vectors = vectors[offset : offset + self._batch_size]
            points = [
                {
                    "id": chunk.qdrant_point_id,
                    "vector": list(vector),
                    "payload": chunk.qdrant_payload(),
                }
                for chunk, vector in zip(
                    batch_chunks,
                    batch_vectors,
                    strict=True,
                )
            ]
            self._request(
                "PUT",
                f"{self._collection_url}/points?wait=true",
                payload={"points": points},
                accepted={200, 201, 202},
            )

    def delete_points(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        for offset in range(0, len(point_ids), self._batch_size):
            self._request(
                "POST",
                f"{self._collection_url}/points/delete?wait=true",
                payload={"points": list(point_ids[offset : offset + self._batch_size])},
                accepted={200, 201, 202},
            )


_UPSERT_SQL = """
INSERT INTO rag_chunks (
    chunk_id,
    offer_id,
    scope,
    document_id,
    current_source_id,
    historical_source_id,
    chunk_index,
    content_hash,
    bank_key,
    bank_name,
    primary_product,
    product_types,
    product_scores,
    classification_confidence,
    classification_status,
    classification_conflict,
    page_title,
    section_heading,
    source_url,
    canonical_url,
    effective_date,
    campaign_start,
    campaign_end,
    content,
    facts,
    metadata,
    embedding_context,
    embedding_model,
    embedding_dimension,
    qdrant_point_id
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s::JSONB, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s::JSONB, %s::JSONB, %s, %s, %s, %s
)
ON CONFLICT (chunk_id) DO UPDATE SET
    offer_id = EXCLUDED.offer_id,
    bank_key = EXCLUDED.bank_key,
    bank_name = EXCLUDED.bank_name,
    primary_product = EXCLUDED.primary_product,
    product_types = EXCLUDED.product_types,
    product_scores = EXCLUDED.product_scores,
    classification_confidence = EXCLUDED.classification_confidence,
    classification_status = EXCLUDED.classification_status,
    classification_conflict = EXCLUDED.classification_conflict,
    page_title = EXCLUDED.page_title,
    section_heading = EXCLUDED.section_heading,
    source_url = EXCLUDED.source_url,
    canonical_url = EXCLUDED.canonical_url,
    effective_date = EXCLUDED.effective_date,
    campaign_start = EXCLUDED.campaign_start,
    campaign_end = EXCLUDED.campaign_end,
    content = EXCLUDED.content,
    facts = EXCLUDED.facts,
    metadata = EXCLUDED.metadata,
    embedding_context = EXCLUDED.embedding_context,
    embedding_model = EXCLUDED.embedding_model,
    embedding_dimension = EXCLUDED.embedding_dimension,
    qdrant_point_id = EXCLUDED.qdrant_point_id,
    updated_at = NOW()
"""


def _upsert_parameters(chunk: PreparedChunk) -> tuple[Any, ...]:
    return (
        chunk.chunk_id,
        chunk.offer_id,
        chunk.scope,
        chunk.document_id,
        chunk.current_source_id,
        chunk.historical_source_id,
        chunk.chunk_index,
        chunk.content_hash,
        chunk.bank_key,
        chunk.bank_name,
        chunk.primary_product,
        list(chunk.product_types),
        _json(chunk.product_scores),
        chunk.classification_confidence,
        chunk.classification_status,
        chunk.classification_conflict,
        chunk.page_title,
        chunk.section_heading,
        chunk.source_url,
        chunk.canonical_url,
        chunk.effective_date,
        chunk.campaign_start,
        chunk.campaign_end,
        chunk.content,
        _json(list(chunk.facts)),
        _json(chunk.metadata),
        chunk.embedding_context,
        chunk.embedding_model,
        chunk.embedding_dimension,
        chunk.qdrant_point_id,
    )


def sync_postgres_chunks(connection: Any, chunks: Sequence[PreparedChunk]) -> list[str]:
    by_document: dict[str, list[PreparedChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)
    if not by_document:
        return []

    existing_rows = connection.execute(
        """
        SELECT document_id, chunk_id, qdrant_point_id::TEXT
        FROM rag_chunks
        WHERE document_id = ANY(%s)
        """,
        (list(by_document),),
    ).fetchall()
    desired = {
        chunk.chunk_id
        for document_chunks in by_document.values()
        for chunk in document_chunks
    }
    stale_points = [
        str(point_id)
        for _document_id, chunk_id, point_id in existing_rows
        if str(chunk_id).strip() not in desired
    ]

    transaction = getattr(connection, "transaction", None)
    context = transaction() if transaction else _NullContext()
    with context:
        for document_id, document_chunks in by_document.items():
            connection.execute(
                """
                DELETE FROM rag_chunks
                WHERE document_id = %s
                  AND NOT (chunk_id = ANY(%s))
                """,
                (document_id, [chunk.chunk_id for chunk in document_chunks]),
            )
        with connection.cursor() as cursor:
            cursor.executemany(
                _UPSERT_SQL,
                [_upsert_parameters(chunk) for chunk in chunks],
            )
    return stale_points


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _fact_map(
    rows: Iterable[Sequence[Any]],
    key_index: int = 0,
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        source_id = int(row[key_index])
        output.setdefault(source_id, []).append(
            {
                "fact_type": row[1],
                "fact_text": row[2],
                "normalized_value": row[3],
                "evidence_text": row[4],
                "confidence": row[5],
            }
        )
    return output


def load_source_documents(
    connection: Any,
    scope: Scope,
    *,
    limit: int | None = None,
) -> list[SourceDocument]:
    if scope == "current":
        query = """
            SELECT
                d.id,
                d.bank_id,
                b.bank_key,
                b.bank_name,
                d.source_url,
                d.page_title,
                d.raw_text,
                d.campaign_type_code,
                d.confidence,
                d.verified,
                d.auto_accepted,
                d.record_key,
                d.label_source,
                d.updated_at,
                COALESCE(state.classification, '{}'::JSONB)
            FROM documents AS d
            JOIN banks AS b ON b.id = d.bank_id
            LEFT JOIN document_intake_state AS state
              ON state.document_id = d.id
            WHERE BTRIM(d.raw_text) <> ''
            ORDER BY d.id
        """
    else:
        query = """
            SELECT
                d.id,
                d.bank_id,
                b.bank_key,
                b.bank_name,
                d.source_url,
                d.page_title,
                d.raw_text,
                d.product_type_code,
                d.classification_confidence,
                d.verified,
                d.classification_decision,
                d.archive_key,
                d.canonical_url,
                d.snapshot_date,
                COALESCE(d.classification_payload, '{}'::JSONB),
                d.quality_status,
                d.searchable,
                d.content_hash,
                d.source_dataset,
                d.pipeline_version
            FROM historical_documents AS d
            JOIN banks AS b ON b.id = d.bank_id
            WHERE BTRIM(d.raw_text) <> ''
            ORDER BY d.id
        """
    parameters: tuple[Any, ...] = ()
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        query += " LIMIT %s"
        parameters = (limit,)
    rows = connection.execute(query, parameters).fetchall()
    if not rows:
        return []
    source_ids = [int(row[0]) for row in rows]

    if scope == "current":
        fact_rows = connection.execute(
            """
            SELECT
                document_id,
                fact_type,
                fact_text,
                normalized_value,
                evidence_text,
                confidence
            FROM comparison_facts
            WHERE document_id = ANY(%s)
            ORDER BY document_id, source_chunk, id
            """,
            (source_ids,),
        ).fetchall()
    else:
        fact_rows = connection.execute(
            """
            SELECT
                historical_document_id,
                fact_type,
                fact_text,
                normalized_value,
                evidence_text,
                confidence
            FROM historical_facts
            WHERE historical_document_id = ANY(%s)
              AND (decision = 'accepted' OR review_status = 'approved')
            ORDER BY historical_document_id, source_chunk, id
            """,
            (source_ids,),
        ).fetchall()
    facts = _fact_map(fact_rows)

    documents: list[SourceDocument] = []
    for row in rows:
        source_id = int(row[0])
        raw_text = str(row[6])
        if scope == "current":
            payload = row[14] if isinstance(row[14], dict) else {}
            digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            documents.append(
                SourceDocument(
                    scope="current",
                    source_id=source_id,
                    bank_id=int(row[1]),
                    bank_key=str(row[2]),
                    bank_name=str(row[3]),
                    source_url=str(row[4]) if row[4] else None,
                    canonical_url=canonicalize_url(str(row[4]) if row[4] else None),
                    page_title=str(row[5]) if row[5] else None,
                    raw_text=raw_text,
                    content_hash=digest,
                    primary_product=str(row[7]) if row[7] else None,
                    classification_confidence=_clamp_confidence(row[8]),
                    classification_decision=(
                        str(payload.get("decision"))
                        if payload.get("decision")
                        else None
                    ),
                    classification_payload=payload,
                    verified=bool(row[9]),
                    effective_date=(row[13].date() if row[13] else None),
                    facts=tuple(facts.get(source_id, ())),
                    metadata={
                        "record_key": str(row[11]),
                        "label_source": str(row[12]) if row[12] else None,
                        "auto_accepted": bool(row[10]),
                        "source_updated_at": row[13].isoformat() if row[13] else None,
                    },
                )
            )
        else:
            payload = row[14] if isinstance(row[14], dict) else {}
            digest = str(row[17]).strip()
            if not _HASH_PATTERN.fullmatch(digest):
                digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            documents.append(
                SourceDocument(
                    scope="historical",
                    source_id=source_id,
                    bank_id=int(row[1]),
                    bank_key=str(row[2]),
                    bank_name=str(row[3]),
                    source_url=str(row[4]) if row[4] else None,
                    canonical_url=canonicalize_url(
                        str(row[12]) if row[12] else str(row[4])
                    ),
                    page_title=str(row[5]) if row[5] else None,
                    raw_text=raw_text,
                    content_hash=digest,
                    primary_product=str(row[7]) if row[7] else None,
                    classification_confidence=_clamp_confidence(row[8]),
                    classification_decision=str(row[10]) if row[10] else None,
                    classification_payload=payload,
                    verified=bool(row[9]),
                    effective_date=row[13],
                    facts=tuple(facts.get(source_id, ())),
                    metadata={
                        "archive_key": str(row[11]),
                        "quality_status": str(row[15]),
                        "searchable": bool(row[16]),
                        "source_dataset": str(row[18]),
                        "pipeline_version": str(row[19]),
                    },
                )
            )
    return documents


def _batched(
    values: Sequence[SourceDocument],
    size: int,
) -> Iterator[Sequence[SourceDocument]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def index_documents(
    connection: Any,
    documents: Sequence[SourceDocument],
    settings: RagV2Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    vector_index: VectorIndex | None = None,
    document_batch_size: int = 16,
    max_words: int = 220,
    overlap_words: int = 40,
    dry_run: bool = False,
) -> IndexReport:
    if (embedding_provider is None) != (vector_index is None):
        raise ValueError(
            "embedding_provider and vector_index must be provided together"
        )
    report = IndexReport()
    collection_ready = False
    for document_batch in _batched(documents, document_batch_size):
        chunks = [
            chunk
            for document in document_batch
            for chunk in prepare_document(
                document,
                settings,
                max_words=max_words,
                overlap_words=overlap_words,
            )
        ]
        batch_report = IndexReport(
            documents=len(document_batch),
            chunks=len(chunks),
            facts_attached=sum(len(chunk.facts) for chunk in chunks),
        )
        if dry_run:
            report.add(batch_report)
            continue
        if not chunks:
            report.add(batch_report)
            continue

        if embedding_provider is not None and vector_index is not None:
            vectors = embedding_provider.embed(
                [chunk.embedding_context for chunk in chunks]
            )
            if len(vectors) != len(chunks):
                raise RuntimeError(
                    "Embedding provider returned an invalid vector count"
                )
            dimension = len(vectors[0])
            if not collection_ready:
                vector_index.ensure_collection(dimension)
                collection_ready = True
            vector_index.upsert(chunks, vectors)

        stale_points = sync_postgres_chunks(connection, chunks)
        if vector_index is not None and stale_points:
            vector_index.delete_points(stale_points)
        batch_report.stale_points_removed = len(stale_points)
        report.add(batch_report)
    return report


def _connect(settings: RagV2Settings) -> Any:
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError("psycopg is required for indexing") from error
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )


def _env_int(name: str, default: int, minimum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def main() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise RuntimeError("python-dotenv is required for the indexer CLI") from error
    load_dotenv(backend_dir / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Build the PostgreSQL and Qdrant RAG V2 indexes."
    )
    parser.add_argument(
        "--scope",
        choices=("current", "historical", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-remote", action="store_true")
    args = parser.parse_args()

    settings = RagV2Settings.from_env()
    scopes: tuple[Scope, ...] = (
        ("current", "historical")
        if args.scope == "all"
        else (args.scope,)
    )
    max_words = _env_int("RAG_V2_CHUNK_MAX_WORDS", 220, 16)
    overlap_words = _env_int("RAG_V2_CHUNK_OVERLAP_WORDS", 40, 0)
    if overlap_words >= max_words:
        raise ValueError("RAG_V2_CHUNK_OVERLAP_WORDS must be smaller than max words")
    document_batch_size = _env_int("RAG_V2_INDEX_DOCUMENT_BATCH_SIZE", 16, 1)
    embedding_batch_size = _env_int("RAG_V2_EMBEDDING_BATCH_SIZE", 64, 1)

    embedding_provider: EvrenEmbeddingProvider | None = None
    vector_index: QdrantRestIndex | None = None
    if not args.skip_remote and not args.dry_run:
        if not settings.evren_ready or not settings.qdrant_ready:
            raise RuntimeError(
                "EVREN and Qdrant must be configured unless --skip-remote is used"
            )
        embedding_provider = EvrenEmbeddingProvider(
            settings,
            batch_size=embedding_batch_size,
        )
        vector_index = QdrantRestIndex(
            settings,
            batch_size=embedding_batch_size,
        )

    total = IndexReport()
    try:
        with _connect(settings) as connection:
            if not args.dry_run:
                migrated = connection.execute(
                    "SELECT to_regclass('public.rag_chunks') IS NOT NULL"
                ).fetchone()
                if migrated != (True,):
                    raise RuntimeError("RAG V2 migration 0003 has not been applied")
            for scope in scopes:
                documents = load_source_documents(connection, scope, limit=args.limit)
                result = index_documents(
                    connection,
                    documents,
                    settings,
                    embedding_provider=embedding_provider,
                    vector_index=vector_index,
                    document_batch_size=document_batch_size,
                    max_words=max_words,
                    overlap_words=overlap_words,
                    dry_run=args.dry_run,
                )
                total.add(result)
                print(
                    f"indexed scope={scope} documents={result.documents} "
                    f"chunks={result.chunks} facts={result.facts_attached}"
                )
    finally:
        if embedding_provider is not None:
            embedding_provider.close()
        if vector_index is not None:
            vector_index.close()

    print(
        f"index complete documents={total.documents} chunks={total.chunks} "
        f"facts={total.facts_attached} stale={total.stale_points_removed}"
    )


if __name__ == "__main__":
    main()
