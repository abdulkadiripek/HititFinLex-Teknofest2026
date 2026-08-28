from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from .identity import normalize_text
from .models import Evidence
from .routing import BANKS


CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
NUMBER_PATTERN = re.compile(
    r"(?<![\w\]])[-+]?\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?%?"
    r"|(?<![\w\]])[-+]?\d+(?:[.,]\d+)?%?"
)
WORD_PATTERN = re.compile(r"[\wÇĞİÖŞÜçğıöşü]+", re.UNICODE)
STOP_WORDS = {
    "aciklanan",
    "ait",
    "ancak",
    "bankasi",
    "belirtilmistir",
    "bilgi",
    "bulunmaktadir",
    "finansmani",
    "icin",
    "ile",
    "kaynak",
    "olarak",
    "olan",
    "teklif",
    "verilmistir",
}

UNIT_PATTERNS = {
    "currency": re.compile(r"^(?:tl|try|lira|₺)(?:\b|$)", re.IGNORECASE),
    "percent": re.compile(r"^(?:%|yuzde\b)", re.IGNORECASE),
    "month": re.compile(r"^(?:ay|aydir|aylik)\b", re.IGNORECASE),
    "year": re.compile(r"^(?:yil|yildir|sene)\b", re.IGNORECASE),
    "day": re.compile(r"^(?:gun|gundur|gunluk)\b", re.IGNORECASE),
    "count": re.compile(r"^(?:adet|kez)\b", re.IGNORECASE),
}

UNIT_ALIASES = {
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",
    "tl": "currency",
    "try": "currency",
    "lira": "currency",
    "month": "month",
    "months": "month",
    "ay": "month",
    "year": "year",
    "years": "year",
    "yil": "year",
    "day": "day",
    "days": "day",
    "gun": "day",
    "count": "count",
    "adet": "count",
}


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    issues: list[str] = field(default_factory=list)


def _canonical_number(value: str) -> str | None:
    cleaned = value.strip().rstrip("%").replace(" ", "")
    if "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    elif re.fullmatch(r"[-+]?\d{1,3}\.\d{3}", cleaned):
        cleaned = cleaned.replace(".", "")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def _numbers(text: str) -> set[str]:
    output: set[str] = set()
    for match in NUMBER_PATTERN.finditer(text):
        canonical = _canonical_number(match.group(0))
        if canonical is not None:
            output.add(canonical)
    return output


def _unit_after(text: str, start: int, end: int, raw_number: str) -> str | None:
    if raw_number.rstrip().endswith("%"):
        return "percent"
    prefix = normalize_text(text[max(0, start - 12) : start]).rstrip()
    if prefix.endswith("%") or prefix.endswith("yuzde"):
        return "percent"
    suffix = normalize_text(text[end : end + 18]).lstrip(" ,.;:()[]{}-")
    for unit, pattern in UNIT_PATTERNS.items():
        if pattern.match(suffix):
            return unit
    return None


def _numeric_mentions(text: str) -> set[tuple[str, str | None]]:
    output: set[tuple[str, str | None]] = set()
    for match in NUMBER_PATTERN.finditer(text):
        canonical = _canonical_number(match.group(0))
        if canonical is None:
            continue
        output.add(
            (
                canonical,
                _unit_after(text, match.start(), match.end(), match.group(0)),
            )
        )
    return output


def _claims(answer: str) -> list[str]:
    claims: list[str] = []
    for line in answer.splitlines():
        cleaned = re.sub(
            r"^(?:[-*+\u2022]|\d+[.)])\s+",
            "",
            line.strip(),
        )
        if not cleaned:
            continue
        pieces = re.split(r"(?<=[.!?])\s+", cleaned)
        for piece in pieces:
            candidate = piece.strip()
            if not candidate:
                continue
            without_citations = CITATION_PATTERN.sub("", candidate)
            if not re.search(
                r"[\wÇĞİÖŞÜçğıöşü]",
                without_citations,
                re.UNICODE,
            ) and claims:
                claims[-1] = f"{claims[-1]} {candidate}"
            else:
                claims.append(candidate)
    return claims


def _evidence_text(item: Evidence) -> str:
    fact_text = " ".join(
        f"{fact.fact_text} {fact.evidence_text} {fact.normalized_value or ''}"
        for fact in item.facts
    )
    product_labels = " ".join(
        [item.primary_product or "", *item.product_types]
    ).replace("_", " ")
    return " ".join(
        [
            item.bank_name,
            product_labels,
            item.page_title or "",
            item.section_heading or "",
            item.effective_date.isoformat() if item.effective_date else "",
            item.campaign_start.isoformat() if item.campaign_start else "",
            item.campaign_end.isoformat() if item.campaign_end else "",
            item.content,
            fact_text,
        ]
    )


def _evidence_numeric_mentions(item: Evidence) -> set[tuple[str, str | None]]:
    output = _numeric_mentions(_evidence_text(item))
    for fact in item.facts:
        normalized = fact.normalized_value or {}
        value = normalized.get("value")
        if value is None:
            continue
        canonical = _canonical_number(str(value))
        if canonical is None:
            continue
        raw_unit = normalize_text(str(normalized.get("unit") or ""))
        unit = UNIT_ALIASES.get(raw_unit)
        if unit is None and normalized.get("currency"):
            unit = "currency"
        output.add((canonical, unit))
    return output


def _named_banks(text: str) -> set[str]:
    normalized = normalize_text(text)
    names: set[str] = set()
    for bank, aliases in BANKS.items():
        if any(normalize_text(alias) in normalized for alias in aliases):
            names.add(bank)
    return names


def _content_words(text: str) -> set[str]:
    return {
        normalize_text(word)
        for word in WORD_PATTERN.findall(text)
        if len(normalize_text(word)) >= 4
        and normalize_text(word) not in STOP_WORDS
        and not normalize_text(word).isdigit()
    }


def _has_word_support(claim_words: set[str], source_words: set[str]) -> bool:
    if not claim_words:
        return True
    supported = 0
    for claim_word in claim_words:
        if claim_word in source_words or any(
            len(claim_word) >= 4
            and len(source_word) >= 4
            and claim_word[:4] == source_word[:4]
            for source_word in source_words
        ):
            supported += 1
    required = 1 if len(claim_words) == 1 else max(
        2,
        (len(claim_words) + 1) // 2,
    )
    return supported >= required


def validate_answer(answer: str, evidence: list[Evidence]) -> ValidationResult:
    issues: list[str] = []
    evidence_by_id = {item.source_id: item for item in evidence}
    citations = {f"S{value}" for value in CITATION_PATTERN.findall(answer)}
    unknown = sorted(citations.difference(evidence_by_id))
    if unknown:
        issues.append("unknown_citations:" + ",".join(unknown))
    if not citations:
        issues.append("missing_citations")

    for index, claim in enumerate(_claims(answer), start=1):
        claim_content = CITATION_PATTERN.sub("", claim)
        if not re.search(r"[\wÇĞİÖŞÜçğıöşü]", claim_content, re.UNICODE):
            issues.append(f"claim_{index}_empty")
        claim_ids = [f"S{value}" for value in CITATION_PATTERN.findall(claim)]
        if not claim_ids:
            issues.append(f"claim_{index}_missing_citation")
            continue
        cited = [evidence_by_id[item] for item in claim_ids if item in evidence_by_id]
        if not cited:
            continue
        cited_offers = {item.offer_id for item in cited}
        if len(cited_offers) > 1:
            issues.append(f"claim_{index}_cross_offer_context")
        source_text = " ".join(_evidence_text(item) for item in cited)
        answer_mentions = _numeric_mentions(claim_content)
        source_mentions = set().union(
            *(_evidence_numeric_mentions(item) for item in cited)
        )
        missing_mentions = sorted(
            mention
            for mention in answer_mentions
            if mention not in source_mentions
            and not (
                mention[1] is None
                and any(value == mention[0] for value, _unit in source_mentions)
            )
        )
        if missing_mentions:
            issues.append(
                f"claim_{index}_unsupported_numbers:"
                + ",".join(
                    f"{number}:{unit or 'unspecified'}"
                    for number, unit in missing_mentions
                )
            )

        claim_banks = _named_banks(claim)
        cited_banks = set().union(*(_named_banks(item.bank_name) for item in cited))
        if claim_banks.difference(cited_banks):
            issues.append(f"claim_{index}_bank_context_mismatch")

        claim_words = _content_words(claim_content)
        source_words = _content_words(source_text)
        if claim_words and not _has_word_support(claim_words, source_words):
            issues.append(f"claim_{index}_unsupported_text")

    return ValidationResult(valid=not issues, issues=issues)
