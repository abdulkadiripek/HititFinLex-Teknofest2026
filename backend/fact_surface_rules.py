from __future__ import annotations

import re
import unicodedata


MONEY_LABELS = {
    "EKSPERTIZ_UCRETI",
    "FINANSMAN_TUTARI",
    "HARCAMA_ESIGI",
    "HARCAMA_UST_LIMITI",
    "INDIRIM_TUTARI",
    "IPOTEK_TESIS_UCRETI",
    "ISLEM_ALT_LIMITI",
    "ISLEM_UST_LIMITI",
    "MINIMUM_BAKIYE",
    "ODUL_TUTARI",
    "TAHSIS_UCRETI",
}

PERCENT_LABELS = {
    "INDIRIM_ORANI",
    "KAR_PAYI_ORANI",
    "KAR_PAYLASIM_ORANI",
}

MONTH_PATTERN = (
    r"ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|"
    r"eylul|ekim|kasim|aralik"
)


def _fold_text(value: str) -> str:
    translated = value.translate(str.maketrans({"ı": "i", "İ": "I"}))
    decomposed = unicodedata.normalize("NFKD", translated)
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip().casefold()


def _has_currency(value: str) -> bool:
    folded = _fold_text(value)
    return bool(
        re.search(
            r"(?:\b(?:tl|try|usd|eur)\b|turk lirasi|dolar|euro|avro|₺)",
            folded,
        )
    )


def _has_percent(value: str) -> bool:
    folded = _fold_text(value)
    return "%" in value or "yuzde" in folded


def _parse_plain_integer(value: str) -> int | None:
    match = re.fullmatch(r"\s*(\d{1,3})\s*", value)
    return int(match.group(1)) if match else None


def validate_entity_surface(label: str, value: str) -> str | None:
    """Return a rejection reason when an entity surface has an invalid unit."""
    folded = _fold_text(value)

    if label in MONEY_LABELS and not _has_currency(value):
        return "money_entity_missing_currency"

    if label in PERCENT_LABELS and not _has_percent(value):
        return "percent_entity_missing_marker"

    if label == "TAKSIT_SAYISI":
        if any(
            cue in folded
            for cue in (
                "tl",
                "try",
                "usd",
                "eur",
                "₺",
                "%",
                "yuzde",
                "puan",
                "ay",
                "yil",
                "gun",
            )
        ):
            return "count_entity_has_invalid_unit"
        count = _parse_plain_integer(value)
        if count is None:
            return "count_entity_invalid_format"
        if not 1 <= count <= 120:
            return "count_entity_out_of_range"

    if label == "VADE_SURESI" and not re.fullmatch(
        r"\s*\d{1,3}\s*(?:ay|yil|gun)(?:a|i|e|dir|lik|dan|den)?\s*",
        folded,
    ):
        return "duration_entity_missing_unit"

    if label == "KAMPANYA_TARIH_ARALIGI":
        textual_dates = re.findall(
            rf"\b\d{{1,2}}\s+(?:{MONTH_PATTERN})(?:\s+20\d{{2}})?\b",
            folded,
        )
        numeric_dates = re.findall(
            r"\b\d{1,2}[./-]\d{1,2}[./-]20\d{2}\b",
            folded,
        )
        if len(textual_dates) + len(numeric_dates) < 2:
            return "date_range_incomplete"

    return None
