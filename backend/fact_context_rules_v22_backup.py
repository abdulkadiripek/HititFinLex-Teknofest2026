from __future__ import annotations

import re
import unicodedata


def fold_text(value: str) -> str:
    translated = value.translate(str.maketrans({"ı": "i", "İ": "I"}))
    decomposed = unicodedata.normalize("NFKD", translated)
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip().casefold()


def excluded_context_reason(label: str, evidence: str) -> str | None:
    folded = fold_text(evidence)

    if label == "VADE_SURESI":
        if any(
            cue in folded
            for cue in ("erken odeme", "tazminat", "kalan vade")
        ):
            return "excluded_early_payment_context"

        first_payment_cues = (
            "ilk taksit",
            "lk taksit",
            "ilk odeme",
            "lk odeme",
            "odeme tarihi",
        )
        delayed_payment_phrase = (
            "en gec" in folded
            and "gun sonrasina kadar" in folded
        )
        if any(cue in folded for cue in first_payment_cues) or (
            delayed_payment_phrase
        ):
            return "excluded_first_payment_timing_context"

        membership_cues = (
            "sistemde kalan",
            "sistemde kaldiktan",
            "devlet katkisi",
            "emeklilik sistemi",
            "56 yas",
        )
        if "yil" in folded and any(
            cue in folded for cue in membership_cues
        ):
            return "excluded_membership_duration_context"

    if label == "FINANSMAN_TUTARI":
        if any(
            cue in folded
            for cue in (
                "deger",
                "bedel",
                "fiyat",
                "rayic",
                "satis tutari",
                "fatura tutari",
            )
        ):
            return "conflicting_property_value_context"

        document_cues = (
            "gelir belgesi",
            "gelir beyani",
            "belge beyani",
        )
        threshold_cues = (
            "asmayan",
            "asmasi",
            "muaf",
            "talep edilmeyecek",
            "istenmeyecek",
        )
        if any(cue in folded for cue in document_cues) and any(
            cue in folded for cue in threshold_cues
        ):
            return "excluded_document_requirement_threshold_context"

    return None
