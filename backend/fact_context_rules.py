from __future__ import annotations

import re
import unicodedata


AUTO_THRESHOLDS = {
    "EKSPERTIZ_UCRETI": 0.55,
    "FINANSMAN_TUTARI": 0.75,
    "HARCAMA_ESIGI": 0.85,
    "HARCAMA_UST_LIMITI": 0.70,
    "INDIRIM_ORANI": 0.90,
    "INDIRIM_TUTARI": 0.90,
    "IPOTEK_TESIS_UCRETI": 0.55,
    "ISLEM_ALT_LIMITI": 0.85,
    "ISLEM_UST_LIMITI": 0.85,
    "KAMPANYA_TARIH_ARALIGI": 0.90,
    "KAR_PAYI_ORANI": 0.80,
    "KAR_PAYLASIM_ORANI": 0.95,
    "MINIMUM_BAKIYE": 0.85,
    "ODUL_TUTARI": 0.85,
    "TAHSIS_UCRETI": 0.55,
    "TAKSIT_SAYISI": 0.90,
    "VADE_SURESI": 0.85,
}


def fold_text(value: str) -> str:
    translated = value.translate(str.maketrans({"ı": "i", "İ": "I"}))
    decomposed = unicodedata.normalize("NFKD", translated)
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip().casefold()


def campaign_date_context_pass(evidence: str) -> bool:
    folded = fold_text(evidence)
    has_date_cue = any(
        cue in folded
        for cue in (
            "kampanya",
            "ampanya araligi",
            "tarih",
            "gecerli",
        )
    )
    return has_date_cue and bool(re.search(r"\d", folded))


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

        if "taahhut" in folded and "uzat" in folded:
            return "excluded_commitment_extension_context"

        if any(
            cue in folded for cue in ("oturan", "ikamet")
        ) and any(
            cue in folded for cue in ("en az", "suresi", "boyunca")
        ):
            return "excluded_residency_duration_context"

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
                "odenecek toplam tutar",
                "toplam geri odeme",
                "toplam geri odenen",
            )
        ):
            return "excluded_total_repayment_context"

        if any(
            cue in folded
            for cue in ("arasi vade", "araliginda vade")
        ):
            return "excluded_asset_value_tier_context"

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
