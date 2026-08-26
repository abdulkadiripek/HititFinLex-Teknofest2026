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


MONEY_TOKEN_PATTERN = re.compile(
    r"\b\d[\d. ]*(?:,\d+)?\s*(?:tl|try|turk lirasi|usd|eur)\b"
)
PERCENT_TOKEN_PATTERN = re.compile(
    r"(?:%\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*%)"
)


def _token_boundaries(
    folded: str,
    start: int,
    end: int,
    pattern: re.Pattern[str],
) -> tuple[int, int]:
    left = max(0, start - 100)
    right = min(len(folded), end + 130)
    for match in pattern.finditer(folded):
        if match.end() <= start:
            left = max(left, match.end())
        elif match.start() >= end:
            right = min(right, match.start())
            break
    return left, right


def _cue_distance(
    folded: str,
    start: int,
    end: int,
    cues: tuple[str, ...],
    radius: int = 130,
) -> int | None:
    best = None
    window_start = max(0, start - radius)
    window_end = min(len(folded), end + radius)
    window = folded[window_start:window_end]
    for cue in cues:
        cursor = 0
        while True:
            position = window.find(cue, cursor)
            if position < 0:
                break
            absolute = window_start + position
            if absolute >= end:
                distance = absolute - end
            elif absolute + len(cue) <= start:
                distance = start - (absolute + len(cue))
            else:
                distance = 0
            best = distance if best is None else min(best, distance)
            cursor = position + 1
    return best


def campaign_amount_role_at(
    evidence: str,
    start: int,
    end: int,
) -> str | None:
    folded = fold_text(evidence)
    left, right = _token_boundaries(
        folded,
        start,
        end,
        MONEY_TOKEN_PATTERN,
    )
    before = folded[left:start]
    after = folded[end:right]
    around = folded[left:right]

    threshold_after = bool(
        re.search(
            r"^\s*(?:['\u2019]?(?:ye|ya|e|a))?\s*"
            r"(?:ve\s+)?(?:uzeri|ustu|uzerindeyse|ustundeyse)\b",
            after,
        )
    )
    amount_followed_by_spend = bool(
        re.search(
            r"^\s*(?:['\u2019]?(?:ye|ya|e|a))?\s*"
            r"(?:(?:deger|tutar)inde\s+)?"
            r"(?:tek\s+seferde\s+)?(?:harcama|alisveris)\b",
            after[:70],
        )
    )
    threshold_before = any(
        cue in before[-65:]
        for cue in ("en az", "minimum", "alt limit", "harcama esigi")
    ) and any(cue in around for cue in ("harcama", "alisveris"))
    if threshold_after or amount_followed_by_spend or threshold_before:
        return "spend_threshold"

    reward_cues = (
        "worldpuan",
        "world puan",
        "wordpuan",
        "iade",
        "hediye",
        "odul",
        "bonus",
        "promosyon",
        "chip para",
        "puan kazan",
        "puan hediye",
    )
    reward_after = any(cue in after[:100] for cue in reward_cues)
    reward_before = any(cue in before[-65:] for cue in reward_cues)
    rewarded_spend_position = bool(
        re.search(r"(?:harcamaya|alisverise)\s*$", before[-55:])
    )
    future = folded[end : min(len(folded), end + 220)]
    aggregate_reward = (
        bool(re.match(r"^\s*,?\s*toplam(?:da)?\b", after))
        and any(cue in future for cue in reward_cues)
    )
    tier_reward = (
        bool(
            re.search(
                r"(?:kadarsa|arasindaysa|uzerindeyse|ustundeyse)\s*$",
                before[-55:],
            )
        )
        and any(cue in future for cue in reward_cues)
    )
    if (
        reward_after
        or reward_before
        or rewarded_spend_position
        or aggregate_reward
        or tier_reward
    ):
        return "reward_amount"

    if "indirim" in after[:65] or "indirim" in before[-55:]:
        return "discount_amount"
    return None


def campaign_amount_roles(value: str, evidence: str) -> set[str]:
    folded_evidence = fold_text(evidence)
    folded_value = fold_text(value)
    if not folded_value:
        return set()

    roles = set()
    cursor = 0
    while True:
        start = folded_evidence.find(folded_value, cursor)
        if start < 0:
            break
        end = start + len(folded_value)
        role = campaign_amount_role_at(folded_evidence, start, end)
        if role is not None:
            roles.add(role)
        cursor = max(end, start + 1)
    return roles


def campaign_percent_role_at(
    evidence: str,
    start: int,
    end: int,
) -> str | None:
    folded = fold_text(evidence)
    discount_distance = _cue_distance(
        folded,
        start,
        end,
        ("indirim", "iade"),
    )
    finance_distance = _cue_distance(
        folded,
        start,
        end,
        (
            "kar payi",
            "kar orani",
            "finansman orani",
            "oranla",
            "oranlarla",
            "orandan",
        ),
    )
    if finance_distance is not None and (
        discount_distance is None or finance_distance < discount_distance
    ):
        return "finance_rate"
    if discount_distance is not None:
        return "discount_percent"
    return None


def campaign_percent_roles(value: str, evidence: str) -> set[str]:
    folded_evidence = fold_text(evidence)
    folded_value = fold_text(value)
    if not folded_value:
        return set()

    roles = set()
    cursor = 0
    while True:
        start = folded_evidence.find(folded_value, cursor)
        if start < 0:
            break
        end = start + len(folded_value)
        role = campaign_percent_role_at(folded_evidence, start, end)
        if role is not None:
            roles.add(role)
        cursor = max(end, start + 1)
    return roles


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


def excluded_context_reason(
    label: str,
    evidence: str,
    value: str = "",
) -> str | None:
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
            for cue in ("abonman", "police", "sigorta limiti")
        ) and any(
            cue in folded
            for cue in ("maksimum limit", "teminat", "gemi", "kamyon")
        ):
            return "excluded_insurance_policy_limit_context"

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

    if value and label in {"HARCAMA_ESIGI", "ODUL_TUTARI"}:
        roles = campaign_amount_roles(value, evidence)
        if (
            label == "HARCAMA_ESIGI"
            and "spend_threshold" not in roles
            and roles & {"reward_amount", "discount_amount"}
        ):
            return "excluded_reward_amount_context"
        if (
            label == "ODUL_TUTARI"
            and "reward_amount" not in roles
            and "spend_threshold" in roles
        ):
            return "excluded_spend_threshold_context"

    if value and label == "INDIRIM_TUTARI":
        roles = campaign_amount_roles(value, evidence)
        if "discount_amount" not in roles and roles & {
            "reward_amount",
            "spend_threshold",
        }:
            return "excluded_non_discount_amount_context"

    if value and label in {"INDIRIM_ORANI", "KAR_PAYI_ORANI"}:
        roles = campaign_percent_roles(value, evidence)
        if label == "INDIRIM_ORANI" and (
            "discount_percent" not in roles and "finance_rate" in roles
        ):
            return "excluded_finance_rate_context"
        if label == "KAR_PAYI_ORANI" and (
            "finance_rate" not in roles and "discount_percent" in roles
        ):
            return "excluded_discount_rate_context"

    if label == "ODUL_TUTARI" and any(
        cue in folded
        for cue in (
            "para cek",
            "nakit cek",
            "atm'den cek",
            "atm den cek",
        )
    ) and any(cue in folded for cue in ("ucretsiz", "kadar", "limit")):
        return "excluded_withdrawal_limit_context"

    return None
