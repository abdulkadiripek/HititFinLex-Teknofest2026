from __future__ import annotations

import json
import re
from typing import Any

from archive_common_v28 import fold_text


def date_range_base(value: str) -> str:
    folded = fold_text(value)
    folded = re.sub(r"\b20\d{2}\b", "", folded)
    folded = re.sub(r"\s*(?:-|\u2013|\u2014)\s*", "-", folded)
    return re.sub(r"\s+", " ", folded).strip()


def metadata_supports_product(classification: dict[str, Any]) -> bool:
    product_label = str(classification.get("product_type", {}).get("label") or "")
    rule = classification.get("strong_rule") or {}
    reason = str(rule.get("reason") or "")
    return (
        str(rule.get("label") or "") == product_label
        and reason.startswith(("url:", "url_advisory:", "title:"))
    )


def classification_quality(
    classification: dict[str, Any],
    min_product_confidence: float,
) -> str:
    if str(classification.get("decision") or "").upper() != "ACCEPTED":
        return "review"
    product = classification.get("product_type") or {}
    score = float(product.get("score") or 0.0)
    if score >= min_product_confidence or metadata_supports_product(classification):
        return "accepted"
    return "review"


def fact_identity(fact: dict[str, Any]) -> tuple[str, str]:
    fact_type = str(fact["fact_type"])
    normalized = fact.get("normalized_value")
    if normalized is not None and fact_type != "KAMPANYA_TARIH_ARALIGI":
        return (
            fact_type,
            json.dumps(
                normalized,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    folded = fold_text(str(fact["fact_text"]))
    if fact_type == "KAMPANYA_TARIH_ARALIGI":
        folded = re.sub(r"\s*(?:-|\u2013|\u2014)\s*", "-", folded)
    if fact_type == "MEVDUAT_GUVENCESI" and folded in {
        "tmsf",
        "tasarruf mevduati sigorta fonu",
    }:
        folded = "tmsf"
    return fact_type, folded


def deduplicate_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decision_rank = {"accepted": 2, "review": 1, "rejected": 0}
    best: dict[tuple[str, str], dict[str, Any]] = {}
    dated_bases_with_year = {
        date_range_base(str(fact["fact_text"]))
        for fact in facts
        if str(fact["fact_type"]) == "KAMPANYA_TARIH_ARALIGI"
        and re.search(r"\b20\d{2}\b", str(fact["fact_text"]))
    }
    evidence_with_specific_branch = {
        fold_text(str(fact.get("evidence_text") or ""))
        for fact in facts
        if str(fact["fact_type"]) == "BASVURU_KANALI"
        and fold_text(str(fact["fact_text"]))
        in {
            "mobil sube",
            "internet subesi",
            "internet sube",
        }
    }
    for fact in facts:
        if (
            str(fact["fact_type"]) == "BASVURU_KANALI"
            and fold_text(str(fact["fact_text"])) == "sube"
        ):
            folded_evidence = fold_text(
                str(fact.get("evidence_text") or "")
            )
            without_specific = re.sub(
                r"\b(?:mobil sube|internet subesi|internet sube)\b",
                " ",
                folded_evidence,
            )
            if (
                folded_evidence in evidence_with_specific_branch
                and not re.search(r"\bsube\b", without_specific)
            ):
                continue
        if (
            str(fact["fact_type"]) == "KAMPANYA_TARIH_ARALIGI"
            and not re.search(r"\b20\d{2}\b", str(fact["fact_text"]))
            and date_range_base(str(fact["fact_text"]))
            in dated_bases_with_year
        ):
            continue
        key = fact_identity(fact)
        current = best.get(key)
        if current is None or (
            decision_rank[str(fact["decision"])],
            float(fact["confidence"]),
            len(str(fact["fact_text"])),
        ) > (
            decision_rank[str(current["decision"])],
            float(current["confidence"]),
            len(str(current["fact_text"])),
        ):
            best[key] = fact
    return list(best.values())
