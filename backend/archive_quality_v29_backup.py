from __future__ import annotations

from typing import Any

from archive_common_v28 import fold_text


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


def deduplicate_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decision_rank = {"accepted": 2, "review": 1, "rejected": 0}
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for fact in facts:
        key = (str(fact["fact_type"]), fold_text(str(fact["fact_text"])))
        current = best.get(key)
        if current is None or (
            decision_rank[str(fact["decision"])],
            float(fact["confidence"]),
        ) > (
            decision_rank[str(current["decision"])],
            float(current["confidence"]),
        ):
            best[key] = fact
    return list(best.values())
