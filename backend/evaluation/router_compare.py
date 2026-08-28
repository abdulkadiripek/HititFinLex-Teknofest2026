"""Run the checked-in multi-turn set against the deterministic query router.

This runner evaluates routing only. It does not call EVREN, PostgreSQL,
Qdrant, answer generation, or answer validation. Context fixtures represent
verified state produced after a turn; they are never treated as model text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.rag_v2_metrics import (  # noqa: E402
    DATASET_LABEL,
    DatasetValidationError,
    compare_records,
    dataset_records,
    load_scenarios,
)
from rag_v2.identity import normalize_text  # noqa: E402
from rag_v2.models import OfferReference, QueryRoute, SessionState  # noqa: E402
from rag_v2.routing import BANKS, PRODUCTS, QueryRouter  # noqa: E402


DEFAULT_DATASET = Path(__file__).with_name(
    "multiturn_scenarios.silver_unverified.json"
)
ROUTING_METRICS = (
    "standalone_query_accuracy",
    "standalone_context_accuracy",
    "inherited_bank_accuracy",
    "inherited_product_accuracy",
    "inherited_date_accuracy",
    "inherited_scope_accuracy",
    "inheritance_accuracy",
    "topic_change_clear_accuracy",
    "clarification_accuracy",
    "session_isolation_pass_rate",
)
UNEXECUTED_METRICS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "citation_accuracy",
    "numeric_accuracy",
    "unsupported_rejection_rate",
)

BANK_DISPLAY = {
    "Adil Katilim": "Adil Katılım",
    "Albaraka Turk": "Albaraka Türk",
    "Dunya Katilim": "Dünya Katılım",
    "Hayat Finans": "Hayat Finans",
    "Kuveyt Turk": "Kuveyt Türk",
    "T.O.M. Katilim": "T.O.M. Katılım",
    "Emlak Katilim": "Emlak Katılım",
    "Turkiye Finans": "Türkiye Finans",
    "Vakif Katilim": "Vakıf Katılım",
    "Ziraat Katilim": "Ziraat Katılım",
}
PRODUCT_DISPLAY = {
    "KONUT_FINANSMANI": "konut finansmanı",
    "TASIT_FINANSMANI": "taşıt finansmanı",
    "IHTIYAC_FINANSMANI": "ihtiyaç finansmanı",
    "TICARI_FINANSMAN": "KOBİ finansmanı",
    "KATILMA_HESABI": "katılma hesabı",
    "KART": "kredi kartı",
    "KART_KAMPANYASI": "kart kampanyası",
    "YATIRIM_URUNU": "yatırım ürünü",
    "ODEME_TRANSFER": "ödeme transferi",
    "SIGORTA_TEKAFUL": "sigorta/tekafül",
}
FIELD_DISPLAY = {
    "amount": "tutar",
    "rate": "oran",
    "maturity": "vade",
    "fee": "ücret",
    "reward": "ödül",
    "spending_threshold": "harcama eşiği",
    "campaign_date": "kampanya tarihi",
}


def _canonical_bank(value: str) -> str:
    normalized = normalize_text(value)
    for canonical, aliases in BANKS.items():
        candidates = (canonical, BANK_DISPLAY.get(canonical, canonical), *aliases)
        if any(normalize_text(candidate) == normalized for candidate in candidates):
            return canonical
    return value


def _canonical_product(value: str) -> str:
    if value in PRODUCTS:
        return value
    normalized = normalize_text(value)
    for canonical, aliases in PRODUCTS.items():
        candidates = (canonical, PRODUCT_DISPLAY.get(canonical, canonical), *aliases)
        if any(normalize_text(candidate) == normalized for candidate in candidates):
            return canonical
    return value


def _display_banks(values: Sequence[str]) -> list[str]:
    return [BANK_DISPLAY.get(value, value) for value in values]


def _display_products(values: Sequence[str]) -> list[str]:
    return [PRODUCT_DISPLAY.get(value, value) for value in values]


def _display_fields(values: Sequence[str]) -> list[str]:
    return [FIELD_DISPLAY.get(value, value) for value in values]


def _state_from_fixture(
    fixture: Mapping[str, Any],
    route: QueryRoute,
) -> SessionState:
    products = [
        _canonical_product(str(value))
        for value in fixture.get("active_products", route.product_types)
    ]
    banks = [
        _canonical_bank(str(value))
        for value in fixture.get("active_banks", route.banks)
    ]
    active_offer_ids = [str(value) for value in fixture.get("active_offer_ids", [])]
    ranked_offers: list[OfferReference] = []
    for item in fixture.get("ranked_offers", []):
        item_products = [
            _canonical_product(str(value))
            for value in item.get("product_types", products)
        ]
        ranked_offers.append(
            OfferReference(
                offer_id=str(item["offer_id"]),
                bank=_canonical_bank(str(item["bank"])),
                product_types=item_products,
                document_id=item.get("document_id"),
                rank=int(item["rank"]),
            )
        )
    return SessionState(
        active_banks=banks,
        active_products=products,
        active_scope=fixture.get("active_scope", route.scope),
        active_year=fixture.get("active_year"),
        active_date_from=fixture.get("active_date_from"),
        active_date_to=fixture.get("active_date_to"),
        active_offer_ids=active_offer_ids,
        ranked_offers=ranked_offers,
        last_intent=route.intent,
        last_field_types=list(route.field_types),
        last_standalone_query=route.standalone_query,
    )


def _derived_state(route: QueryRoute) -> SessionState:
    return SessionState(
        active_banks=list(route.banks),
        active_products=list(route.product_types),
        active_scope=route.scope,
        active_year=route.year,
        active_date_from=route.date_from,
        active_date_to=route.date_to,
        active_offer_ids=list(route.offer_ids),
        last_intent=route.intent,
        last_field_types=list(route.field_types),
        last_standalone_query=route.standalone_query,
    )


def _cleared_fields(before: SessionState, route: QueryRoute) -> list[str]:
    cleared: list[str] = []
    comparisons = (
        ("banks", before.active_banks, route.banks),
        ("product_types", before.active_products, route.product_types),
        ("scope", before.active_scope, route.scope),
        ("year", before.active_year, route.year),
        ("date_from", before.active_date_from, route.date_from),
        ("date_to", before.active_date_to, route.date_to),
        ("offer_ids", before.active_offer_ids, route.offer_ids),
    )
    for field_name, old_value, new_value in comparisons:
        if field_name == "scope":
            if before.last_standalone_query and old_value != new_value:
                cleared.append(field_name)
            continue
        if isinstance(old_value, list):
            if old_value and set(old_value) != set(new_value):
                cleared.append(field_name)
            continue
        if old_value is not None and old_value != new_value:
            cleared.append(field_name)
    return cleared


def _inherited_context(route: QueryRoute) -> dict[str, Any]:
    output: dict[str, Any] = {}
    inherited = set(route.inherited_fields)
    if "banks" in inherited:
        output["banks"] = _display_banks(route.banks)
    if "product_types" in inherited:
        output["product_types"] = _display_products(route.product_types)
    if "scope" in inherited:
        output["scope"] = route.scope
    if "year" in inherited:
        output["year"] = route.year
    if "date_from" in inherited or "date_range" in inherited:
        output["date_from"] = route.date_from.isoformat() if route.date_from else None
    if "date_to" in inherited or "date_range" in inherited:
        output["date_to"] = route.date_to.isoformat() if route.date_to else None
    if "offer_ids" in inherited:
        output["offer_ids"] = list(route.offer_ids)
    return output


def _serialize_route(route: QueryRoute, before: SessionState) -> dict[str, Any]:
    return {
        "standalone_query": route.standalone_query,
        "intent": route.intent,
        "banks": _display_banks(route.banks),
        "product_types": _display_products(route.product_types),
        "field_types": _display_fields(route.field_types),
        "scope": route.scope,
        "year": route.year,
        "date_from": route.date_from.isoformat() if route.date_from else None,
        "date_to": route.date_to.isoformat() if route.date_to else None,
        "offer_ids": list(route.offer_ids),
        "inherited_fields": list(route.inherited_fields),
        "inherited_context": _inherited_context(route),
        "needs_clarification": route.needs_clarification,
        "clarification_question": route.clarification_question,
        "cleared_fields": _cleared_fields(before, route),
    }


def _legacy_no_memory_output(query: str) -> dict[str, Any]:
    return {
        "standalone_query": query.strip(),
        "inherited_context": {},
        "needs_clarification": False,
        "cleared_fields": [],
    }


def _isolation_route_matches(
    labels: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bool:
    for name in ("banks", "product_types", "offer_ids"):
        expected = sorted(normalize_text(str(value)) for value in labels.get(name, []))
        actual = sorted(normalize_text(str(value)) for value in result.get(name, []))
        if expected != actual:
            return False
    return result.get("year") == labels.get("year")


def build_router_records(
    scenarios: Sequence[Mapping[str, Any]],
    router: QueryRouter | None = None,
) -> list[dict[str, Any]]:
    """Execute every turn while keeping fixture state isolated by session key."""

    resolver = router or QueryRouter()
    records = dataset_records(scenarios)
    indexed = {record["record_id"]: record for record in records}
    for scenario in scenarios:
        session_states: dict[str, SessionState] = {}
        for turn in scenario["turns"]:
            session_key = str(turn.get("session_key") or "default")
            before = session_states.get(session_key, SessionState())
            route = resolver.resolve(str(turn["user_query"]), state=before)
            record_id = f"{scenario['scenario_id']}/{turn['turn_id']}"
            indexed[record_id]["legacy"] = _legacy_no_memory_output(
                str(turn["user_query"])
            )
            indexed[record_id]["v2"] = _serialize_route(route, before)
            labels = indexed[record_id]["labels"]
            if isinstance(labels.get("isolation_expected"), bool):
                indexed[record_id]["legacy"]["isolation_passed"] = True
                indexed[record_id]["v2"]["isolation_passed"] = (
                    _isolation_route_matches(labels, indexed[record_id]["v2"])
                )
            fixture = turn.get("context_after_turn")
            session_states[session_key] = (
                _state_from_fixture(fixture, route)
                if isinstance(fixture, Mapping)
                else _derived_state(route)
            )
    return records


def _metric_subset(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "legacy_no_memory": {
            name: deepcopy(report["legacy"][name])
            for name in ROUTING_METRICS
            if name in report["legacy"]
        },
        "v2_deterministic_router": {
            name: deepcopy(report["v2"][name])
            for name in ROUTING_METRICS
            if name in report["v2"]
        },
        "delta_v2_minus_legacy": {
            name: deepcopy(report["delta_v2_minus_legacy"][name])
            for name in ROUTING_METRICS
            if name in report["delta_v2_minus_legacy"]
        },
    }


def _standalone_context_metric(
    records: Sequence[Mapping[str, Any]],
    provider: str,
) -> dict[str, Any]:
    evaluated = [
        record
        for record in records
        if record["labels"].get("needs_clarification") is False
    ]
    matched: list[str] = []
    for record in evaluated:
        labels = record["labels"]
        result = record[provider]
        text = normalize_text(str(result.get("standalone_query") or ""))
        required = [
            *map(str, labels.get("banks", [])),
            *map(str, labels.get("product_types", [])),
            *map(str, labels.get("field_types", [])),
        ]
        year = labels.get("year")
        date_from = labels.get("date_from")
        date_to = labels.get("date_to")
        if year is not None:
            required.append(str(year))
        if date_from:
            required.append(str(date_from))
        if date_to:
            required.append(str(date_to))
        scope = labels.get("scope")
        if scope == "current":
            required.append("guncel")
        elif scope == "historical" and not (year or date_from or date_to):
            required.append("gecmis")
        elif scope == "all":
            required.append("tum donem")
        if all(normalize_text(value) in text for value in required):
            matched.append(record["record_id"])
    value = len(matched) / len(evaluated) if evaluated else 0.0
    return {
        "status": "available",
        "value": round(value, 6),
        "evaluated_records": len(evaluated),
        "record_ids": [record["record_id"] for record in evaluated],
        "matched_records": matched,
        "definition": (
            "Non-clarification standalone queries must contain every labeled "
            "bank, product, field, period boundary, and scope marker."
        ),
    }


def run_router_comparison(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    dataset_path: Path | None = None,
    include_records: bool = False,
) -> dict[str, Any]:
    records = build_router_records(scenarios)
    paired_report = compare_records(records)
    metrics = _metric_subset(paired_report)
    legacy_context = _standalone_context_metric(records, "legacy")
    v2_context = _standalone_context_metric(records, "v2")
    metrics["legacy_no_memory"]["standalone_context_accuracy"] = legacy_context
    metrics["v2_deterministic_router"]["standalone_context_accuracy"] = v2_context
    metrics["delta_v2_minus_legacy"]["standalone_context_accuracy"] = {
        "status": "available",
        "value": round(v2_context["value"] - legacy_context["value"], 6),
        "evaluated_records": legacy_context["evaluated_records"],
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_label": DATASET_LABEL,
        "evaluation_kind": "deterministic_multi_turn_routing",
        "providers": {
            "legacy_no_memory": (
                "Raw user query with no structured server-side conversation state."
            ),
            "v2_deterministic_router": (
                "rag_v2.routing.QueryRouter with per-session verified state fixtures."
            ),
        },
        "dataset": {
            "scenario_count": len(scenarios),
            "turn_count": len(records),
            "label_status": DATASET_LABEL,
            "sha256": (
                hashlib.sha256(dataset_path.read_bytes()).hexdigest()
                if dataset_path is not None
                else None
            ),
        },
        "metrics": metrics,
        "unexecuted_metrics": {
            name: {
                "status": "unavailable",
                "reason": "router_only_run_does_not_execute_retrieval_or_answers",
            }
            for name in UNEXECUTED_METRICS
        },
        "summary": {
            "status": "measured_silver_unverified",
            "comparison_policy": "same_complete_turn_set",
            "record_count": len(records),
            "claim": (
                "Scores describe this silver routing set only; no production "
                "quality or retrieval improvement claim is made."
            ),
        },
    }
    if include_records:
        report["records"] = records
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a stateless baseline and the deterministic V2 router on "
            "the shared silver_unverified multi-turn set."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-records", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scenarios = load_scenarios(args.dataset)
        report = run_router_comparison(
            scenarios,
            dataset_path=args.dataset,
            include_records=args.include_records,
        )
    except (OSError, json.JSONDecodeError, DatasetValidationError, ValueError) as exc:
        error = {"status": "error", "error": str(exc)}
        print(json.dumps(error, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
