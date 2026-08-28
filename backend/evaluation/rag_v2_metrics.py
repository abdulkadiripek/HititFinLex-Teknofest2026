"""Deterministic, provider-neutral metrics for paired RAG evaluations.

The module never calls a model, database, or remote service. Legacy and V2
results are evaluated only when both are present for every labeled record of
a metric. This prevents comparisons over different subsets.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DATASET_LABEL = "silver_unverified"
INTENTS = {"lookup", "compare", "list", "calculate", "historical", "clarification"}
SCOPES = {"current", "historical", "all"}
INHERITABLE_FIELDS = {
    "banks",
    "product_types",
    "scope",
    "year",
    "date_from",
    "date_to",
    "offer_ids",
}
REQUIRED_EXPECTED_KEYS = {
    "standalone_query",
    "intent",
    "banks",
    "product_types",
    "field_types",
    "scope",
    "year",
    "date_from",
    "date_to",
    "offer_ids",
    "inherited_fields",
    "needs_clarification",
    "clarification_question",
    "topic_change_cleared_fields",
}
PROVIDERS = ("legacy", "v2")
_MISSING = object()


class DatasetValidationError(ValueError):
    """Raised when an evaluation asset violates the checked schema."""


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetValidationError(message)


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def _validate_date(value: Any, field_name: str, record_id: str) -> None:
    if value is None:
        return
    _ensure(isinstance(value, str), f"{record_id}: {field_name} must be a string or null")
    _ensure(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None,
        f"{record_id}: {field_name} must use YYYY-MM-DD",
    )
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise DatasetValidationError(
            f"{record_id}: {field_name} must be a valid calendar date"
        ) from error


def validate_scenarios(scenarios: Any, minimum_count: int = 30) -> list[dict[str, Any]]:
    """Validate and return a multi-turn silver evaluation dataset."""

    _ensure(isinstance(scenarios, list), "dataset must be a list of scenarios")
    _ensure(len(scenarios) >= minimum_count, f"dataset must contain at least {minimum_count} scenarios")
    scenario_ids: set[str] = set()
    record_ids: set[str] = set()

    for scenario in scenarios:
        _ensure(isinstance(scenario, dict), "each scenario must be an object")
        scenario_id = scenario.get("scenario_id")
        _ensure(isinstance(scenario_id, str) and scenario_id, "scenario_id must be a non-empty string")
        _ensure(scenario_id not in scenario_ids, f"duplicate scenario_id: {scenario_id}")
        scenario_ids.add(scenario_id)
        _ensure(
            scenario.get("label") == DATASET_LABEL,
            f"{scenario_id}: label must be exactly {DATASET_LABEL}",
        )
        _ensure(
            _is_string_list(scenario.get("categories")) and bool(scenario.get("categories")),
            f"{scenario_id}: categories must contain at least one non-empty string",
        )
        turns = scenario.get("turns")
        _ensure(isinstance(turns, list) and len(turns) >= 2, f"{scenario_id}: at least two turns are required")
        turn_ids: set[str] = set()

        for turn in turns:
            _ensure(isinstance(turn, dict), f"{scenario_id}: every turn must be an object")
            turn_id = turn.get("turn_id")
            _ensure(isinstance(turn_id, str) and turn_id, f"{scenario_id}: turn_id must be non-empty")
            _ensure(turn_id not in turn_ids, f"{scenario_id}: duplicate turn_id {turn_id}")
            turn_ids.add(turn_id)
            record_id = f"{scenario_id}/{turn_id}"
            _ensure(record_id not in record_ids, f"duplicate record_id: {record_id}")
            record_ids.add(record_id)
            _ensure(
                isinstance(turn.get("user_query"), str) and turn["user_query"].strip(),
                f"{record_id}: user_query must be non-empty",
            )
            if "session_key" in turn:
                _ensure(
                    isinstance(turn["session_key"], str)
                    and bool(turn["session_key"].strip()),
                    f"{record_id}: session_key must be a non-empty string",
                )
            if "simulated_untrusted_assistant_answer" in turn:
                _ensure(
                    isinstance(turn["simulated_untrusted_assistant_answer"], str),
                    f"{record_id}: simulated assistant answer must be a string",
                )
            if "requires_fresh_retrieval" in turn:
                _ensure(
                    isinstance(turn["requires_fresh_retrieval"], bool),
                    f"{record_id}: requires_fresh_retrieval must be boolean",
                )
            expected = turn.get("expected")
            _ensure(isinstance(expected, dict), f"{record_id}: expected must be an object")
            missing = REQUIRED_EXPECTED_KEYS.difference(expected)
            _ensure(not missing, f"{record_id}: missing expected keys: {sorted(missing)}")
            _ensure(
                isinstance(expected["standalone_query"], str) and expected["standalone_query"].strip(),
                f"{record_id}: standalone_query must be non-empty",
            )
            _ensure(expected["intent"] in INTENTS, f"{record_id}: invalid intent")
            for field_name in ("banks", "product_types", "field_types", "offer_ids", "inherited_fields"):
                _ensure(_is_string_list(expected[field_name]), f"{record_id}: {field_name} must be a string list")
            _ensure(
                set(expected["inherited_fields"]).issubset(INHERITABLE_FIELDS),
                f"{record_id}: inherited_fields contains an unsupported field",
            )
            _ensure(expected["scope"] in SCOPES, f"{record_id}: invalid scope")
            _ensure(
                expected["year"] is None
                or (
                    isinstance(expected["year"], int)
                    and not isinstance(expected["year"], bool)
                    and 1900 <= expected["year"] <= 2100
                ),
                f"{record_id}: year must be an integer or null",
            )
            _validate_date(expected["date_from"], "date_from", record_id)
            _validate_date(expected["date_to"], "date_to", record_id)
            if expected["date_from"] and expected["date_to"]:
                _ensure(
                    expected["date_from"] <= expected["date_to"],
                    f"{record_id}: date_from must not be after date_to",
                )
            _ensure(
                isinstance(expected["needs_clarification"], bool),
                f"{record_id}: needs_clarification must be boolean",
            )
            clarification = expected["clarification_question"]
            if expected["needs_clarification"]:
                _ensure(
                    isinstance(clarification, str) and clarification.strip(),
                    f"{record_id}: clarification_question is required",
                )
            else:
                _ensure(clarification is None, f"{record_id}: clarification_question must be null")
            _ensure(
                _is_string_list(expected["topic_change_cleared_fields"]),
                f"{record_id}: topic_change_cleared_fields must be a string list",
            )
            for optional_label in ("relevant_ids", "relevance_grades", "citation", "numbers"):
                if optional_label in expected:
                    _validate_optional_label(expected[optional_label], optional_label, record_id)
            _validate_label_relationships(expected, record_id)
            if "should_reject" in expected:
                _ensure(isinstance(expected["should_reject"], bool), f"{record_id}: should_reject must be boolean")
            if "isolation_expected" in expected:
                _ensure(
                    isinstance(expected["isolation_expected"], bool),
                    f"{record_id}: isolation_expected must be boolean",
                )
            context = turn.get("context_after_turn")
            if context is not None:
                _validate_context_fixture(context, record_id)

    return scenarios


def _validate_context_fixture(value: Any, record_id: str) -> None:
    _ensure(
        isinstance(value, dict),
        f"{record_id}: context_after_turn must be an object",
    )
    allowed = {
        "active_banks",
        "active_products",
        "active_scope",
        "active_year",
        "active_date_from",
        "active_date_to",
        "active_offer_ids",
        "ranked_offers",
    }
    _ensure(
        set(value).issubset(allowed),
        f"{record_id}: context_after_turn contains unsupported fields",
    )
    for field_name in (
        "active_banks",
        "active_products",
        "active_offer_ids",
    ):
        if field_name in value:
            _ensure(
                _is_string_list(value[field_name]),
                f"{record_id}: {field_name} must be a string list",
            )
    scope = value.get("active_scope")
    if scope is not None:
        _ensure(scope in SCOPES, f"{record_id}: invalid active_scope")
    year = value.get("active_year")
    if year is not None:
        _ensure(
            isinstance(year, int)
            and not isinstance(year, bool)
            and 1900 <= year <= 2100,
            f"{record_id}: active_year must be an integer or null",
        )
    for field_name in ("active_date_from", "active_date_to"):
        if field_name in value:
            _validate_date(value[field_name], field_name, record_id)
    if value.get("active_date_from") and value.get("active_date_to"):
        _ensure(
            value["active_date_from"] <= value["active_date_to"],
            f"{record_id}: active date range is reversed",
        )
    ranked_offers = value.get("ranked_offers", [])
    _ensure(
        isinstance(ranked_offers, list),
        f"{record_id}: ranked_offers must be a list",
    )
    seen_ranks: set[int] = set()
    seen_offer_ids: set[str] = set()
    for item in ranked_offers:
        _ensure(
            isinstance(item, dict),
            f"{record_id}: every ranked offer must be an object",
        )
        offer_id = item.get("offer_id")
        bank = item.get("bank")
        rank = item.get("rank")
        _ensure(
            isinstance(offer_id, str) and bool(offer_id),
            f"{record_id}: ranked offer_id must be non-empty",
        )
        _ensure(
            isinstance(bank, str) and bool(bank),
            f"{record_id}: ranked bank must be non-empty",
        )
        _ensure(
            isinstance(rank, int) and not isinstance(rank, bool) and rank > 0,
            f"{record_id}: ranked offer rank must be a positive integer",
        )
        _ensure(
            rank not in seen_ranks and offer_id not in seen_offer_ids,
            f"{record_id}: ranked offers must have unique ranks and ids",
        )
        seen_ranks.add(rank)
        seen_offer_ids.add(offer_id)


def _validate_optional_label(value: Any, label_name: str, record_id: str) -> None:
    if value is None:
        return
    if label_name == "relevant_ids":
        _ensure(_is_string_list(value), f"{record_id}: relevant_ids must be a string list or null")
        return
    if label_name == "relevance_grades":
        _ensure(isinstance(value, dict), f"{record_id}: relevance_grades must be an object or null")
        _ensure(
            all(isinstance(key, str) and key and isinstance(grade, (int, float)) and grade >= 0 for key, grade in value.items()),
            f"{record_id}: relevance_grades must map ids to non-negative numbers",
        )
        return
    _ensure(isinstance(value, dict), f"{record_id}: {label_name} must be an object or null")
    _ensure(
        _is_string_list(value.get("required_ids" if label_name == "citation" else "required")),
        f"{record_id}: {label_name} required values must be a string list",
    )
    _ensure(
        _is_string_list(value.get("allowed_ids" if label_name == "citation" else "allowed")),
        f"{record_id}: {label_name} allowed values must be a string list",
    )


def _validate_label_relationships(labels: Mapping[str, Any], record_id: str) -> None:
    relevant_ids = labels.get("relevant_ids")
    relevance_grades = labels.get("relevance_grades")
    if isinstance(relevance_grades, Mapping):
        _ensure(
            isinstance(relevant_ids, list),
            f"{record_id}: relevance_grades requires relevant_ids",
        )
        _ensure(
            set(map(str, relevance_grades)).issubset(set(map(str, relevant_ids))),
            f"{record_id}: relevance_grades ids must be present in relevant_ids",
        )
    citation = labels.get("citation")
    if isinstance(citation, Mapping):
        _ensure(
            set(map(str, citation["required_ids"])).issubset(set(map(str, citation["allowed_ids"]))),
            f"{record_id}: required citation ids must be allowed",
        )
    numbers = labels.get("numbers")
    if isinstance(numbers, Mapping):
        _ensure(
            {_normalize_number(value) for value in numbers["required"]}.issubset(
                {_normalize_number(value) for value in numbers["allowed"]}
            ),
            f"{record_id}: required numbers must be allowed",
        )


def load_scenarios(path: str | Path, minimum_count: int = 30) -> list[dict[str, Any]]:
    """Load a JSON or JSONL scenario file and apply the strict schema."""

    dataset_path = Path(path)
    text = dataset_path.read_text(encoding="utf-8-sig")
    if dataset_path.suffix.lower() == ".jsonl":
        scenarios = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            _ensure(payload.get("dataset_label") == DATASET_LABEL, f"dataset_label must be {DATASET_LABEL}")
            scenarios = payload.get("scenarios")
        else:
            scenarios = payload
    return validate_scenarios(scenarios, minimum_count=minimum_count)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("I", "\u0131").replace("\u0130", "i").casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _normalize_sequence(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set)):
        return ()
    return tuple(sorted({_normalize_text(value) for value in values if _normalize_text(value)}))


def _normalize_number(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = text.replace("\u20ba", " tl ").replace("\u0131", "i").replace("\u00fc", "u").replace("%", "% ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _deduplicate(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key and key not in seen:
            output.append(key)
            seen.add(key)
    return output


def _path_get(payload: Any, *paths: Sequence[str]) -> Any:
    if not isinstance(payload, Mapping):
        return _MISSING
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = _MISSING
                break
            current = current[key]
        if current is not _MISSING:
            return current
    return _MISSING


def _expected_to_labels(expected: Mapping[str, Any]) -> dict[str, Any]:
    labels = deepcopy(dict(expected))
    inherited_context: dict[str, Any] = {}
    for field_name in expected.get("inherited_fields", []):
        inherited_context[field_name] = deepcopy(expected.get(field_name))
    labels["inherited_context"] = inherited_context
    labels["cleared_fields"] = deepcopy(expected.get("topic_change_cleared_fields", []))
    return labels


def dataset_records(scenarios: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        for turn in scenario["turns"]:
            records.append(
                {
                    "record_id": f"{scenario['scenario_id']}/{turn['turn_id']}",
                    "scenario_id": scenario["scenario_id"],
                    "turn_id": turn["turn_id"],
                    "labels": _expected_to_labels(turn["expected"]),
                    "legacy": None,
                    "v2": None,
                }
            )
    return records


def load_comparison(path: str | Path) -> dict[str, Any]:
    """Load provider outputs and optional common labels from JSON or JSONL."""

    comparison_path = Path(path)
    text = comparison_path.read_text(encoding="utf-8-sig")
    if comparison_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        payload: dict[str, Any] = {"records": records, "dataset_label": DATASET_LABEL}
    else:
        raw = json.loads(text)
        _ensure(isinstance(raw, dict), "comparison input must be an object")
        payload = raw
    _ensure(payload.get("dataset_label") == DATASET_LABEL, f"comparison dataset_label must be {DATASET_LABEL}")
    _ensure(isinstance(payload.get("records"), list), "comparison records must be a list")
    seen: set[str] = set()
    for item in payload["records"]:
        _ensure(isinstance(item, dict), "each comparison record must be an object")
        record_id = item.get("record_id")
        _ensure(isinstance(record_id, str) and record_id, "comparison record_id must be non-empty")
        _ensure(record_id not in seen, f"duplicate comparison record_id: {record_id}")
        seen.add(record_id)
        for provider in PROVIDERS:
            _ensure(
                item.get(provider) is None or isinstance(item.get(provider), dict),
                f"{record_id}: {provider} must be an object or null",
            )
        _ensure(item.get("labels") is None or isinstance(item.get("labels"), dict), f"{record_id}: labels must be an object")
        labels = item.get("labels") or {}
        for optional_label in ("relevant_ids", "relevance_grades", "citation", "numbers"):
            if optional_label in labels:
                _validate_optional_label(labels[optional_label], optional_label, record_id)
        for boolean_label in ("should_reject", "isolation_expected", "needs_clarification"):
            if boolean_label in labels:
                _ensure(isinstance(labels[boolean_label], bool), f"{record_id}: {boolean_label} must be boolean")
        _validate_label_relationships(labels, record_id)
    return payload


def _merge_labels(base: dict[str, Any], extra: Mapping[str, Any], record_id: str) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in extra.items():
        if key in merged and merged[key] is not None and value is not None and merged[key] != value:
            raise DatasetValidationError(f"{record_id}: comparison label conflicts with the scenario dataset: {key}")
        merged[key] = deepcopy(value)
    return merged


def merge_dataset_and_comparison(
    scenarios: Sequence[Mapping[str, Any]], comparison: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Merge outputs by record id while retaining one shared label object."""

    records = dataset_records(scenarios)
    if comparison is None:
        return records
    indexed = {item["record_id"]: item for item in comparison.get("records", [])}
    known = {record["record_id"] for record in records}
    unknown = sorted(set(indexed).difference(known))
    _ensure(not unknown, f"comparison contains record ids outside the dataset: {unknown[:5]}")
    for record in records:
        supplied = indexed.get(record["record_id"])
        if supplied is None:
            continue
        if supplied.get("labels"):
            record["labels"] = _merge_labels(record["labels"], supplied["labels"], record["record_id"])
        record["legacy"] = deepcopy(supplied.get("legacy"))
        record["v2"] = deepcopy(supplied.get("v2"))
    return records


def _available(value: float, count: int, record_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "status": "available",
        "value": round(float(value), 6),
        "evaluated_records": count,
        "record_ids": list(record_ids),
    }


def _unavailable(reason: str, labeled_count: int = 0, missing_record_ids: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "value": None,
        "evaluated_records": 0,
        "labeled_records": labeled_count,
        "missing_paired_record_ids": list(missing_record_ids),
        "reason": reason,
    }


def _paired_labeled_records(
    records: Sequence[Mapping[str, Any]], label_predicate: Callable[[Mapping[str, Any]], bool]
) -> tuple[list[Mapping[str, Any]], dict[str, Any] | None]:
    labeled = [record for record in records if label_predicate(record["labels"])]
    if not labeled:
        return [], _unavailable("required labels are absent")
    missing = [
        record["record_id"]
        for record in labeled
        if not isinstance(record.get("legacy"), Mapping) or not isinstance(record.get("v2"), Mapping)
    ]
    if missing:
        return [], _unavailable(
            "paired legacy and v2 outputs are required for every labeled record",
            labeled_count=len(labeled),
            missing_record_ids=missing,
        )
    return labeled, None


def _result_value(result: Mapping[str, Any], field_name: str) -> Any:
    paths: dict[str, tuple[tuple[str, ...], ...]] = {
        "standalone_query": (("standalone_query",), ("route", "standalone_query")),
        "inherited_context": (("inherited_context",), ("route", "inherited_context")),
        "cleared_fields": (
            ("cleared_fields",),
            ("topic_change_cleared_fields",),
            ("diagnostics", "cleared_fields"),
            ("route", "topic_change_cleared_fields"),
        ),
        "needs_clarification": (("needs_clarification",), ("route", "needs_clarification")),
        "retrieved_ids": (("retrieved_ids",), ("diagnostics", "retrieved_ids")),
        "cited_ids": (("cited_ids",),),
        "answer_numbers": (("answer_numbers",),),
        "should_reject": (("rejected",),),
        "isolation_passed": (("isolation_passed",), ("diagnostics", "isolation_passed")),
    }
    return _path_get(result, *paths[field_name])


def _metric_accuracy(
    records: Sequence[Mapping[str, Any]],
    label_predicate: Callable[[Mapping[str, Any]], bool],
    scorer: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
) -> dict[str, dict[str, Any]]:
    paired, error = _paired_labeled_records(records, label_predicate)
    if error:
        return {provider: deepcopy(error) for provider in PROVIDERS}
    output: dict[str, dict[str, Any]] = {}
    ids = [record["record_id"] for record in paired]
    for provider in PROVIDERS:
        scores = [1.0 if scorer(record["labels"], record[provider]) else 0.0 for record in paired]
        output[provider] = _available(sum(scores) / len(scores), len(scores), ids)
    return output


def _score_standalone(labels: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    actual = _result_value(result, "standalone_query")
    return actual is not _MISSING and _normalize_text(actual) == _normalize_text(labels["standalone_query"])


def _score_inherited_field(labels: Mapping[str, Any], result: Mapping[str, Any], fields: Sequence[str]) -> bool:
    expected = labels.get("inherited_context", {})
    actual = _result_value(result, "inherited_context")
    if not isinstance(actual, Mapping):
        return False
    for field_name in fields:
        expected_value = expected.get(field_name, _MISSING)
        if expected_value is _MISSING:
            continue
        actual_value = actual.get(field_name, _MISSING)
        if isinstance(expected_value, list):
            if _normalize_sequence(actual_value) != _normalize_sequence(expected_value):
                return False
        elif isinstance(expected_value, str):
            if _normalize_text(actual_value) != _normalize_text(expected_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _has_inherited(labels: Mapping[str, Any], fields: Sequence[str]) -> bool:
    inherited = labels.get("inherited_context")
    return isinstance(inherited, Mapping) and any(field_name in inherited for field_name in fields)


def _score_topic_clear(labels: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    actual = _result_value(result, "cleared_fields")
    return actual is not _MISSING and _normalize_sequence(actual) == _normalize_sequence(labels["cleared_fields"])


def _score_clarification(labels: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    actual = _result_value(result, "needs_clarification")
    if actual is _MISSING:
        actual = result.get("status") == "needs_clarification"
    return isinstance(actual, bool) and actual is labels["needs_clarification"]


def _retrieved_ids(result: Mapping[str, Any]) -> list[str]:
    explicit = _result_value(result, "retrieved_ids")
    if isinstance(explicit, list):
        return _deduplicate(explicit)
    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        return []
    ids: list[str] = []
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        value = item.get("retrieval_id", item.get("offer_id", item.get("document_id", item.get("chunk_id"))))
        if value is not None:
            ids.append(str(value))
    return _deduplicate(ids)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = set(_deduplicate(relevant_ids))
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    hits = relevant.intersection(_deduplicate(retrieved_ids)[:k])
    return len(hits) / len(relevant)


def reciprocal_rank_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 10) -> float:
    relevant = set(_deduplicate(relevant_ids))
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    for rank, item_id in enumerate(_deduplicate(retrieved_ids)[:k], start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 10, relevance_grades: Mapping[str, Any] | None = None
) -> float:
    relevant = set(_deduplicate(relevant_ids))
    if not relevant:
        raise ValueError("relevant_ids must not be empty")
    grades = {item_id: 1.0 for item_id in relevant}
    if relevance_grades:
        grades.update(
            {
                str(item_id): float(value)
                for item_id, value in relevance_grades.items()
                if str(item_id) in relevant and float(value) >= 0
            }
        )

    def gain(grade: float, rank: int) -> float:
        return (2.0**grade - 1.0) / math.log2(rank + 1)

    ranked = _deduplicate(retrieved_ids)[:k]
    dcg = sum(gain(grades.get(item_id, 0.0), rank) for rank, item_id in enumerate(ranked, start=1))
    ideal = sorted((grades.get(item_id, 0.0) for item_id in relevant), reverse=True)[:k]
    idcg = sum(gain(grade, rank) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def _retrieval_metric(
    records: Sequence[Mapping[str, Any]], scorer: Callable[[Sequence[str], Sequence[str], Mapping[str, Any]], float]
) -> dict[str, dict[str, Any]]:
    paired, error = _paired_labeled_records(
        records,
        lambda labels: isinstance(labels.get("relevant_ids"), list) and bool(labels["relevant_ids"]),
    )
    if error:
        return {provider: deepcopy(error) for provider in PROVIDERS}
    output: dict[str, dict[str, Any]] = {}
    ids = [record["record_id"] for record in paired]
    for provider in PROVIDERS:
        scores = [
            scorer(_retrieved_ids(record[provider]), record["labels"]["relevant_ids"], record["labels"])
            for record in paired
        ]
        output[provider] = _available(sum(scores) / len(scores), len(scores), ids)
    return output


def _resolved_cited_ids(result: Mapping[str, Any]) -> list[str]:
    explicit = _result_value(result, "cited_ids")
    if isinstance(explicit, list):
        return _deduplicate(explicit)
    citations = result.get("citations")
    citation_map = result.get("citation_map")
    if isinstance(citations, list):
        if isinstance(citation_map, Mapping):
            return _deduplicate(citation_map.get(str(citation), f"unknown:{citation}") for citation in citations)
        return _deduplicate(citations)
    answer = result.get("answer")
    if not isinstance(answer, str):
        return []
    source_refs = re.findall(r"\[(S\d+)\]", answer, flags=re.IGNORECASE)
    if not source_refs:
        return []
    evidence = result.get("evidence")
    evidence_map: dict[str, str] = {}
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, Mapping) or item.get("source_id") is None:
                continue
            reference = str(item["source_id"]).strip().strip("[]").upper()
            value = item.get("retrieval_id", item.get("offer_id", item.get("document_id", item.get("chunk_id"))))
            if value is not None:
                evidence_map[reference] = str(value)
    return _deduplicate(evidence_map.get(reference.upper(), f"unknown:{reference.upper()}") for reference in source_refs)


def _score_citations(labels: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    citation = labels["citation"]
    required = set(map(str, citation["required_ids"]))
    allowed = set(map(str, citation["allowed_ids"]))
    actual = set(_resolved_cited_ids(result))
    return required.issubset(actual) and actual.issubset(allowed)


_NUMBER_PATTERN = re.compile(
    r"(?<![\w])(?:%\s*)?\d+(?:[.,]\d+)*(?:\s*(?:tl|try|ay|yil|gun|hafta|oran|puan))?",
    flags=re.IGNORECASE,
)


def _answer_numbers(result: Mapping[str, Any]) -> list[str]:
    explicit = _result_value(result, "answer_numbers")
    if isinstance(explicit, list):
        return _deduplicate(_normalize_number(value) for value in explicit)
    answer = result.get("answer")
    if not isinstance(answer, str):
        return []
    without_citations = re.sub(r"\[(?:S)?\d+\]", " ", answer, flags=re.IGNORECASE)
    without_citations = without_citations.replace("\u0131", "i").replace("\u00fc", "u")
    return _deduplicate(_normalize_number(match.group(0)) for match in _NUMBER_PATTERN.finditer(without_citations))


def _score_numbers(labels: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    numeric = labels["numbers"]
    required = {_normalize_number(value) for value in numeric["required"]}
    allowed = {_normalize_number(value) for value in numeric["allowed"]}
    actual = set(_answer_numbers(result))
    return required.issubset(actual) and actual.issubset(allowed)


def _score_rejection(labels: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    explicit = _result_value(result, "should_reject")
    if isinstance(explicit, bool):
        return explicit
    return result.get("status") in {"rejected", "insufficient_evidence"}


def _score_isolation(labels: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    actual = _result_value(result, "isolation_passed")
    return isinstance(actual, bool) and actual is labels["isolation_expected"]


def _attach_deltas(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "legacy": {},
        "v2": {},
        "delta_v2_minus_legacy": {},
    }
    for metric_name, values in metrics.items():
        legacy = values["legacy"]
        v2 = values["v2"]
        report["legacy"][metric_name] = legacy
        report["v2"][metric_name] = v2
        if legacy["status"] == "available" and v2["status"] == "available":
            report["delta_v2_minus_legacy"][metric_name] = {
                "status": "available",
                "value": round(v2["value"] - legacy["value"], 6),
                "evaluated_records": legacy["evaluated_records"],
            }
        else:
            report["delta_v2_minus_legacy"][metric_name] = _unavailable(
                "both paired provider metrics must be available before a delta is reported"
            )
    return report


def compare_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute a paired legacy/V2 report from merged evaluation records."""

    metrics: dict[str, dict[str, Any]] = {}
    metrics["standalone_query_accuracy"] = _metric_accuracy(
        records, lambda labels: isinstance(labels.get("standalone_query"), str), _score_standalone
    )
    metrics["inherited_bank_accuracy"] = _metric_accuracy(
        records,
        lambda labels: _has_inherited(labels, ("banks",)),
        lambda labels, result: _score_inherited_field(labels, result, ("banks",)),
    )
    metrics["inherited_product_accuracy"] = _metric_accuracy(
        records,
        lambda labels: _has_inherited(labels, ("product_types",)),
        lambda labels, result: _score_inherited_field(labels, result, ("product_types",)),
    )
    date_fields = ("year", "date_from", "date_to")
    metrics["inherited_date_accuracy"] = _metric_accuracy(
        records,
        lambda labels: _has_inherited(labels, date_fields),
        lambda labels, result: _score_inherited_field(labels, result, date_fields),
    )
    metrics["inherited_scope_accuracy"] = _metric_accuracy(
        records,
        lambda labels: _has_inherited(labels, ("scope",)),
        lambda labels, result: _score_inherited_field(labels, result, ("scope",)),
    )
    metrics["inheritance_accuracy"] = _metric_accuracy(
        records,
        lambda labels: bool(labels.get("inherited_context")),
        lambda labels, result: _score_inherited_field(
            labels, result, tuple(labels.get("inherited_context", {}).keys())
        ),
    )
    metrics["topic_change_clear_accuracy"] = _metric_accuracy(
        records,
        lambda labels: isinstance(labels.get("cleared_fields"), list),
        _score_topic_clear,
    )
    metrics["clarification_accuracy"] = _metric_accuracy(
        records, lambda labels: isinstance(labels.get("needs_clarification"), bool), _score_clarification
    )
    for k in (1, 3, 5, 10):
        metrics[f"recall_at_{k}"] = _retrieval_metric(
            records, lambda retrieved, relevant, _labels, cutoff=k: recall_at_k(retrieved, relevant, cutoff)
        )
    metrics["mrr_at_10"] = _retrieval_metric(
        records, lambda retrieved, relevant, _labels: reciprocal_rank_at_k(retrieved, relevant, 10)
    )
    metrics["ndcg_at_10"] = _retrieval_metric(
        records,
        lambda retrieved, relevant, labels: ndcg_at_k(
            retrieved, relevant, 10, labels.get("relevance_grades")
        ),
    )
    metrics["citation_accuracy"] = _metric_accuracy(
        records, lambda labels: isinstance(labels.get("citation"), Mapping), _score_citations
    )
    metrics["numeric_accuracy"] = _metric_accuracy(
        records, lambda labels: isinstance(labels.get("numbers"), Mapping), _score_numbers
    )
    metrics["unsupported_rejection_rate"] = _metric_accuracy(
        records, lambda labels: labels.get("should_reject") is True, _score_rejection
    )
    metrics["session_isolation_pass_rate"] = _metric_accuracy(
        records, lambda labels: isinstance(labels.get("isolation_expected"), bool), _score_isolation
    )
    report = _attach_deltas(metrics)
    available = sum(1 for value in report["delta_v2_minus_legacy"].values() if value["status"] == "available")
    report["summary"] = {
        "dataset_label": DATASET_LABEL,
        "record_count": len(records),
        "comparison_policy": "paired_complete_records_only",
        "available_metric_deltas": available,
        "status": "measured" if available else "unavailable",
        "claim": "No improvement claim is made by this utility.",
    }
    return report
