"""Reproducible same-set retrieval comparison for the live corpus.

This module is read-only. It never mutates PostgreSQL, Qdrant, or the corpus.
All provider failures are reduced to fixed error codes so connection details
and credentials cannot enter reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.rag_v2_metrics import DATASET_LABEL, compare_records  # noqa: E402
from rag_v2.identity import canonicalize_url  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CASES = Path(__file__).with_name("retrieval_cases.silver_unverified.json")
DEFAULT_CORPUS = PROJECT_DIR / "data" / "ara" / "dokumanlar.jsonl"
RETRIEVAL_METRICS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_10",
)
INTENTS = {"lookup", "compare", "list", "calculate", "historical", "clarification"}
SCOPES = {"current", "historical", "all"}


class RetrievalAssetError(ValueError):
    """Raised for an invalid fixed evaluation asset."""


class RetrieverUnavailable(RuntimeError):
    """A sanitized retriever failure safe to include in a report."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class SourceLabel:
    record_key: str
    bank_key: str
    page_title: str
    source_url: str


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    case_id: str
    query: str
    route: dict[str, Any]
    relevant_record_keys: tuple[str, ...]
    relevance_grades: dict[str, float]
    source_labels: tuple[SourceLabel, ...]


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    record_key: str
    bank_key: str
    page_title: str
    source_url: str


class RetrievalBackend(Protocol):
    name: str

    def retrieve(self, case: RetrievalCase, top_k: int) -> list[str]:
        ...

    def close(self) -> None:
        ...


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RetrievalAssetError(message)


def _string_list(value: Any, *, allow_empty: bool = True) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _validate_route(route: Any, case_id: str) -> dict[str, Any]:
    _ensure(isinstance(route, dict), f"{case_id}: route must be an object")
    required = {
        "intent",
        "banks",
        "product_types",
        "field_types",
        "scope",
        "year",
        "date_from",
        "date_to",
        "offer_ids",
    }
    missing = required.difference(route)
    _ensure(not missing, f"{case_id}: route is missing {sorted(missing)}")
    _ensure(route["intent"] in INTENTS, f"{case_id}: invalid route intent")
    _ensure(route["scope"] in SCOPES, f"{case_id}: invalid route scope")
    for name in ("banks", "product_types", "field_types", "offer_ids"):
        _ensure(_string_list(route[name]), f"{case_id}: route {name} must be a string list")
    _ensure(
        route["year"] is None
        or (
            isinstance(route["year"], int)
            and not isinstance(route["year"], bool)
            and 1900 <= route["year"] <= 2100
        ),
        f"{case_id}: route year must be null or an integer from 1900 to 2100",
    )
    for name in ("date_from", "date_to"):
        value = route[name]
        _ensure(
            value is None
            or (
                isinstance(value, str)
                and len(value) == 10
                and value[4] == "-"
                and value[7] == "-"
            ),
            f"{case_id}: route {name} must be null or YYYY-MM-DD",
        )
    return dict(route)


def load_retrieval_cases(path: str | Path, minimum_count: int = 30) -> tuple[list[RetrievalCase], str]:
    """Load and validate fixed silver retrieval cases."""

    asset_path = Path(path)
    raw = asset_path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    _ensure(isinstance(payload, dict), "retrieval dataset must be an object")
    _ensure(payload.get("dataset_label") == DATASET_LABEL, f"dataset_label must be {DATASET_LABEL}")
    cases_payload = payload.get("cases")
    _ensure(isinstance(cases_payload, list), "cases must be a list")
    _ensure(len(cases_payload) >= minimum_count, f"at least {minimum_count} retrieval cases are required")
    case_ids: set[str] = set()
    cases: list[RetrievalCase] = []

    for item in cases_payload:
        _ensure(isinstance(item, dict), "each retrieval case must be an object")
        case_id = item.get("case_id")
        _ensure(isinstance(case_id, str) and bool(case_id), "case_id must be non-empty")
        _ensure(case_id not in case_ids, f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        _ensure(item.get("label") == DATASET_LABEL, f"{case_id}: label must be exactly {DATASET_LABEL}")
        query = item.get("query")
        _ensure(isinstance(query, str) and bool(query.strip()), f"{case_id}: query must be non-empty")
        route = _validate_route(item.get("route"), case_id)
        relevant = item.get("relevant_record_keys")
        _ensure(
            _string_list(relevant, allow_empty=False) and len(relevant) == len(set(relevant)),
            f"{case_id}: relevant_record_keys must be a unique non-empty string list",
        )
        grades = item.get("relevance_grades")
        _ensure(isinstance(grades, dict), f"{case_id}: relevance_grades must be an object")
        _ensure(set(grades) == set(relevant), f"{case_id}: relevance grade ids must equal relevant ids")
        _ensure(
            all(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 for value in grades.values()),
            f"{case_id}: relevance grades must be positive numbers",
        )
        labels_payload = item.get("source_labels")
        _ensure(isinstance(labels_payload, list), f"{case_id}: source_labels must be a list")
        labels: list[SourceLabel] = []
        for source in labels_payload:
            _ensure(isinstance(source, dict), f"{case_id}: source label must be an object")
            values = [source.get(name) for name in ("record_key", "bank_key", "page_title", "source_url")]
            _ensure(all(isinstance(value, str) and bool(value.strip()) for value in values), f"{case_id}: source label values must be non-empty")
            labels.append(
                SourceLabel(
                    record_key=source["record_key"],
                    bank_key=source["bank_key"],
                    page_title=source["page_title"],
                    source_url=source["source_url"],
                )
            )
        _ensure(
            len(labels) == len(relevant)
            and {label.record_key for label in labels} == set(relevant),
            f"{case_id}: source label ids must equal relevant ids",
        )
        cases.append(
            RetrievalCase(
                case_id=case_id,
                query=query.strip(),
                route=route,
                relevant_record_keys=tuple(relevant),
                relevance_grades={key: float(value) for key, value in grades.items()},
                source_labels=tuple(labels),
            )
        )
    return cases, hashlib.sha256(raw).hexdigest()


class CorpusManifest:
    def __init__(self, records: Sequence[CorpusRecord], sha256: str) -> None:
        self.records_by_key = {record.record_key: record for record in records}
        self.record_count = len(records)
        self.keys_by_url: dict[str, str] = {}
        for record in records:
            url = canonicalize_url(record.source_url)
            if url in self.keys_by_url and self.keys_by_url[url] != record.record_key:
                raise RetrievalAssetError("corpus contains duplicate canonical source URLs")
            self.keys_by_url[url] = record.record_key
        self.sha256 = sha256

    def resolve_url(self, source_url: str | None) -> str:
        if not source_url:
            return "unmapped:missing-url"
        canonical = canonicalize_url(source_url)
        record_key = self.keys_by_url.get(canonical)
        if record_key:
            return record_key
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"unmapped:{digest}"


def load_corpus_manifest(path: str | Path) -> CorpusManifest:
    corpus_path = Path(path)
    digest = hashlib.sha256()
    records: list[CorpusRecord] = []
    seen: set[str] = set()
    with corpus_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RetrievalAssetError(f"corpus line {line_number} is invalid JSON") from error
            values = [item.get(name) for name in ("kayit_id", "banka_key", "sayfa_basligi", "kaynak_url")]
            if not all(isinstance(value, str) and bool(value.strip()) for value in values):
                continue
            record_key = item["kayit_id"]
            _ensure(record_key not in seen, f"duplicate corpus record key: {record_key}")
            seen.add(record_key)
            records.append(
                CorpusRecord(
                    record_key=record_key,
                    bank_key=item["banka_key"],
                    page_title=item["sayfa_basligi"],
                    source_url=item["kaynak_url"],
                )
            )
    _ensure(bool(records), "corpus manifest contains no usable records")
    return CorpusManifest(records, digest.hexdigest())


def validate_cases_against_corpus(cases: Sequence[RetrievalCase], corpus: CorpusManifest) -> None:
    """Require every silver label to match the checked-in corpus metadata."""

    for case in cases:
        for expected in case.source_labels:
            actual = corpus.records_by_key.get(expected.record_key)
            _ensure(actual is not None, f"{case.case_id}: corpus record is missing: {expected.record_key}")
            _ensure(actual.bank_key == expected.bank_key, f"{case.case_id}: corpus bank label changed")
            _ensure(actual.page_title == expected.page_title, f"{case.case_id}: corpus title label changed")
            _ensure(
                canonicalize_url(actual.source_url) == canonicalize_url(expected.source_url),
                f"{case.case_id}: corpus source URL label changed",
            )


class LegacyLiveBackend:
    name = "legacy"

    def __init__(self, corpus: CorpusManifest) -> None:
        self.corpus = corpus
        self._module: Any = None
        self._connection_context: Any = None
        self._connection: Any = None
        self._model: Any = None
        self._text_column: str | None = None
        try:
            import hybrid_search

            self._module = hybrid_search
            self._connection_context = hybrid_search.get_connection()
            self._connection = self._connection_context.__enter__()
            self._text_column = hybrid_search.inspect_chunk_table(self._connection)
            self._model = hybrid_search.load_model()
        except Exception as error:
            self.close()
            raise RetrieverUnavailable("legacy_database_or_model_unavailable") from error

    def retrieve(self, case: RetrievalCase, top_k: int) -> list[str]:
        try:
            vector = self._module.encode_query(self._model, case.query)
            lexical = self._module.build_lexical_query(case.query)
            rows = self._module.search_database(
                self._connection,
                self._text_column,
                vector,
                lexical,
                top_k,
            )
        except Exception as error:
            raise RetrieverUnavailable("legacy_retrieval_unavailable") from error
        return [self.corpus.resolve_url(row[3]) for row in rows]

    def close(self) -> None:
        if self._connection_context is not None:
            try:
                self._connection_context.__exit__(None, None, None)
            except Exception:
                pass
            self._connection_context = None
        if self._module is not None:
            try:
                self._module.close_connection_pool()
            except Exception:
                pass


class RagV2LiveBackend:
    name = "v2"

    def __init__(self, corpus: CorpusManifest) -> None:
        self.corpus = corpus
        self._pool: Any = None
        self._evren: Any = None
        self._qdrant: Any = None
        self._retriever: Any = None
        try:
            from rag_v2.database import RagDatabasePool
            from rag_v2.providers import EvrenClient, QdrantRestClient
            from rag_v2.retrieval import HybridRetriever
            from rag_v2.settings import RagV2Settings

            settings = RagV2Settings.from_env()
            if not settings.evren_ready or not settings.qdrant_ready:
                raise RetrieverUnavailable("rag_v2_remote_configuration_missing")
            self._pool = RagDatabasePool(settings)
            self._pool.open()
            with self._pool.connection() as connection:
                row = connection.execute(
                    "SELECT to_regclass('public.rag_chunks') AS relation"
                ).fetchone()
                if not row or row["relation"] is None:
                    raise RetrieverUnavailable("rag_v2_index_table_missing")
                count = connection.execute("SELECT COUNT(*) AS count FROM rag_chunks").fetchone()
                if not count or int(count["count"]) <= 0:
                    raise RetrieverUnavailable("rag_v2_index_empty")
            self._evren = EvrenClient(settings)
            self._qdrant = QdrantRestClient(settings)
            self._retriever = HybridRetriever(self._pool, settings, self._evren, self._qdrant)
        except RetrieverUnavailable:
            self.close()
            raise
        except Exception as error:
            self.close()
            raise RetrieverUnavailable("rag_v2_database_or_configuration_unavailable") from error

    def retrieve(self, case: RetrievalCase, top_k: int) -> list[str]:
        try:
            from rag_v2.models import QueryRoute

            route = QueryRoute.model_validate(
                {
                    "standalone_query": case.query,
                    **case.route,
                    "inherited_fields": [],
                    "needs_clarification": False,
                    "clarification_question": None,
                }
            )
            records, diagnostics, issues = self._retriever.retrieve(route, top_k, use_reranker=False)
        except Exception as error:
            raise RetrieverUnavailable("rag_v2_retrieval_unavailable") from error
        if "dense_provider_unavailable" in issues or int(diagnostics.get("dense_candidates", 0)) <= 0:
            raise RetrieverUnavailable("rag_v2_dense_retrieval_unavailable")
        return [self.corpus.resolve_url(record.source_url) for record in records]

    def close(self) -> None:
        for client_name in ("_qdrant", "_evren"):
            client = getattr(self, client_name, None)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
                setattr(self, client_name, None)
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                pass
            self._pool = None


def _provider_results(
    backend: RetrievalBackend | None,
    cases: Sequence[RetrievalCase],
    top_k: int,
    unavailable_code: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if backend is None:
        return {}, {"status": "unavailable", "reason": unavailable_code or "backend_not_created"}
    results: dict[str, dict[str, Any]] = {}
    try:
        for case in cases:
            ids = backend.retrieve(case, top_k)
            results[case.case_id] = {"retrieved_ids": list(dict.fromkeys(map(str, ids)))[:top_k]}
    except RetrieverUnavailable as error:
        return results, {"status": "unavailable", "reason": error.code}
    except Exception:
        return results, {"status": "unavailable", "reason": "backend_execution_failed"}
    finally:
        backend.close()
    return results, {"status": "available", "evaluated_cases": len(results)}


def _filtered_metric_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full = compare_records(records)
    return {
        "legacy": {name: full["legacy"][name] for name in RETRIEVAL_METRICS},
        "v2": {name: full["v2"][name] for name in RETRIEVAL_METRICS},
        "delta_v2_minus_legacy": {
            name: full["delta_v2_minus_legacy"][name] for name in RETRIEVAL_METRICS
        },
    }


def compare_with_backends(
    cases: Sequence[RetrievalCase],
    corpus: CorpusManifest,
    legacy_backend: RetrievalBackend | None,
    v2_backend: RetrievalBackend | None,
    *,
    top_k: int = 10,
    dataset_sha256: str = "",
    legacy_unavailable_code: str | None = None,
    v2_unavailable_code: str | None = None,
) -> dict[str, Any]:
    """Run both retrievers on the exact same ordered case objects."""

    if top_k < 10:
        for backend in (legacy_backend, v2_backend):
            if backend is not None:
                backend.close()
        raise RetrievalAssetError("top_k must be at least 10 for the required metrics")
    legacy_results, legacy_status = _provider_results(
        legacy_backend, cases, top_k, legacy_unavailable_code
    )
    v2_results, v2_status = _provider_results(v2_backend, cases, top_k, v2_unavailable_code)
    metric_records: list[dict[str, Any]] = []
    rendered_cases: list[dict[str, Any]] = []
    for case in cases:
        legacy = legacy_results.get(case.case_id)
        v2 = v2_results.get(case.case_id)
        metric_records.append(
            {
                "record_id": case.case_id,
                "labels": {
                    "relevant_ids": list(case.relevant_record_keys),
                    "relevance_grades": dict(case.relevance_grades),
                },
                "legacy": legacy,
                "v2": v2,
            }
        )
        rendered_cases.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "relevant_record_keys": list(case.relevant_record_keys),
                "legacy_retrieved_record_keys": legacy["retrieved_ids"] if legacy else None,
                "v2_retrieved_record_keys": v2["retrieved_ids"] if v2 else None,
            }
        )
    metrics = _filtered_metric_report(metric_records)
    measured = all(
        metrics["delta_v2_minus_legacy"][name]["status"] == "available"
        for name in RETRIEVAL_METRICS
    )
    return {
        "schema_version": "1.0",
        "dataset_label": DATASET_LABEL,
        "run_type": "same_set_live_retrieval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured" if measured else "unavailable",
        "comparison_policy": "identical_ordered_cases_and_complete_paired_outputs",
        "dataset_sha256": dataset_sha256,
        "corpus_sha256": corpus.sha256,
        "corpus_record_count": corpus.record_count,
        "case_count": len(cases),
        "top_k": top_k,
        "v2_use_reranker": False,
        "providers": {"legacy": legacy_status, "v2": v2_status},
        "metrics": metrics,
        "cases": rendered_cases,
        "claim": "No improvement claim is made; labels are silver_unverified.",
    }


def _unavailable_before_backends(
    cases: Sequence[RetrievalCase], dataset_sha256: str, reason: str
) -> dict[str, Any]:
    empty_corpus = CorpusManifest([], "")
    return compare_with_backends(
        cases,
        empty_corpus,
        None,
        None,
        dataset_sha256=dataset_sha256,
        legacy_unavailable_code=reason,
        v2_unavailable_code=reason,
    )


def run_live_comparison(
    cases_path: str | Path = DEFAULT_CASES,
    corpus_path: str | Path = DEFAULT_CORPUS,
    *,
    top_k: int = 10,
    validate_only: bool = False,
) -> dict[str, Any]:
    if top_k < 10:
        raise RetrievalAssetError("top_k must be at least 10 for the required metrics")
    cases, dataset_sha256 = load_retrieval_cases(cases_path)
    try:
        corpus = load_corpus_manifest(corpus_path)
        validate_cases_against_corpus(cases, corpus)
    except (OSError, RetrievalAssetError, json.JSONDecodeError):
        return _unavailable_before_backends(cases, dataset_sha256, "corpus_missing_or_label_mismatch")
    if validate_only:
        return compare_with_backends(
            cases,
            corpus,
            None,
            None,
            top_k=top_k,
            dataset_sha256=dataset_sha256,
            legacy_unavailable_code="validate_only",
            v2_unavailable_code="validate_only",
        )

    load_dotenv(PROJECT_DIR / ".env")
    v2: RetrievalBackend | None = None
    try:
        v2 = RagV2LiveBackend(corpus)
    except RetrieverUnavailable as error:
        return compare_with_backends(
            cases,
            corpus,
            None,
            None,
            top_k=top_k,
            dataset_sha256=dataset_sha256,
            legacy_unavailable_code="paired_run_skipped_v2_unavailable",
            v2_unavailable_code=error.code,
        )
    legacy: RetrievalBackend | None = None
    try:
        legacy = LegacyLiveBackend(corpus)
    except RetrieverUnavailable as error:
        v2.close()
        return compare_with_backends(
            cases,
            corpus,
            None,
            None,
            top_k=top_k,
            dataset_sha256=dataset_sha256,
            legacy_unavailable_code=error.code,
            v2_unavailable_code="paired_run_skipped_legacy_unavailable",
        )
    return compare_with_backends(
        cases,
        corpus,
        legacy,
        v2,
        top_k=top_k,
        dataset_sha256=dataset_sha256,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run legacy and RAG V2 retrieval on the same silver_unverified corpus cases."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_live_comparison(
            args.cases,
            args.corpus,
            top_k=args.top_k,
            validate_only=args.validate_only,
        )
    except (OSError, json.JSONDecodeError, RetrievalAssetError) as error:
        payload = {"status": "error", "error": str(error)}
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
