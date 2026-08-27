from __future__ import annotations

import argparse
import json
from datetime import date
from typing import Any

from dotenv import load_dotenv

from archive_common_v28 import canonicalize_url, open_connection
from classifier_service import canonicalize_product_label, product_label_variants
from hybrid_search import (
    RRF_CONSTANT,
    build_lexical_query,
    encode_query,
    load_model,
)


def archive_search_ready(connection) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                to_regclass('public.historical_documents') IS NOT NULL,
                to_regclass('public.historical_document_chunks') IS NOT NULL
            """
        )
        documents_ready, chunks_ready = cursor.fetchone()
        return bool(documents_ready and chunks_ready)


def search_historical_database(
    connection,
    query_vector,
    lexical_query: str,
    top_k: int,
    *,
    bank_names: list[str] | None = None,
    product_types: list[str] | None = None,
    has_facts: bool | None = None,
    min_confidence: float = 0.0,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[tuple]:
    if not archive_search_ready(connection):
        return []

    filters = ["d.searchable IS TRUE", "d.quality_status = 'accepted'"]
    filter_parameters: list[Any] = []
    if bank_names:
        filters.append("b.bank_name = ANY(%s)")
        filter_parameters.append(bank_names)
    if product_types:
        filters.append("d.product_type_code = ANY(%s)")
        filter_parameters.append(
            list(
                dict.fromkeys(
                    variant
                    for code in product_types
                    for variant in product_label_variants(code)
                )
            )
        )
    if has_facts is not None:
        predicate = "EXISTS" if has_facts else "NOT EXISTS"
        filters.append(
            f"{predicate} ("
            "SELECT 1 FROM historical_facts hf "
            "WHERE hf.historical_document_id = d.id "
            "AND hf.decision = 'accepted'"
            ")"
        )
    if min_confidence > 0:
        filters.append("COALESCE(d.classification_confidence, 0) >= %s")
        filter_parameters.append(min_confidence)
    if date_from:
        filters.append("d.snapshot_date >= %s")
        filter_parameters.append(date_from)
    if date_to:
        filters.append("d.snapshot_date <= %s")
        filter_parameters.append(date_to)

    candidate_count = min(max(top_k * 12, 60), 600)
    query = f"""
        WITH query_input AS (
            SELECT
                %s::vector AS query_vector,
                websearch_to_tsquery('simple', %s)
                    || websearch_to_tsquery('turkish', %s) AS text_query
        ),
        eligible_chunks AS (
            SELECT
                c.id,
                c.historical_document_id,
                c.content,
                c.embedding,
                c.search_vector,
                d.archive_key,
                d.page_title,
                d.source_url,
                d.archive_url,
                d.snapshot_date,
                d.product_type_code,
                d.verified,
                b.bank_name
            FROM historical_document_chunks c
            JOIN historical_documents d
                ON d.id = c.historical_document_id
            JOIN banks b ON b.id = d.bank_id
            WHERE {' AND '.join(filters)}
        ),
        semantic_candidates AS (
            SELECT
                e.id,
                1 - (e.embedding <=> q.query_vector) AS semantic_similarity,
                ROW_NUMBER() OVER (
                    ORDER BY e.embedding <=> q.query_vector
                ) AS semantic_rank
            FROM eligible_chunks e
            CROSS JOIN query_input q
            WHERE e.embedding IS NOT NULL
            ORDER BY e.embedding <=> q.query_vector
            LIMIT %s
        ),
        lexical_candidates AS (
            SELECT
                e.id,
                ts_rank_cd(e.search_vector, q.text_query) AS lexical_score,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(e.search_vector, q.text_query) DESC
                ) AS lexical_rank
            FROM eligible_chunks e
            CROSS JOIN query_input q
            WHERE e.search_vector @@ q.text_query
            ORDER BY lexical_score DESC
            LIMIT %s
        ),
        candidate_ids AS (
            SELECT id FROM semantic_candidates
            UNION
            SELECT id FROM lexical_candidates
        ),
        fused AS (
            SELECT
                ids.id,
                semantic.semantic_similarity,
                lexical.lexical_score,
                COALESCE(1.0 / (%s + semantic.semantic_rank), 0.0)
                    + COALESCE(1.0 / (%s + lexical.lexical_rank), 0.0)
                    AS hybrid_score
            FROM candidate_ids ids
            LEFT JOIN semantic_candidates semantic ON semantic.id = ids.id
            LEFT JOIN lexical_candidates lexical ON lexical.id = ids.id
        ),
        ranked_documents AS (
            SELECT
                e.id AS chunk_id,
                e.historical_document_id,
                e.archive_key,
                e.bank_name,
                e.page_title,
                e.source_url,
                e.archive_url,
                e.snapshot_date,
                e.product_type_code,
                e.verified,
                e.content,
                fused.semantic_similarity,
                fused.lexical_score,
                fused.hybrid_score,
                ROW_NUMBER() OVER (
                    PARTITION BY e.historical_document_id
                    ORDER BY fused.hybrid_score DESC
                ) AS document_rank
            FROM fused
            JOIN eligible_chunks e ON e.id = fused.id
        )
        SELECT
            chunk_id,
            historical_document_id,
            archive_key,
            bank_name,
            page_title,
            source_url,
            archive_url,
            snapshot_date,
            product_type_code,
            content,
            semantic_similarity,
            lexical_score,
            hybrid_score,
            verified
        FROM ranked_documents
        WHERE document_rank = 1
        ORDER BY hybrid_score DESC
        LIMIT %s
    """
    parameters = (
        query_vector,
        lexical_query,
        lexical_query,
        *filter_parameters,
        candidate_count,
        candidate_count,
        RRF_CONSTANT,
        RRF_CONSTANT,
        top_k,
    )
    with connection.cursor() as cursor:
        cursor.execute(query, parameters)
        return cursor.fetchall()


def fetch_history_overview() -> dict[str, Any]:
    with open_connection() as connection:
        if not archive_search_ready(connection):
            return {
                "historical_document_count": 0,
                "searchable_document_count": 0,
                "review_document_count": 0,
                "historical_fact_count": 0,
                "historical_chunk_count": 0,
                "embedded_chunk_count": 0,
                "history_start_date": None,
                "history_end_date": None,
                "banks": [],
                "product_types": [],
            }
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COUNT(*),
                    COUNT(*) FILTER (WHERE searchable IS TRUE),
                    COUNT(*) FILTER (WHERE quality_status = 'review'),
                    MIN(snapshot_date),
                    MAX(snapshot_date)
                FROM historical_documents
                """
            )
            total, searchable, review, start_date, end_date = cursor.fetchone()
            cursor.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE decision = 'accepted')
                FROM historical_facts
                """
            )
            fact_count = cursor.fetchone()[0]
            cursor.execute(
                """
                SELECT COUNT(*), COUNT(*) FILTER (WHERE embedding IS NOT NULL)
                FROM historical_document_chunks
                """
            )
            chunk_count, embedded_count = cursor.fetchone()
            cursor.execute(
                """
                SELECT b.bank_name, COUNT(*)
                FROM historical_documents d
                JOIN banks b ON b.id = d.bank_id
                GROUP BY b.bank_name
                ORDER BY COUNT(*) DESC, b.bank_name
                """
            )
            banks = [
                {"name": str(name), "count": int(count)}
                for name, count in cursor.fetchall()
            ]
            cursor.execute(
                """
                SELECT COALESCE(product_type_code, 'UNCLASSIFIED'), COUNT(*)
                FROM historical_documents
                GROUP BY product_type_code
                ORDER BY COUNT(*) DESC, product_type_code
                """
            )
            product_types = [
                {"code": str(code), "count": int(count)}
                for code, count in cursor.fetchall()
            ]
    return {
        "historical_document_count": int(total),
        "searchable_document_count": int(searchable),
        "review_document_count": int(review),
        "historical_fact_count": int(fact_count),
        "historical_chunk_count": int(chunk_count),
        "embedded_chunk_count": int(embedded_count),
        "history_start_date": start_date,
        "history_end_date": end_date,
        "banks": banks,
        "product_types": product_types,
    }


def fetch_url_versions(source_url: str) -> list[dict[str, Any]]:
    canonical_url = canonicalize_url(source_url)
    with open_connection() as connection:
        if not archive_search_ready(connection):
            return []
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.id,
                    d.archive_key,
                    b.bank_name,
                    d.page_title,
                    d.source_url,
                    d.archive_url,
                    d.snapshot_date,
                    d.product_type_code,
                    d.classification_confidence,
                    d.quality_status,
                    COUNT(f.id) FILTER (WHERE f.decision = 'accepted')
                FROM historical_documents d
                JOIN banks b ON b.id = d.bank_id
                LEFT JOIN historical_facts f
                    ON f.historical_document_id = d.id
                WHERE d.canonical_url = %s
                GROUP BY d.id, b.bank_name
                ORDER BY d.snapshot_date NULLS LAST, d.id
                """,
                (canonical_url,),
            )
            rows = cursor.fetchall()
    return [
        {
            "document_id": int(row[0]),
            "archive_key": str(row[1]),
            "bank_name": str(row[2]),
            "page_title": row[3],
            "source_url": row[4],
            "archive_url": row[5],
            "snapshot_date": row[6],
            "product_type_code": (
                canonicalize_product_label(row[7]) if row[7] else None
            ),
            "classification_confidence": (
                float(row[8]) if row[8] is not None else None
            ),
            "quality_status": str(row[9]),
            "fact_count": int(row[10]),
        }
        for row in rows
    ]


def fetch_historical_comparison(
    product_type_code: str,
    *,
    bank_names: list[str] | None = None,
    as_of: date | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    filters = [
        "d.product_type_code = ANY(%s)",
        "d.quality_status = 'accepted'",
        "d.searchable IS TRUE",
    ]
    parameters: list[Any] = [list(product_label_variants(product_type_code))]
    if bank_names:
        filters.append("b.bank_name = ANY(%s)")
        parameters.append(bank_names)
    if as_of:
        filters.append("d.snapshot_date <= %s")
        parameters.append(as_of)
    parameters.append(limit)

    with open_connection() as connection:
        if not archive_search_ready(connection):
            return []
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                WITH ranked AS (
                    SELECT
                        d.id,
                        d.archive_key,
                        b.bank_name,
                        d.page_title,
                        d.source_url,
                        d.archive_url,
                        d.snapshot_date,
                        d.product_type_code,
                        d.classification_confidence,
                        d.verified,
                        ROW_NUMBER() OVER (
                            PARTITION BY b.bank_name
                            ORDER BY d.snapshot_date DESC NULLS LAST, d.id DESC
                        ) AS bank_rank
                    FROM historical_documents d
                    JOIN banks b ON b.id = d.bank_id
                    WHERE {' AND '.join(filters)}
                ),
                selected AS (
                    SELECT *
                    FROM ranked
                    WHERE bank_rank = 1
                    ORDER BY bank_name
                    LIMIT %s
                )
                SELECT
                    s.id,
                    s.archive_key,
                    s.bank_name,
                    s.page_title,
                    s.source_url,
                    s.archive_url,
                    s.snapshot_date,
                    s.product_type_code,
                    s.classification_confidence,
                    s.verified,
                    f.fact_type,
                    f.fact_text,
                    f.normalized_value,
                    f.evidence_text,
                    f.extraction_method,
                    f.confidence,
                    f.review_status
                FROM selected s
                LEFT JOIN historical_facts f
                    ON f.historical_document_id = s.id
                   AND f.decision = 'accepted'
                ORDER BY s.bank_name, f.fact_type, f.confidence DESC NULLS LAST
                """,
                parameters,
            )
            rows = cursor.fetchall()

    items: dict[int, dict[str, Any]] = {}
    for row in rows:
        document_id = int(row[0])
        item = items.setdefault(
            document_id,
            {
                "document_id": document_id,
                "archive_key": str(row[1]),
                "bank_name": str(row[2]),
                "page_title": row[3],
                "source_url": row[4],
                "archive_url": row[5],
                "snapshot_date": row[6],
                "product_type_code": (
                    canonicalize_product_label(row[7]) if row[7] else None
                ),
                "classification_confidence": (
                    float(row[8]) if row[8] is not None else None
                ),
                "verified": bool(row[9]),
                "verification_warning": (
                    None if row[9] else (
                        "This result was generated automatically and has "
                        "not been human verified."
                    )
                ),
                "attributes": {},
            },
        )
        if row[10] is None:
            continue
        fact_verified = bool(
            row[16] == "approved"
            or any(
                marker in str(row[14]).casefold()
                for marker in ("human", "manual", "review_approved")
            )
        )
        item["attributes"].setdefault(str(row[10]), []).append(
            {
                "text": str(row[11]),
                "normalized_value": row[12],
                "evidence_text": row[13],
                "source": str(row[14]),
                "confidence": float(row[15]),
                "verified": fact_verified,
                "verification_warning": (
                    None if fact_verified else (
                        "This extracted fact has not been human verified."
                    )
                ),
            }
        )
    return list(items.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search historical HititFinLex data.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bank", action="append", default=[])
    parser.add_argument("--product-type", action="append", default=[])
    parser.add_argument("--date-from", type=date.fromisoformat)
    parser.add_argument("--date-to", type=date.fromisoformat)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 50:
        parser.error("--top-k must be between 1 and 50")
    return args


def main() -> None:
    load_dotenv()
    args = parse_args()
    query = " ".join(args.query).strip()
    model = load_model()
    query_vector = encode_query(model, query)
    lexical_query = build_lexical_query(query)
    with open_connection() as connection:
        rows = search_historical_database(
            connection,
            query_vector,
            lexical_query,
            args.top_k,
            bank_names=args.bank,
            product_types=args.product_type,
            date_from=args.date_from,
            date_to=args.date_to,
        )
    print("Query:", query)
    print("Historical results:", len(rows))
    for rank, row in enumerate(rows, start=1):
        print(
            f"[{rank}] date={row[7] or '-'} bank={row[3]} "
            f"product={row[8] or '-'} hybrid={float(row[12]):.6f}"
        )
        print("  Title:", row[4] or "-")
        print("  URL:", row[5] or "-")
        print("  Text:", " ".join(str(row[9]).split())[:700])
    print("Summary:", json.dumps({"count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
