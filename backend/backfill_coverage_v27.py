from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from coverage_rules_v27 import (
    ELIGIBLE_PRODUCT_CODES,
    PIPELINE_VERSION,
    RuleFact,
    extract_coverage_facts,
    fold_text,
)
from db.runtime_schema import require_migrated_tables
from hybrid_search import get_connection


UPSERT_SQL = """
INSERT INTO comparison_facts (
    document_id,
    fact_type,
    fact_text,
    normalized_value,
    evidence_text,
    extraction_method,
    confidence,
    source_chunk,
    fact_key
)
SELECT %s, %s, %s, %s, %s, %s, %s, 0, %s
WHERE NOT EXISTS (
    SELECT 1
    FROM comparison_facts existing
    WHERE existing.document_id = %s
      AND existing.fact_type = %s
      AND LOWER(BTRIM(existing.fact_text)) = LOWER(BTRIM(%s))
)
AND NOT EXISTS (
    SELECT 1
    FROM passages p
    JOIN entities e ON e.passage_id = p.id
    WHERE p.document_id = %s
      AND e.entity_label = %s
      AND LOWER(BTRIM(e.entity_text)) = LOWER(BTRIM(%s))
)
ON CONFLICT (document_id, fact_key) DO UPDATE SET
    normalized_value = EXCLUDED.normalized_value,
    evidence_text = EXCLUDED.evidence_text,
    confidence = GREATEST(comparison_facts.confidence, EXCLUDED.confidence),
    updated_at = NOW()
RETURNING id
"""


@dataclass(frozen=True)
class Document:
    document_id: int
    bank_name: str
    page_title: str
    source_url: str
    product_code: str
    raw_text: str
    existing_fact_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill high-confidence structured facts for documents with no "
            "existing comparison facts. Dry-run is the default."
        )
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--document-id", action="append", type=int, default=[])
    parser.add_argument("--campaign-type", default="")
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument(
        "--include-covered",
        action="store_true",
        help="Also scan documents that already contain at least one fact.",
    )
    parser.add_argument("--show-evidence", action="store_true")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write accepted facts to PostgreSQL. Without this flag, dry-run only.",
    )
    args = parser.parse_args()
    args.campaign_type = args.campaign_type.strip().upper()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if any(document_id < 1 for document_id in args.document_id):
        parser.error("--document-id values must be positive")
    if not 0.90 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0.90 and 1.0")
    return args


def table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT to_regclass(%s) IS NOT NULL",
        (f"public.{table_name}",),
    )
    return bool(cursor.fetchone()[0])


def available_facts_sql(has_comparison_facts: bool) -> str:
    generated_sql = ""
    if has_comparison_facts:
        generated_sql = """
            UNION ALL
            SELECT document_id
            FROM comparison_facts
        """
    return f"""
        SELECT p.document_id
        FROM passages p
        JOIN entities e ON e.passage_id = p.id
        {generated_sql}
    """


def load_documents(args: argparse.Namespace) -> list[Document]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            facts_sql = available_facts_sql(
                table_exists(cursor, "comparison_facts")
            )
            filters = [
                "d.campaign_type_code = ANY(%s)",
                "COALESCE(NULLIF(BTRIM(d.raw_text), ''), "
                "NULLIF(BTRIM(d.summary_text), ''), "
                "NULLIF(BTRIM(d.page_title), '')) IS NOT NULL",
            ]
            parameters: list[Any] = [sorted(ELIGIBLE_PRODUCT_CODES)]
            if args.campaign_type:
                filters.append("d.campaign_type_code = %s")
                parameters.append(args.campaign_type)
            if args.document_id:
                filters.append("d.id = ANY(%s)")
                parameters.append(args.document_id)
            if not args.include_covered:
                filters.append("COALESCE(fc.fact_count, 0) = 0")
            parameters.append(args.limit)

            query = f"""
                WITH available_facts AS ({facts_sql}),
                fact_counts AS (
                    SELECT document_id, COUNT(*) AS fact_count
                    FROM available_facts
                    GROUP BY document_id
                )
                SELECT
                    d.id,
                    b.bank_name,
                    COALESCE(d.page_title, ''),
                    COALESCE(d.source_url, ''),
                    COALESCE(d.campaign_type_code, ''),
                    COALESCE(
                        NULLIF(BTRIM(d.raw_text), ''),
                        NULLIF(BTRIM(d.summary_text), ''),
                        NULLIF(BTRIM(d.page_title), ''),
                        ''
                    ),
                    COALESCE(fc.fact_count, 0)
                FROM documents d
                JOIN banks b ON b.id = d.bank_id
                LEFT JOIN fact_counts fc ON fc.document_id = d.id
                WHERE {' AND '.join(filters)}
                ORDER BY
                    CASE WHEN COALESCE(fc.fact_count, 0) = 0 THEN 0 ELSE 1 END,
                    d.id
                LIMIT %s
            """
            cursor.execute(query, parameters)
            return [Document(*row) for row in cursor.fetchall()]


def make_fact_key(document_id: int, fact: RuleFact) -> str:
    value = "|".join(
        (
            str(document_id),
            PIPELINE_VERSION,
            fact.fact_type,
            fold_text(fact.fact_text),
            fold_text(fact.evidence_text),
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_facts(
    accepted: list[tuple[Document, RuleFact]],
) -> tuple[int, int]:
    inserted = 0
    touched_documents = set()
    with get_connection() as connection:
        with connection.cursor() as cursor:
            require_migrated_tables(cursor, ("comparison_facts",))
            for document, fact in accepted:
                base = (
                    document.document_id,
                    fact.fact_type,
                    fact.fact_text,
                    Jsonb(fact.normalized_value)
                    if fact.normalized_value is not None
                    else None,
                    fact.evidence_text,
                    PIPELINE_VERSION,
                    fact.confidence,
                    make_fact_key(document.document_id, fact),
                )
                duplicate_checks = (
                    document.document_id,
                    fact.fact_type,
                    fact.fact_text,
                    document.document_id,
                    fact.fact_type,
                    fact.fact_text,
                )
                cursor.execute(UPSERT_SQL, base + duplicate_checks)
                if cursor.fetchone() is not None:
                    inserted += 1
                    touched_documents.add(document.document_id)
    return inserted, len(touched_documents)


def current_coverage() -> tuple[int, int, float]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            facts_sql = available_facts_sql(
                table_exists(cursor, "comparison_facts")
            )
            cursor.execute(
                f"""
                WITH available_facts AS ({facts_sql})
                SELECT
                    (SELECT COUNT(*) FROM documents),
                    COUNT(DISTINCT document_id)
                FROM available_facts
                """
            )
            total, covered = cursor.fetchone()
    percentage = round((covered / total * 100) if total else 0.0, 1)
    return int(total), int(covered), percentage


def main() -> None:
    load_dotenv()
    args = parse_args()
    total_documents, covered_before, coverage_before = current_coverage()
    documents = load_documents(args)

    accepted: list[tuple[Document, RuleFact]] = []
    document_hits = set()
    fact_types = Counter()
    product_types = Counter()
    banks = Counter()

    print("Pipeline:", PIPELINE_VERSION)
    print("Mode:", "DATABASE_WRITE" if args.write else "DRY_RUN")
    print("Current coverage:", f"{covered_before}/{total_documents} ({coverage_before}%)")
    print("Selected documents:", len(documents))

    for index, document in enumerate(documents, start=1):
        source_text = "\n".join(
            part for part in (document.page_title, document.raw_text) if part
        )
        facts = [
            fact
            for fact in extract_coverage_facts(
                source_text,
                document.product_code,
                page_title=document.page_title,
                source_url=document.source_url,
            )
            if fact.confidence >= args.min_confidence
        ]
        if not facts:
            continue
        document_hits.add(document.document_id)
        product_types[document.product_code] += 1
        banks[document.bank_name] += 1
        print(
            f"[{index}/{len(documents)}] document={document.document_id} "
            f"product={document.product_code} bank={document.bank_name} "
            f"new_facts={len(facts)} title={document.page_title[:80]}"
        )
        for fact in facts:
            fact_types[fact.fact_type] += 1
            accepted.append((document, fact))
            print(
                f"  ACCEPT {fact.fact_type:<28} {fact.confidence:.2f} "
                f"| {fact.fact_text} | {fact.rule_name}"
            )
            if args.show_evidence:
                print("    EVIDENCE |", fact.evidence_text)

    projected_covered = min(total_documents, covered_before + len(document_hits))
    projected_percentage = round(
        (projected_covered / total_documents * 100)
        if total_documents
        else 0.0,
        1,
    )

    inserted = 0
    touched = 0
    if args.write and accepted:
        inserted, touched = save_facts(accepted)

    summary = {
        "mode": "write" if args.write else "dry_run",
        "documents_scanned": len(documents),
        "documents_with_new_facts": len(document_hits),
        "accepted_fact_candidates": len(accepted),
        "inserted_facts": inserted,
        "touched_documents": touched,
        "coverage_before": {
            "covered": covered_before,
            "total": total_documents,
            "percentage": coverage_before,
        },
        "projected_after": {
            "covered": projected_covered,
            "total": total_documents,
            "percentage": projected_percentage,
        },
        "fact_types": dict(fact_types.most_common()),
        "product_types": dict(product_types.most_common()),
        "banks": dict(banks.most_common()),
    }
    print("Summary:", json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
