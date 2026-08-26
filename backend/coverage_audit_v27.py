from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass

from dotenv import load_dotenv

from coverage_rules_v27 import (
    ELIGIBLE_PRODUCT_CODES,
    extract_coverage_facts,
)
from hybrid_search import get_connection


@dataclass(frozen=True)
class AuditDocument:
    document_id: int
    bank_name: str
    page_title: str
    source_url: str
    product_code: str
    raw_text: str
    fact_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report overall, eligible, and recoverable fact coverage."
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    if args.top < 1 or args.top > 100:
        parser.error("--top must be between 1 and 100")
    return args


def table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT to_regclass(%s) IS NOT NULL",
        (f"public.{table_name}",),
    )
    return bool(cursor.fetchone()[0])


def load_documents() -> list[AuditDocument]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            generated_sql = ""
            if table_exists(cursor, "comparison_facts"):
                generated_sql = """
                    UNION ALL
                    SELECT document_id
                    FROM comparison_facts
                """
            cursor.execute(
                f"""
                WITH available_facts AS (
                    SELECT p.document_id
                    FROM passages p
                    JOIN entities e ON e.passage_id = p.id
                    {generated_sql}
                ),
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
                ORDER BY d.id
                """
            )
            return [AuditDocument(*row) for row in cursor.fetchall()]


def ratio(numerator: int, denominator: int) -> float:
    return round((numerator / denominator * 100) if denominator else 0.0, 1)


def build_report(documents: list[AuditDocument], top: int) -> dict:
    total = len(documents)
    covered = [document for document in documents if document.fact_count > 0]
    eligible = [
        document
        for document in documents
        if document.product_code in ELIGIBLE_PRODUCT_CODES
        and bool(document.raw_text.strip())
    ]
    eligible_covered = [
        document for document in eligible if document.fact_count > 0
    ]
    eligible_gaps = [
        document for document in eligible if document.fact_count == 0
    ]

    recoverable = []
    projected_fact_types = Counter()
    for document in eligible_gaps:
        facts = extract_coverage_facts(
            "\n".join(
                part
                for part in (document.page_title, document.raw_text)
                if part
            ),
            document.product_code,
            page_title=document.page_title,
            source_url=document.source_url,
        )
        if facts:
            recoverable.append((document, facts))
            projected_fact_types.update(fact.fact_type for fact in facts)

    gap_products = Counter(document.product_code for document in eligible_gaps)
    gap_banks = Counter(document.bank_name for document in eligible_gaps)
    recoverable_products = Counter(
        document.product_code for document, _facts in recoverable
    )
    recoverable_banks = Counter(
        document.bank_name for document, _facts in recoverable
    )

    projected_covered = min(total, len(covered) + len(recoverable))
    return {
        "overall": {
            "total_documents": total,
            "covered_documents": len(covered),
            "gap_documents": total - len(covered),
            "coverage_percentage": ratio(len(covered), total),
        },
        "eligible": {
            "eligible_documents": len(eligible),
            "covered_documents": len(eligible_covered),
            "gap_documents": len(eligible_gaps),
            "coverage_percentage": ratio(len(eligible_covered), len(eligible)),
        },
        "recoverable": {
            "documents": len(recoverable),
            "projected_overall_covered": projected_covered,
            "projected_overall_percentage": ratio(projected_covered, total),
            "fact_types": dict(projected_fact_types.most_common()),
        },
        "top_eligible_gaps_by_product": dict(gap_products.most_common(top)),
        "top_eligible_gaps_by_bank": dict(gap_banks.most_common(top)),
        "recoverable_by_product": dict(recoverable_products.most_common(top)),
        "recoverable_by_bank": dict(recoverable_banks.most_common(top)),
        "sample_recoverable_documents": [
            {
                "document_id": document.document_id,
                "bank_name": document.bank_name,
                "page_title": document.page_title,
                "product_code": document.product_code,
                "fact_types": sorted({fact.fact_type for fact in facts}),
            }
            for document, facts in recoverable[:top]
        ],
    }


def print_human(report: dict) -> None:
    overall = report["overall"]
    eligible = report["eligible"]
    recoverable = report["recoverable"]
    print("HititFinLex Coverage Audit V2.7")
    print(
        "Overall coverage:",
        f"{overall['covered_documents']}/{overall['total_documents']}",
        f"({overall['coverage_percentage']}%)",
    )
    print(
        "Eligible coverage:",
        f"{eligible['covered_documents']}/{eligible['eligible_documents']}",
        f"({eligible['coverage_percentage']}%)",
    )
    print("Eligible gaps:", eligible["gap_documents"])
    print("Recoverable high-confidence gaps:", recoverable["documents"])
    print(
        "Projected overall coverage:",
        f"{recoverable['projected_overall_covered']}/"
        f"{overall['total_documents']}",
        f"({recoverable['projected_overall_percentage']}%)",
    )
    print(
        "Top gap products:",
        json.dumps(
            report["top_eligible_gaps_by_product"],
            ensure_ascii=True,
            sort_keys=True,
        ),
    )
    print(
        "Recoverable products:",
        json.dumps(
            report["recoverable_by_product"],
            ensure_ascii=True,
            sort_keys=True,
        ),
    )
    print(
        "Projected fact types:",
        json.dumps(
            recoverable["fact_types"],
            ensure_ascii=True,
            sort_keys=True,
        ),
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    report = build_report(load_documents(), args.top)
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print_human(report)


if __name__ == "__main__":
    main()
