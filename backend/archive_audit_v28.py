from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from archive_common_v28 import (
    archive_tables_exist,
    iter_archive_documents,
    load_live_fingerprints,
    open_connection,
)
from historical_search_v28 import fetch_history_overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the V2.8 historical dataset and database state."
    )
    parser.add_argument(
        "--dataset-zip",
        default="HititFinLex_VeriSeti_2.zip",
    )
    return parser.parse_args()


def audit_zip(zip_path: Path) -> dict:
    with open_connection() as connection:
        _, live_urls, live_hashes = load_live_fingerprints(connection)
    counters = Counter()
    bank_counts = Counter()
    hashes = set()
    urls = set()
    dates = []
    for document in iter_archive_documents(zip_path):
        counters["archive_rows"] += 1
        bank_counts[document.bank_name] += 1
        if document.content_hash in hashes:
            counters["duplicate_archive_text"] += 1
        hashes.add(document.content_hash)
        urls.add(document.canonical_url)
        if document.content_hash in live_hashes:
            counters["exact_live_text_overlap"] += 1
        if document.canonical_url in live_urls:
            counters["live_url_version_overlap"] += 1
        if document.snapshot_date:
            dates.append(document.snapshot_date)
    return {
        **dict(counters),
        "unique_archive_texts": len(hashes),
        "unique_archive_urls": len(urls),
        "start_date": min(dates).isoformat() if dates else None,
        "end_date": max(dates).isoformat() if dates else None,
        "banks": dict(bank_counts.most_common()),
    }


def audit_database() -> dict:
    with open_connection() as connection:
        if not archive_tables_exist(connection):
            return {"schema_ready": False}
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT quality_status, COUNT(*)
                FROM historical_documents
                GROUP BY quality_status
                ORDER BY quality_status
                """
            )
            quality = {str(key): int(value) for key, value in cursor.fetchall()}
            cursor.execute(
                """
                SELECT decision, COUNT(*)
                FROM historical_facts
                GROUP BY decision
                ORDER BY decision
                """
            )
            facts = {str(key): int(value) for key, value in cursor.fetchall()}
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM historical_facts
                WHERE decision = 'review' AND review_status = 'pending'
                """
            )
            pending_fact_reviews = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM historical_documents d
                WHERE d.searchable IS TRUE
                  AND NOT EXISTS (
                      SELECT 1
                      FROM historical_document_chunks c
                      WHERE c.historical_document_id = d.id
                        AND c.embedding IS NOT NULL
                  )
                """
            )
            missing_embeddings = int(cursor.fetchone()[0])
    overview = fetch_history_overview()
    overview["history_start_date"] = (
        overview["history_start_date"].isoformat()
        if overview["history_start_date"]
        else None
    )
    overview["history_end_date"] = (
        overview["history_end_date"].isoformat()
        if overview["history_end_date"]
        else None
    )
    return {
        "schema_ready": True,
        "quality_status": quality,
        "fact_decisions": facts,
        "pending_fact_reviews": pending_fact_reviews,
        "searchable_documents_missing_embeddings": missing_embeddings,
        "overview": overview,
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    result = {
        "zip": audit_zip(Path(args.dataset_zip)),
        "database": audit_database(),
    }
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
