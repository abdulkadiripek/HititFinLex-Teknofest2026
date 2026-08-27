from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

from hybrid_search import get_connection


API_URL = "http://127.0.0.1:8000"


def table_count(cursor, table_name: str, where_clause: str = "") -> int:
    cursor.execute(
        "SELECT to_regclass(%s) IS NOT NULL",
        (f"public.{table_name}",),
    )
    if not cursor.fetchone()[0]:
        return 0
    query = f"SELECT COUNT(*) FROM {table_name} {where_clause}"
    cursor.execute(query)
    return int(cursor.fetchone()[0])


def database_snapshot() -> dict[str, int]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            return {
                "documents": table_count(cursor, "documents"),
                "chunks": table_count(cursor, "document_chunks"),
                "facts": table_count(cursor, "comparison_facts"),
                "document_reviews": table_count(
                    cursor,
                    "document_intake_review_queue",
                    "WHERE review_status = 'pending'",
                ),
                "fact_reviews": table_count(
                    cursor,
                    "comparison_fact_review_queue",
                    "WHERE review_status = 'pending'",
                ),
            }


def load_existing_housing_document() -> dict:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.record_key,
                    b.bank_key,
                    b.bank_name,
                    d.source_url,
                    COALESCE(d.page_title, ''),
                    d.raw_text
                FROM documents d
                JOIN banks b ON b.id = d.bank_id
                WHERE d.campaign_type_code = 'KONUT_FINANSMANI'
                  AND LENGTH(BTRIM(d.raw_text)) >= 20
                  AND (
                      LOWER(COALESCE(d.page_title, '')) LIKE '%konut%'
                      OR LOWER(d.raw_text) LIKE '%konut finans%'
                      OR LOWER(d.raw_text) LIKE '%mortgage%'
                  )
                ORDER BY d.confidence DESC NULLS LAST, d.id
                LIMIT 1
                """
            )
            row = cursor.fetchone()
    if row is None:
        raise RuntimeError("No suitable existing housing document was found.")
    return {
        "record_key": row[0],
        "bank_key": row[1],
        "bank_name": row[2],
        "source_url": row[3],
        "page_title": row[4],
        "raw_text": row[5],
        "classification_threshold": 0.8,
        "ner_threshold": 0.4,
        "review_threshold": 0.6,
        "allow_update": False,
    }


def post_intake(client: httpx.Client, payload: dict) -> dict:
    response = client.post("/intake", json=payload)
    response.raise_for_status()
    return response.json()


def main():
    load_dotenv()
    admin_api_key = os.getenv("HITITFINLEX_ADMIN_API_KEY", "").strip()
    if not admin_api_key:
        raise RuntimeError(
            "HITITFINLEX_ADMIN_API_KEY is required for the write check."
        )
    payload = load_existing_housing_document()
    before = database_snapshot()

    with httpx.Client(
        base_url=API_URL,
        timeout=300.0,
        headers={"X-API-Key": admin_api_key},
    ) as client:
        health_response = client.get("/health")
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("intake_duplicate_gate") != "record_hash_first_v1":
            raise RuntimeError(
                "Safety stop: API 0.8.1 duplicate gate is not active. "
                "No database write request was sent."
            )

        preview = post_intake(client, {**payload, "write": False})
        result = post_intake(client, {**payload, "write": True})

    after = database_snapshot()
    assert result["status"] == "UNCHANGED", result
    assert result["database"]["action"] == "unchanged_skipped", result
    assert result["database"]["chunks_written"] == 0, result
    assert result["database"]["facts_written"] == 0, result
    assert before == after, {"before": before, "after": after}

    print("Existing record selected:", payload["record_key"])
    print("Dry-run status:", preview["status"])
    print("Database action: unchanged_skipped")
    print("Database counts unchanged:", before)
    print("Incremental write safety: OK")


if __name__ == "__main__":
    main()
