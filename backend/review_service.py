from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from hybrid_search import get_connection
from intake_service import PRODUCT_TITLES


REVIEW_STATUSES = {"pending", "approved", "rejected"}
PRODUCT_TYPE_CHOICES = sorted(PRODUCT_TITLES)
HUMAN_FACT_METHOD = "human_review_v1"


class ReviewNotFoundError(RuntimeError):
    pass


def table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT to_regclass(%s) IS NOT NULL",
        (f"public.{table_name}",),
    )
    return bool(cursor.fetchone()[0])


def validate_status(status: str) -> str:
    cleaned = status.strip().casefold()
    if cleaned not in REVIEW_STATUSES:
        raise ValueError(
            "review_status must be pending, approved, or rejected."
        )
    return cleaned


def queue_status_counts(cursor, table_name: str) -> dict[str, int]:
    counts = {status: 0 for status in sorted(REVIEW_STATUSES)}
    if not table_exists(cursor, table_name):
        return counts
    cursor.execute(
        f"""
        SELECT review_status, COUNT(*)
        FROM {table_name}
        GROUP BY review_status
        """
    )
    for status, count in cursor.fetchall():
        counts[str(status)] = int(count)
    return counts


def review_summary() -> dict[str, Any]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            document_counts = queue_status_counts(
                cursor,
                "document_intake_review_queue",
            )
            fact_counts = queue_status_counts(
                cursor,
                "comparison_fact_review_queue",
            )
    return {
        "document_reviews": document_counts,
        "fact_reviews": fact_counts,
        "pending_total": (
            document_counts["pending"] + fact_counts["pending"]
        ),
        "product_type_choices": PRODUCT_TYPE_CHOICES,
    }


def list_document_reviews(
    review_status: str = "pending",
    limit: int = 50,
) -> list[dict[str, Any]]:
    review_status = validate_status(review_status)
    with get_connection() as connection:
        with connection.cursor() as cursor:
            if not table_exists(cursor, "document_intake_review_queue"):
                return []
            cursor.execute(
                """
                SELECT
                    id,
                    record_key,
                    bank_key,
                    bank_name,
                    source_url,
                    COALESCE(page_title, ''),
                    LEFT(raw_text, 700),
                    content_hash,
                    classification,
                    review_reason,
                    review_status,
                    created_at,
                    updated_at
                FROM document_intake_review_queue
                WHERE review_status = %s
                ORDER BY created_at, id
                LIMIT %s
                """,
                (review_status, limit),
            )
            rows = cursor.fetchall()
    return [
        {
            "id": int(row[0]),
            "record_key": row[1],
            "bank_key": row[2],
            "bank_name": row[3],
            "source_url": row[4],
            "page_title": row[5],
            "raw_text_preview": row[6],
            "content_hash": row[7].strip(),
            "classification": row[8],
            "review_reason": row[9],
            "review_status": row[10],
            "created_at": row[11],
            "updated_at": row[12],
        }
        for row in rows
    ]


def list_fact_reviews(
    review_status: str = "pending",
    limit: int = 50,
) -> list[dict[str, Any]]:
    review_status = validate_status(review_status)
    with get_connection() as connection:
        with connection.cursor() as cursor:
            if not table_exists(cursor, "comparison_fact_review_queue"):
                return []
            cursor.execute(
                """
                SELECT
                    q.id,
                    q.document_id,
                    b.bank_name,
                    COALESCE(d.page_title, ''),
                    d.source_url,
                    q.fact_type,
                    q.fact_text,
                    q.normalized_value,
                    q.evidence_text,
                    q.extraction_method,
                    q.confidence,
                    q.source_chunk,
                    q.review_reason,
                    q.review_status,
                    q.created_at,
                    q.updated_at
                FROM comparison_fact_review_queue q
                JOIN documents d ON d.id = q.document_id
                JOIN banks b ON b.id = d.bank_id
                WHERE q.review_status = %s
                ORDER BY q.created_at, q.id
                LIMIT %s
                """,
                (review_status, limit),
            )
            rows = cursor.fetchall()
    return [
        {
            "id": int(row[0]),
            "document_id": int(row[1]),
            "bank_name": row[2],
            "page_title": row[3],
            "source_url": row[4],
            "fact_type": row[5],
            "fact_text": row[6],
            "normalized_value": row[7],
            "evidence_text": row[8],
            "extraction_method": row[9],
            "confidence": float(row[10]),
            "source_chunk": int(row[11]),
            "review_reason": row[12],
            "review_status": row[13],
            "created_at": row[14],
            "updated_at": row[15],
        }
        for row in rows
    ]


def load_pending_document_review(review_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            if not table_exists(cursor, "document_intake_review_queue"):
                raise ReviewNotFoundError(
                    "Document review queue does not exist."
                )
            cursor.execute(
                """
                SELECT
                    id,
                    record_key,
                    bank_key,
                    bank_name,
                    source_url,
                    COALESCE(page_title, ''),
                    raw_text,
                    content_hash,
                    classification,
                    review_reason
                FROM document_intake_review_queue
                WHERE id = %s AND review_status = 'pending'
                """,
                (review_id,),
            )
            row = cursor.fetchone()
    if row is None:
        raise ReviewNotFoundError(
            f"Pending document review {review_id} was not found."
        )
    return {
        "id": int(row[0]),
        "record_key": row[1],
        "bank_key": row[2],
        "bank_name": row[3],
        "source_url": row[4],
        "page_title": row[5],
        "raw_text": row[6],
        "content_hash": row[7].strip(),
        "classification": row[8],
        "review_reason": row[9],
    }


def set_document_review_status(review_id: int, status: str) -> dict[str, Any]:
    status = validate_status(status)
    if status == "pending":
        raise ValueError("Resolution status cannot be pending.")
    with get_connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                if not table_exists(cursor, "document_intake_review_queue"):
                    raise ReviewNotFoundError(
                        "Document review queue does not exist."
                    )
                cursor.execute(
                    """
                    UPDATE document_intake_review_queue
                    SET review_status = %s, updated_at = NOW()
                    WHERE id = %s AND review_status = 'pending'
                    RETURNING id, record_key, review_status
                    """,
                    (status, review_id),
                )
                row = cursor.fetchone()
    if row is None:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                if not table_exists(
                    cursor,
                    "document_intake_review_queue",
                ):
                    raise ReviewNotFoundError(
                        "Document review queue does not exist."
                    )
                cursor.execute(
                    """
                    SELECT id, record_key, review_status
                    FROM document_intake_review_queue
                    WHERE id = %s
                    """,
                    (review_id,),
                )
                existing = cursor.fetchone()
        if existing is None or existing[2] != status:
            raise ReviewNotFoundError(
                f"Pending document review {review_id} was not found."
            )
        row = existing
    return {
        "id": int(row[0]),
        "record_key": row[1],
        "review_status": row[2],
    }


def reject_fact_review(review_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                if not table_exists(cursor, "comparison_fact_review_queue"):
                    raise ReviewNotFoundError(
                        "Fact review queue does not exist."
                    )
                cursor.execute(
                    """
                    UPDATE comparison_fact_review_queue
                    SET review_status = 'rejected', updated_at = NOW()
                    WHERE id = %s AND review_status = 'pending'
                    RETURNING id, document_id, fact_type, fact_text
                    """,
                    (review_id,),
                )
                row = cursor.fetchone()
    if row is None:
        raise ReviewNotFoundError(
            f"Pending fact review {review_id} was not found."
        )
    return {
        "id": int(row[0]),
        "document_id": int(row[1]),
        "fact_type": row[2],
        "fact_text": row[3],
        "review_status": "rejected",
    }


def approve_fact_review(review_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                if not table_exists(cursor, "comparison_fact_review_queue"):
                    raise ReviewNotFoundError(
                        "Fact review queue does not exist."
                    )
                if not table_exists(cursor, "comparison_facts"):
                    raise RuntimeError("Table public.comparison_facts was not found.")
                cursor.execute(
                    """
                    SELECT
                        id,
                        document_id,
                        fact_type,
                        fact_text,
                        normalized_value,
                        evidence_text,
                        confidence,
                        source_chunk,
                        fact_key
                    FROM comparison_fact_review_queue
                    WHERE id = %s AND review_status = 'pending'
                    FOR UPDATE
                    """,
                    (review_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ReviewNotFoundError(
                        f"Pending fact review {review_id} was not found."
                    )

                cursor.execute(
                    """
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
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, fact_key) DO UPDATE SET
                        fact_type = EXCLUDED.fact_type,
                        fact_text = EXCLUDED.fact_text,
                        normalized_value = EXCLUDED.normalized_value,
                        evidence_text = EXCLUDED.evidence_text,
                        extraction_method = EXCLUDED.extraction_method,
                        confidence = EXCLUDED.confidence,
                        source_chunk = EXCLUDED.source_chunk,
                        updated_at = NOW()
                    """,
                    (
                        row[1],
                        row[2],
                        row[3],
                        Jsonb(row[4]) if row[4] is not None else None,
                        row[5],
                        HUMAN_FACT_METHOD,
                        row[6],
                        row[7],
                        row[8],
                    ),
                )
                cursor.execute(
                    """
                    UPDATE comparison_fact_review_queue
                    SET review_status = 'approved', updated_at = NOW()
                    WHERE id = %s
                    """,
                    (review_id,),
                )
    return {
        "id": int(row[0]),
        "document_id": int(row[1]),
        "fact_type": row[2],
        "fact_text": row[3],
        "review_status": "approved",
        "extraction_method": HUMAN_FACT_METHOD,
    }
