import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import psycopg

from db import migrate as migration_runner


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "db" / "migrations"


class MigrationManifestTest(unittest.TestCase):
    def test_cli_redacts_database_driver_errors(self):
        provider_secret = "unit-test-secret-" + ("8" * 18)
        stderr = io.StringIO()
        with (
            patch.object(
                migration_runner,
                "main",
                side_effect=psycopg.OperationalError(
                    "connection rejected: " + provider_secret
                ),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = migration_runner.run_cli()
        self.assertEqual(exit_code, 2)
        self.assertIn("database operation unavailable", stderr.getvalue())
        self.assertNotIn(provider_secret, stderr.getvalue())

    def test_manifest_matches_files(self):
        manifest = json.loads((MIGRATIONS / "manifest.json").read_text("utf-8"))
        entries = manifest["migrations"]
        self.assertGreater(len(entries), 0)
        self.assertEqual(
            [entry["version"] for entry in entries],
            sorted(entry["version"] for entry in entries),
        )
        for entry in entries:
            path = MIGRATIONS / entry["file"]
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])

    def test_baseline_declares_required_objects(self):
        sql = (MIGRATIONS / "0001_base.sql").read_text("utf-8").lower()
        for name in (
            "create extension if not exists vector",
            "create table if not exists documents",
            "create table if not exists document_chunks",
            "embedding vector(1024)",
            "create table if not exists historical_documents",
        ):
            self.assertIn(name, sql)

    def test_review_baseline_migration_declares_snapshot_columns(self):
        sql = (
            MIGRATIONS / "0002_review_document_baseline.sql"
        ).read_text("utf-8").lower()
        self.assertIn("add column base_document_exists boolean", sql)
        self.assertIn("add column base_document_hash char(64)", sql)
        self.assertIn("document_intake_review_base_snapshot_check", sql)

    def test_rag_v2_migration_declares_weighted_lexical_and_session_indexes(self):
        sql = (MIGRATIONS / "0003_rag_v2.sql").read_text("utf-8").lower()
        for fragment in (
            "create extension if not exists unaccent with schema public",
            "search_vector tsvector",
            "setweight(",
            "'simple'::regconfig",
            "public.unaccent(",
            "using gin (search_vector)",
            "using gin (product_types)",
            "using gin (facts jsonb_path_ops)",
            "rag_chunks_scope_bank_date_idx",
            "rag_chunks_classification_idx",
            "rag_sessions_expiry_idx",
            "rag_messages_session_time_idx",
            "rag_session_state_updated_idx",
            "rag_turn_evidence_session_time_idx",
        ):
            self.assertIn(fragment, sql)

    def test_rag_v2_migration_carries_classification_and_state_fields(self):
        sql = (MIGRATIONS / "0003_rag_v2.sql").read_text("utf-8").lower()
        for fragment in (
            "classification_confidence double precision",
            "classification_status varchar(16)",
            "primary_product varchar(128)",
            "product_types text[]",
            "product_scores jsonb",
            '"active_banks": []',
            '"active_products": []',
            '"active_scope": "current"',
            '"active_offer_ids": []',
            '"last_source_ids": []',
            '"last_document_ids": []',
            '"last_standalone_query": null',
        ):
            self.assertIn(fragment, sql)

    def test_conversation_migration_adds_non_evidentiary_status(self):
        sql = (
            MIGRATIONS / "0004_rag_v2_conversation.sql"
        ).read_text("utf-8").lower()
        self.assertIn("rag_messages_status_check", sql)
        self.assertIn("'conversational'", sql)
        self.assertIn("drop constraint if exists", sql)

    def test_smoke_verifies_postgres_18_extensions_indexes_and_trigger(self):
        source = (ROOT / "db" / "migrate.py").read_text("utf-8")
        for fragment in (
            "server_version_num // 10000 != 18",
            "public.unaccent(text)",
            "REQUIRED_RAG_INDEX_METHODS",
            "rag_chunks_search_vector_trigger",
        ):
            self.assertIn(fragment, source)


@unittest.skipUnless(
    os.getenv("DATABASE_URL"),
    "DATABASE_URL is required for PostgreSQL migration integration tests",
)
class ReviewBaselinePostgresTest(unittest.TestCase):
    def test_new_accepted_document_blocks_older_pending_review(self):
        import psycopg

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from intake_service import (
            IntakeReviewConflictError,
            content_digest,
            lock_pending_document_review,
            queue_document_review,
            validate_review_document_baseline,
        )

        suffix = uuid4().hex
        record_key = f"review-baseline-{suffix}"
        bank_key = f"review-bank-{suffix[:20]}"
        candidate_text = "Pending candidate document content"
        accepted_text = "Newer accepted document content"
        candidate_digest = content_digest(candidate_text)

        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'document_intake_review_queue'
                          AND column_name IN (
                              'base_document_exists',
                              'base_document_hash'
                          )
                        """
                    )
                    baseline_columns = {
                        str(row[0]) for row in cursor.fetchall()
                    }
                    if baseline_columns != {
                        "base_document_exists",
                        "base_document_hash",
                    }:
                        migration_sql = (
                            MIGRATIONS
                            / "0002_review_document_baseline.sql"
                        ).read_text("utf-8")
                        cursor.execute(migration_sql)
                    cursor.execute(
                        """
                        INSERT INTO banks (bank_key, bank_name)
                        VALUES (%s, %s)
                        RETURNING id
                        """,
                        (bank_key, "Review Baseline Test Bank"),
                    )
                    bank_id = int(cursor.fetchone()[0])
                    review_id = queue_document_review(
                        cursor,
                        record_key=record_key,
                        bank_key=bank_key,
                        bank_name="Review Baseline Test Bank",
                        source_url=f"https://example.test/{suffix}/pending",
                        page_title="Pending candidate",
                        raw_text=candidate_text,
                        digest=candidate_digest,
                        classification={"decision": "REVIEW"},
                        reason="classification_review",
                    )
                    cursor.execute(
                        """
                        INSERT INTO documents (
                            record_key,
                            bank_id,
                            source_url,
                            raw_text,
                            auto_accepted
                        )
                        VALUES (%s, %s, %s, %s, TRUE)
                        """,
                        (
                            record_key,
                            bank_id,
                            f"https://example.test/{suffix}/accepted",
                            accepted_text,
                        ),
                    )
                    cursor.execute(
                        """
                        SELECT updated_at
                        FROM document_intake_review_queue
                        WHERE id = %s
                        """,
                        (review_id,),
                    )
                    review_updated_at = cursor.fetchone()[0]
                    locked = lock_pending_document_review(
                        cursor,
                        review_id=review_id,
                        record_key=record_key,
                        digest=candidate_digest,
                        review_updated_at=review_updated_at,
                    )
                    cursor.execute(
                        """
                        SELECT raw_text
                        FROM documents
                        WHERE record_key = %s
                        FOR UPDATE
                        """,
                        (record_key,),
                    )
                    current_text = str(cursor.fetchone()[0])

                    with self.assertRaises(IntakeReviewConflictError):
                        validate_review_document_baseline(
                            review_id=review_id,
                            candidate_digest=candidate_digest,
                            base_document_exists=locked[
                                "base_document_exists"
                            ],
                            base_document_hash=locked["base_document_hash"],
                            current_document_digest=content_digest(current_text),
                        )

                    cursor.execute(
                        """
                        SELECT d.raw_text, q.review_status
                        FROM documents d
                        JOIN document_intake_review_queue q
                          ON q.record_key = d.record_key
                        WHERE d.record_key = %s AND q.id = %s
                        """,
                        (record_key, review_id),
                    )
                    raw_text, review_status = cursor.fetchone()
                    self.assertEqual(raw_text, accepted_text)
                    self.assertEqual(review_status, "pending")
            finally:
                connection.rollback()


@unittest.skipUnless(
    os.getenv("RAG_V2_TEST_DATABASE_URL"),
    "RAG_V2_TEST_DATABASE_URL is required for destructive-free migration tests",
)
class RagV2PostgresMigrationTest(unittest.TestCase):
    def test_rag_v2_sql_can_run_twice_in_one_rolled_back_transaction(self):
        import psycopg

        migration_sql = (MIGRATIONS / "0003_rag_v2.sql").read_text("utf-8")
        with psycopg.connect(os.environ["RAG_V2_TEST_DATABASE_URL"]) as connection:
            try:
                major = int(
                    connection.execute(
                        "SELECT current_setting('server_version_num')::INTEGER"
                    ).fetchone()[0]
                ) // 10000
                self.assertEqual(major, 18)
                connection.execute(migration_sql)
                connection.execute(migration_sql)
                relation = connection.execute(
                    "SELECT to_regclass('public.rag_sessions')"
                ).fetchone()
                trigger = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM pg_trigger
                    WHERE tgname = 'rag_chunks_search_vector_trigger'
                      AND NOT tgisinternal
                    """
                ).fetchone()
                self.assertIsNotNone(relation[0])
                self.assertEqual(int(trigger[0]), 1)
            finally:
                connection.rollback()


if __name__ == "__main__":
    unittest.main()
