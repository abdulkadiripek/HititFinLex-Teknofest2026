"""Checksum-verified PostgreSQL migrations for HititFinLex."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
BACKEND_ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(BACKEND_ENV_PATH, override=False)

MIGRATION_DIR = Path(__file__).with_name("migrations")
MANIFEST_PATH = MIGRATION_DIR / "manifest.json"
REQUIRED_TABLES = (
    "banks",
    "documents",
    "passages",
    "entities",
    "classification_samples",
    "extraction_samples",
    "document_chunks",
    "comparison_facts",
    "comparison_fact_review_queue",
    "ner_document_state",
    "document_intake_review_queue",
    "document_intake_state",
    "historical_documents",
    "historical_facts",
    "historical_document_chunks",
    "rag_chunks",
    "rag_sessions",
    "rag_messages",
    "rag_session_state",
    "rag_turn_evidence",
)
REQUIRED_RAG_INDEX_METHODS = {
    "rag_chunks_search_vector_idx": "gin",
    "rag_chunks_product_types_idx": "gin",
    "rag_chunks_facts_idx": "gin",
    "rag_chunks_scope_bank_date_idx": "btree",
    "rag_chunks_classification_idx": "btree",
    "rag_sessions_expiry_idx": "btree",
    "rag_messages_session_time_idx": "btree",
    "rag_session_state_updated_idx": "btree",
    "rag_turn_evidence_session_time_idx": "btree",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> list[dict[str, str]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported migration manifest schema")
    migrations = payload.get("migrations")
    if not isinstance(migrations, list) or not migrations:
        raise RuntimeError("Migration manifest is empty")

    seen_versions: set[str] = set()
    seen_files: set[str] = set()
    previous_version = ""
    for migration in migrations:
        version = str(migration.get("version", ""))
        filename = str(migration.get("file", ""))
        expected = str(migration.get("sha256", ""))
        if not version.isdigit() or version <= previous_version:
            raise RuntimeError("Migration versions must be unique and ascending")
        if version in seen_versions or filename in seen_files:
            raise RuntimeError("Duplicate migration manifest entry")
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise RuntimeError(f"Invalid sha256 for migration {version}")

        path = MIGRATION_DIR / filename
        if path.parent != MIGRATION_DIR or not path.is_file():
            raise RuntimeError(f"Missing or unsafe migration file: {filename}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
            )
        if not path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"Empty migration: {filename}")

        seen_versions.add(version)
        seen_files.add(filename)
        previous_version = version
    return migrations


def connection_arguments() -> tuple[tuple[Any, ...], dict[str, Any]]:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return (database_url,), {}

    names = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Set DATABASE_URL or these variables: " + ", ".join(missing)
        )
    return (), {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ["DB_PORT"]),
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


def connect():
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError(
            "psycopg is required for database commands; install backend/requirements.txt"
        ) from error
    args, kwargs = connection_arguments()
    return psycopg.connect(*args, **kwargs)


def ensure_history(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS public.hititfinlex_schema_migrations (
            version VARCHAR(32) PRIMARY KEY,
            filename TEXT NOT NULL,
            sha256 CHAR(64) NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    connection.commit()


def applied_migrations(connection) -> dict[str, tuple[str, str]]:
    rows = connection.execute(
        "SELECT version, filename, sha256 FROM public.hititfinlex_schema_migrations"
    ).fetchall()
    return {str(version): (str(filename), str(checksum)) for version, filename, checksum in rows}


def migrate_up(migrations: list[dict[str, str]]) -> None:
    with connect() as connection:
        ensure_history(connection)
        applied = applied_migrations(connection)
        for migration in migrations:
            version = migration["version"]
            filename = migration["file"]
            checksum = migration["sha256"]
            if version in applied:
                if applied[version] != (filename, checksum):
                    raise RuntimeError(
                        f"Applied migration {version} does not match the repository manifest"
                    )
                print(f"skip {version} {filename}")
                continue

            sql = (MIGRATION_DIR / filename).read_text(encoding="utf-8")
            with connection.transaction():
                connection.execute(sql)
                connection.execute(
                    """
                    INSERT INTO public.hititfinlex_schema_migrations
                        (version, filename, sha256)
                    VALUES (%s, %s, %s)
                    """,
                    (version, filename, checksum),
                )
            print(f"apply {version} {filename}")


def print_status(migrations: list[dict[str, str]]) -> None:
    with connect() as connection:
        history = connection.execute(
            "SELECT to_regclass('public.hititfinlex_schema_migrations')"
        ).fetchone()
        try:
            applied = applied_migrations(connection) if history and history[0] else {}
        except Exception as error:
            if getattr(error, "sqlstate", None) != "42501":
                raise
            connection.rollback()
            markers = {
                "0001": bool(
                    connection.execute(
                        "SELECT to_regclass('public.documents') IS NOT NULL"
                    ).fetchone()[0]
                ),
                "0002": bool(
                    connection.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = 'public'
                              AND table_name = 'document_intake_review_queue'
                              AND column_name = 'base_document_hash'
                        )
                        """
                    ).fetchone()[0]
                ),
                "0003": bool(
                    connection.execute(
                        "SELECT to_regclass('public.rag_sessions') IS NOT NULL"
                    ).fetchone()[0]
                ),
                "0004": bool(
                    connection.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_constraint
                            WHERE conrelid = 'public.rag_messages'::REGCLASS
                              AND contype = 'c'
                              AND pg_get_constraintdef(oid)
                                  ILIKE '%conversational%'
                        )
                        """
                    ).fetchone()[0]
                ),
            }
            for migration in migrations:
                state = (
                    "present-unverified"
                    if markers.get(migration["version"], False)
                    else "pending"
                )
                print(f"{migration['version']} {state} {migration['file']}")
            print("history checksums unavailable to the current database role")
            return
    for migration in migrations:
        state = "applied" if migration["version"] in applied else "pending"
        print(f"{migration['version']} {state} {migration['file']}")


def smoke(migrations: list[dict[str, str]]) -> None:
    migrate_up(migrations)
    with connect() as connection:
        server_version_num = int(
            connection.execute(
                "SELECT current_setting('server_version_num')::INTEGER"
            ).fetchone()[0]
        )
        if server_version_num // 10000 != 18:
            raise RuntimeError(
                "PostgreSQL 18 is required; detected server major "
                f"{server_version_num // 10000}"
            )
        extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        if extension is None:
            raise RuntimeError("pgvector extension is unavailable")
        unaccent_extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'unaccent'"
        ).fetchone()
        if unaccent_extension is None:
            raise RuntimeError("unaccent extension is unavailable")
        public_unaccent = connection.execute(
            "SELECT to_regprocedure('public.unaccent(text)') IS NOT NULL"
        ).fetchone()
        if not public_unaccent or not public_unaccent[0]:
            raise RuntimeError("public.unaccent(text) is unavailable")
        missing = []
        for table in REQUIRED_TABLES:
            row = connection.execute("SELECT to_regclass(%s)", (f"public.{table}",)).fetchone()
            if row is None or row[0] is None:
                missing.append(table)
        embedding_type = connection.execute(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation ON relation.oid = attribute.attrelid
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'document_chunks'
              AND attribute.attname = 'embedding'
              AND NOT attribute.attisdropped
            """
        ).fetchone()
        if missing:
            raise RuntimeError("Missing tables: " + ", ".join(missing))
        if embedding_type is None or embedding_type[0] != "vector(1024)":
            raise RuntimeError(f"Unexpected embedding type: {embedding_type}")
        rag_search_type = connection.execute(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation ON relation.oid = attribute.attrelid
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'rag_chunks'
              AND attribute.attname = 'search_vector'
              AND NOT attribute.attisdropped
            """
        ).fetchone()
        if rag_search_type is None or rag_search_type[0] != "tsvector":
            raise RuntimeError(f"Unexpected RAG search type: {rag_search_type}")
        rag_columns = connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'rag_chunks'
              AND column_name IN (
                  'offer_id',
                  'scope',
                  'document_id',
                  'bank_key',
                  'primary_product',
                  'product_types',
                  'product_scores',
                  'classification_confidence',
                  'classification_status',
                  'classification_conflict',
                  'effective_date',
                  'campaign_start',
                  'campaign_end',
                  'facts',
                  'search_vector'
              )
            """
        ).fetchall()
        required_rag_columns = {
            "offer_id",
            "scope",
            "document_id",
            "bank_key",
            "primary_product",
            "product_types",
            "product_scores",
            "classification_confidence",
            "classification_status",
            "classification_conflict",
            "effective_date",
            "campaign_start",
            "campaign_end",
            "facts",
            "search_vector",
        }
        missing_rag_columns = required_rag_columns.difference(
            str(row[0]) for row in rag_columns
        )
        if missing_rag_columns:
            raise RuntimeError(
                "RAG chunk columns are incomplete: "
                + ", ".join(sorted(missing_rag_columns))
            )
        index_rows = connection.execute(
            """
            SELECT index_relation.relname, access_method.amname
            FROM pg_index AS index_metadata
            JOIN pg_class AS index_relation
              ON index_relation.oid = index_metadata.indexrelid
            JOIN pg_class AS table_relation
              ON table_relation.oid = index_metadata.indrelid
            JOIN pg_namespace AS namespace
              ON namespace.oid = table_relation.relnamespace
            JOIN pg_am AS access_method
              ON access_method.oid = index_relation.relam
            WHERE namespace.nspname = 'public'
              AND index_relation.relname::TEXT = ANY(%s::TEXT[])
            """,
            (list(REQUIRED_RAG_INDEX_METHODS),),
        ).fetchall()
        actual_index_methods = {
            str(index_name): str(method_name)
            for index_name, method_name in index_rows
        }
        mismatched_indexes = sorted(
            index_name
            for index_name, method_name in REQUIRED_RAG_INDEX_METHODS.items()
            if actual_index_methods.get(index_name) != method_name
        )
        if mismatched_indexes:
            raise RuntimeError(
                "RAG indexes are missing or use an unexpected access method: "
                + ", ".join(mismatched_indexes)
            )
        search_trigger = connection.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_trigger AS trigger_metadata
                JOIN pg_class AS relation
                  ON relation.oid = trigger_metadata.tgrelid
                JOIN pg_namespace AS namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname = 'rag_chunks'
                  AND trigger_metadata.tgname =
                      'rag_chunks_search_vector_trigger'
                  AND NOT trigger_metadata.tgisinternal
                  AND trigger_metadata.tgenabled <> 'D'
            )
            """
        ).fetchone()
        if not search_trigger or not search_trigger[0]:
            raise RuntimeError("RAG search vector trigger is unavailable")
        session_columns = connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'rag_sessions'
              AND column_name IN (
                  'token_hash', 'owner_hash', 'expires_at', 'revoked_at'
              )
            """
        ).fetchall()
        if {str(row[0]) for row in session_columns} != {
            "token_hash",
            "owner_hash",
            "expires_at",
            "revoked_at",
        }:
            raise RuntimeError("RAG session security columns are incomplete")
        conversation_status = connection.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conrelid = 'public.rag_messages'::REGCLASS
                  AND contype = 'c'
                  AND pg_get_constraintdef(oid) ILIKE '%conversational%'
            )
            """
        ).fetchone()
        if not conversation_status or not conversation_status[0]:
            raise RuntimeError("RAG conversational status constraint is unavailable")
        print(
            f"migration smoke passed (PostgreSQL "
            f"{server_version_num // 10000}, pgvector {extension[0]}, "
            f"unaccent {unaccent_extension[0]}, vector(1024), RAG V2)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "up", "status", "smoke"))
    args = parser.parse_args()
    migrations = load_manifest()
    if args.command == "check":
        print(f"migration manifest passed ({len(migrations)} migration)")
    elif args.command == "up":
        migrate_up(migrations)
    elif args.command == "status":
        print_status(migrations)
    else:
        smoke(migrations)


def run_cli() -> int:
    try:
        main()
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"database migration failed: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        if error.__class__.__module__.split(".", 1)[0] != "psycopg":
            raise
        print(
            "database migration failed: database operation unavailable",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
