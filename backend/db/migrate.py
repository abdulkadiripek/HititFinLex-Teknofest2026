"""Checksum-verified PostgreSQL migrations for HititFinLex."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
)


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
        CREATE TABLE IF NOT EXISTS hititfinlex_schema_migrations (
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
        "SELECT version, filename, sha256 FROM hititfinlex_schema_migrations"
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
                    INSERT INTO hititfinlex_schema_migrations
                        (version, filename, sha256)
                    VALUES (%s, %s, %s)
                    """,
                    (version, filename, checksum),
                )
            print(f"apply {version} {filename}")


def print_status(migrations: list[dict[str, str]]) -> None:
    with connect() as connection:
        ensure_history(connection)
        applied = applied_migrations(connection)
    for migration in migrations:
        state = "applied" if migration["version"] in applied else "pending"
        print(f"{migration['version']} {state} {migration['file']}")


def smoke(migrations: list[dict[str, str]]) -> None:
    migrate_up(migrations)
    with connect() as connection:
        extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        if extension is None:
            raise RuntimeError("pgvector extension is unavailable")
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
        print(f"migration smoke passed (pgvector {extension[0]}, vector(1024))")


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


if __name__ == "__main__":
    main()
