from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import unquote, urlsplit, urlunsplit

PIPELINE_VERSION = "historical_v2_8"
SOURCE_DATASET = "HititFinLex_VeriSeti_2"
ARCHIVE_MEMBER_SUFFIX = "korpus/korpus_tum.jsonl"


@dataclass(frozen=True)
class ArchiveDocument:
    archive_key: str
    bank_key: str
    bank_name: str
    source_url: str
    canonical_url: str
    canonical_group_key: str
    archive_url: str
    page_title: str
    raw_text: str
    content_hash: str
    snapshot_date: date | None
    collected_at: datetime | None
    source_category: str
    is_campaign_hint: bool


SCHEMA_STATEMENTS = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS historical_documents (
        id BIGSERIAL PRIMARY KEY,
        archive_key VARCHAR(255) NOT NULL UNIQUE,
        bank_id BIGINT NOT NULL REFERENCES banks(id) ON DELETE RESTRICT,
        source_url TEXT NOT NULL,
        canonical_url TEXT NOT NULL,
        canonical_group_key CHAR(64) NOT NULL,
        archive_url TEXT,
        page_title TEXT,
        raw_text TEXT NOT NULL,
        content_hash CHAR(64) NOT NULL UNIQUE,
        snapshot_date DATE,
        collected_at TIMESTAMPTZ,
        source_category VARCHAR(128),
        is_campaign_hint BOOLEAN NOT NULL DEFAULT FALSE,
        campaign_label VARCHAR(16),
        campaign_confidence DOUBLE PRECISION,
        product_type_code VARCHAR(64),
        product_type VARCHAR(128),
        classification_confidence DOUBLE PRECISION,
        classification_decision VARCHAR(16) NOT NULL
            CHECK (classification_decision IN ('ACCEPTED', 'REVIEW', 'FAILED')),
        classification_basis VARCHAR(128),
        classification_payload JSONB,
        quality_status VARCHAR(16) NOT NULL
            CHECK (quality_status IN ('accepted', 'review', 'failed')),
        searchable BOOLEAN NOT NULL DEFAULT FALSE,
        verified BOOLEAN NOT NULL DEFAULT FALSE,
        verification_source VARCHAR(64),
        pipeline_version VARCHAR(64) NOT NULL,
        source_dataset VARCHAR(128) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    ALTER TABLE historical_documents
    ADD COLUMN IF NOT EXISTS verified BOOLEAN NOT NULL DEFAULT FALSE
    """,
    """
    ALTER TABLE historical_documents
    ADD COLUMN IF NOT EXISTS verification_source VARCHAR(64)
    """,
    """
    CREATE INDEX IF NOT EXISTS historical_documents_url_date_idx
    ON historical_documents (canonical_group_key, snapshot_date)
    """,
    """
    CREATE INDEX IF NOT EXISTS historical_documents_filter_idx
    ON historical_documents (
        searchable,
        product_type_code,
        bank_id,
        snapshot_date
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS historical_facts (
        id BIGSERIAL PRIMARY KEY,
        historical_document_id BIGINT NOT NULL
            REFERENCES historical_documents(id) ON DELETE CASCADE,
        fact_type VARCHAR(64) NOT NULL,
        fact_text TEXT NOT NULL,
        normalized_value JSONB,
        evidence_text TEXT NOT NULL,
        extraction_method VARCHAR(64) NOT NULL,
        confidence DOUBLE PRECISION NOT NULL
            CHECK (confidence >= 0 AND confidence <= 1),
        source_chunk INTEGER NOT NULL DEFAULT 0,
        decision VARCHAR(16) NOT NULL
            CHECK (decision IN ('accepted', 'review', 'rejected')),
        decision_reason VARCHAR(128) NOT NULL,
        review_status VARCHAR(16) NOT NULL DEFAULT 'pending'
            CHECK (review_status IN ('pending', 'approved', 'rejected', 'not_required')),
        fact_key CHAR(64) NOT NULL,
        pipeline_version VARCHAR(64) NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (historical_document_id, fact_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS historical_facts_document_type_idx
    ON historical_facts (historical_document_id, fact_type, decision)
    """,
    """
    CREATE INDEX IF NOT EXISTS historical_facts_review_idx
    ON historical_facts (review_status, confidence DESC)
    WHERE decision = 'review'
    """,
    """
    CREATE TABLE IF NOT EXISTS historical_document_chunks (
        id BIGSERIAL PRIMARY KEY,
        historical_document_id BIGINT NOT NULL
            REFERENCES historical_documents(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        token_count INTEGER,
        content_hash CHAR(64) NOT NULL,
        embedding_model VARCHAR(128),
        embedding vector(1024),
        search_vector TSVECTOR,
        metadata JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (historical_document_id, chunk_index)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS historical_chunks_search_idx
    ON historical_document_chunks USING GIN (search_vector)
    """,
    """
    CREATE INDEX IF NOT EXISTS historical_chunks_embedding_idx
    ON historical_document_chunks USING hnsw (embedding vector_cosine_ops)
    """,
)


def fold_text(value: str) -> str:
    translated = value.translate(
        str.maketrans({"\u0131": "i", "\u0130": "I"})
    )
    decomposed = unicodedata.normalize("NFKD", translated)
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip().casefold()


def normalize_content(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_digest(value: str) -> str:
    return sha256_text(normalize_content(value))


def canonicalize_url(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return fold_text(unquote(raw)).rstrip("/")

    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    path = unquote(parsed.path or "/")
    path = re.sub(r"/+", "/", path).rstrip("/") or "/"
    return urlunsplit(("https", host, path, "", ""))


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_date(value: Any) -> date | None:
    parsed = parse_datetime(value)
    if parsed is not None:
        return parsed.date()
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def validate_zip_members(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        path = PurePosixPath(info.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe ZIP member: {info.filename}")
        if info.file_size < 0:
            raise ValueError(f"Invalid ZIP member size: {info.filename}")


def find_archive_member(archive: zipfile.ZipFile) -> str:
    matches = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and info.filename.replace("\\", "/").endswith(ARCHIVE_MEMBER_SUFFIX)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one korpus/korpus_tum.jsonl member, "
            f"found {len(matches)}."
        )
    return matches[0]


def row_to_document(row: dict[str, Any]) -> ArchiveDocument:
    raw_text = str(row.get("ham_metin") or "").strip()
    source_url = str(row.get("kaynak_url") or "").strip()
    canonical_url = canonicalize_url(source_url)
    if not raw_text:
        raise ValueError("Archive row has empty ham_metin.")
    if not canonical_url:
        raise ValueError("Archive row has empty kaynak_url.")

    archive_key = str(row.get("kayit_id") or "").strip()
    if not archive_key:
        archive_key = "archive-" + content_digest(raw_text)[:20]

    return ArchiveDocument(
        archive_key=archive_key,
        bank_key=str(row.get("banka_key") or "unknown").strip(),
        bank_name=str(row.get("banka_adi") or row.get("banka_key") or "Unknown").strip(),
        source_url=source_url,
        canonical_url=canonical_url,
        canonical_group_key=sha256_text(canonical_url),
        archive_url=str(row.get("arsiv_url") or "").strip(),
        page_title=str(row.get("sayfa_basligi") or "").strip(),
        raw_text=raw_text,
        content_hash=content_digest(raw_text),
        snapshot_date=parse_date(row.get("anlik_goruntu_tarihi")),
        collected_at=parse_datetime(row.get("toplanma_tarihi")),
        source_category=str(row.get("kategori_tahmini") or "").strip(),
        is_campaign_hint=bool(row.get("kampanya_mi", False)),
    )


def iter_archive_documents(zip_path: Path) -> Iterator[ArchiveDocument]:
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset ZIP was not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        validate_zip_members(archive)
        member = find_archive_member(archive)
        with archive.open(member) as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid JSONL at {member}:{line_number}: {error}"
                    ) from error
                if bool(row.get("guncel_mi", True)):
                    continue
                yield row_to_document(row)


def ensure_archive_schema(connection) -> None:
    with connection.cursor() as cursor:
        for statement in SCHEMA_STATEMENTS:
            cursor.execute(statement)


def archive_tables_exist(connection) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT to_regclass('public.historical_documents') IS NOT NULL"
        )
        return bool(cursor.fetchone()[0])


def load_live_fingerprints(connection) -> tuple[set[str], set[str], set[str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COALESCE(record_key, ''),
                COALESCE(source_url, ''),
                COALESCE(raw_text, '')
            FROM documents
            """
        )
        rows = cursor.fetchall()
    record_keys = {str(row[0]).strip() for row in rows if str(row[0]).strip()}
    urls = {canonicalize_url(str(row[1])) for row in rows if str(row[1]).strip()}
    hashes = {content_digest(str(row[2])) for row in rows if str(row[2]).strip()}
    return record_keys, urls, hashes


def load_processed_archive_keys(connection) -> dict[str, tuple[str, str]]:
    if not archive_tables_exist(connection):
        return {}
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT archive_key, content_hash, pipeline_version
            FROM historical_documents
            """
        )
        return {
            str(key): (str(digest).strip(), str(version))
            for key, digest, version in cursor.fetchall()
        }


def make_fact_key(
    archive_key: str,
    fact_type: str,
    fact_text: str,
    evidence_text: str,
    extraction_method: str,
) -> str:
    material = "|".join(
        (
            archive_key,
            PIPELINE_VERSION,
            extraction_method,
            fact_type,
            fold_text(fact_text),
            fold_text(evidence_text),
        )
    )
    return sha256_text(material)


def product_title(product_code: str) -> str:
    return product_code.replace("_", " ").title()


def upsert_historical_result(
    connection,
    document: ArchiveDocument,
    classification: dict[str, Any],
    quality_status: str,
    facts: list[dict[str, Any]],
) -> int:
    from psycopg.types.json import Jsonb

    product = classification.get("product_type", {})
    campaign = classification.get("is_campaign", {})
    decision = str(classification.get("decision") or "FAILED").upper()
    if decision not in {"ACCEPTED", "REVIEW", "FAILED"}:
        decision = "FAILED"

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO banks (bank_key, bank_name)
            VALUES (%s, %s)
            ON CONFLICT (bank_key) DO UPDATE
            SET bank_name = EXCLUDED.bank_name
            RETURNING id
            """,
            (document.bank_key, document.bank_name),
        )
        bank_id = cursor.fetchone()[0]
        cursor.execute(
            """
            INSERT INTO historical_documents (
                archive_key,
                bank_id,
                source_url,
                canonical_url,
                canonical_group_key,
                archive_url,
                page_title,
                raw_text,
                content_hash,
                snapshot_date,
                collected_at,
                source_category,
                is_campaign_hint,
                campaign_label,
                campaign_confidence,
                product_type_code,
                product_type,
                classification_confidence,
                classification_decision,
                classification_basis,
                classification_payload,
                quality_status,
                searchable,
                pipeline_version,
                source_dataset
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (archive_key) DO UPDATE SET
                bank_id = EXCLUDED.bank_id,
                source_url = EXCLUDED.source_url,
                canonical_url = EXCLUDED.canonical_url,
                canonical_group_key = EXCLUDED.canonical_group_key,
                archive_url = EXCLUDED.archive_url,
                page_title = EXCLUDED.page_title,
                raw_text = EXCLUDED.raw_text,
                content_hash = EXCLUDED.content_hash,
                snapshot_date = EXCLUDED.snapshot_date,
                collected_at = EXCLUDED.collected_at,
                source_category = EXCLUDED.source_category,
                is_campaign_hint = EXCLUDED.is_campaign_hint,
                campaign_label = EXCLUDED.campaign_label,
                campaign_confidence = EXCLUDED.campaign_confidence,
                product_type_code = CASE
                    WHEN historical_documents.verified
                    THEN historical_documents.product_type_code
                    ELSE EXCLUDED.product_type_code
                END,
                product_type = CASE
                    WHEN historical_documents.verified
                    THEN historical_documents.product_type
                    ELSE EXCLUDED.product_type
                END,
                classification_confidence = CASE
                    WHEN historical_documents.verified
                    THEN historical_documents.classification_confidence
                    ELSE EXCLUDED.classification_confidence
                END,
                classification_decision = CASE
                    WHEN historical_documents.verified
                    THEN historical_documents.classification_decision
                    ELSE EXCLUDED.classification_decision
                END,
                classification_basis = CASE
                    WHEN historical_documents.verified
                    THEN historical_documents.classification_basis
                    ELSE EXCLUDED.classification_basis
                END,
                classification_payload = CASE
                    WHEN historical_documents.verified
                    THEN historical_documents.classification_payload
                    ELSE EXCLUDED.classification_payload
                END,
                quality_status = CASE
                    WHEN historical_documents.verified
                    THEN historical_documents.quality_status
                    ELSE EXCLUDED.quality_status
                END,
                searchable = CASE
                    WHEN historical_documents.verified
                    THEN historical_documents.searchable
                    ELSE EXCLUDED.searchable
                END,
                pipeline_version = EXCLUDED.pipeline_version,
                source_dataset = EXCLUDED.source_dataset,
                updated_at = NOW()
            RETURNING id
            """,
            (
                document.archive_key,
                bank_id,
                document.source_url,
                document.canonical_url,
                document.canonical_group_key,
                document.archive_url or None,
                document.page_title or None,
                document.raw_text,
                document.content_hash,
                document.snapshot_date,
                document.collected_at,
                document.source_category or None,
                document.is_campaign_hint,
                campaign.get("label"),
                float(campaign["score"]) if campaign.get("score") is not None else None,
                product.get("label"),
                product_title(str(product.get("label") or "")),
                float(product["score"]) if product.get("score") is not None else None,
                decision,
                classification.get("decision_basis"),
                Jsonb(classification),
                quality_status,
                quality_status == "accepted",
                PIPELINE_VERSION,
                SOURCE_DATASET,
            ),
        )
        document_id = int(cursor.fetchone()[0])

        cursor.execute(
            """
            DELETE FROM historical_facts
            WHERE historical_document_id = %s
              AND pipeline_version = %s
              AND review_status IN ('pending', 'not_required')
            """,
            (document_id, PIPELINE_VERSION),
        )
        for fact in facts:
            decision_value = str(fact["decision"])
            review_status = (
                "pending" if decision_value == "review" else "not_required"
            )
            fact_key = make_fact_key(
                document.archive_key,
                str(fact["fact_type"]),
                str(fact["fact_text"]),
                str(fact["evidence_text"]),
                str(fact["extraction_method"]),
            )
            cursor.execute(
                """
                INSERT INTO historical_facts (
                    historical_document_id,
                    fact_type,
                    fact_text,
                    normalized_value,
                    evidence_text,
                    extraction_method,
                    confidence,
                    source_chunk,
                    decision,
                    decision_reason,
                    review_status,
                    fact_key,
                    pipeline_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (historical_document_id, fact_key) DO UPDATE SET
                    normalized_value = EXCLUDED.normalized_value,
                    evidence_text = EXCLUDED.evidence_text,
                    confidence = EXCLUDED.confidence,
                    source_chunk = EXCLUDED.source_chunk,
                    decision = EXCLUDED.decision,
                    decision_reason = EXCLUDED.decision_reason,
                    review_status = EXCLUDED.review_status,
                    updated_at = NOW()
                """,
                (
                    document_id,
                    fact["fact_type"],
                    fact["fact_text"],
                    Jsonb(fact.get("normalized_value"))
                    if fact.get("normalized_value") is not None
                    else None,
                    fact["evidence_text"],
                    fact["extraction_method"],
                    float(fact["confidence"]),
                    int(fact.get("source_chunk", 0)),
                    decision_value,
                    fact["decision_reason"],
                    review_status,
                    fact_key,
                    PIPELINE_VERSION,
                ),
            )
    return document_id


def open_connection():
    from hybrid_search import get_connection

    return get_connection()
