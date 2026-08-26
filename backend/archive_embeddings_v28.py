from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice

from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from archive_common_v28 import (
    PIPELINE_VERSION,
    archive_tables_exist,
    ensure_archive_schema,
    open_connection,
)
from generate_embeddings import (
    MODEL_NAME,
    build_chunks,
    encode_chunks,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BGE-M3 embeddings for accepted historical documents. "
            "No database write occurs unless --write is supplied."
        )
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--document-batch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.offset < 0:
        parser.error("--limit and --offset cannot be negative")
    if args.document_batch < 1 or args.batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.max_tokens < 64:
        parser.error("--max-tokens must be at least 64")
    if args.overlap_tokens < 0 or args.overlap_tokens >= args.max_tokens:
        parser.error("--overlap-tokens must be smaller than --max-tokens")
    if args.test and args.write:
        parser.error("--test and --write cannot be used together")
    return args


def load_documents(args: argparse.Namespace) -> list[tuple]:
    with open_connection() as connection:
        if not archive_tables_exist(connection):
            raise RuntimeError(
                "historical_documents was not found. Run archive_ingest_v28.py "
                "with --write first."
            )
        filters = ["d.quality_status = 'accepted'", "d.searchable IS TRUE"]
        if not args.force:
            filters.append(
                "NOT EXISTS ("
                "SELECT 1 FROM historical_document_chunks c "
                "WHERE c.historical_document_id = d.id "
                "AND c.embedding IS NOT NULL"
                ")"
            )
        query = f"""
            SELECT
                d.id,
                b.bank_name,
                COALESCE(d.page_title, ''),
                d.raw_text,
                d.archive_key,
                d.snapshot_date
            FROM historical_documents d
            JOIN banks b ON b.id = d.bank_id
            WHERE {' AND '.join(filters)}
            ORDER BY d.id
        """
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    stop = None if args.limit == 0 else args.offset + args.limit
    rows = rows[args.offset:stop]
    if args.test:
        rows = rows[:3]
    return rows


def batched(values: list, size: int):
    iterator = iter(values)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


def save_chunks(rows: list[tuple], chunks, embeddings, force: bool) -> int:
    metadata_by_id = {
        int(row[0]): {
            "archive_key": str(row[4]),
            "snapshot_date": row[5].isoformat() if row[5] else None,
            "source": PIPELINE_VERSION,
        }
        for row in rows
    }
    document_ids = sorted(metadata_by_id)
    insert_rows = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        content_hash = hashlib.sha256(
            chunk.content.encode("utf-8")
        ).hexdigest()
        metadata = metadata_by_id[chunk.document_id]
        insert_rows.append(
            (
                chunk.document_id,
                chunk.chunk_index,
                chunk.content,
                chunk.token_count,
                content_hash,
                MODEL_NAME,
                embedding,
                chunk.content,
                chunk.content,
                Jsonb(metadata),
            )
        )

    with open_connection() as connection:
        ensure_archive_schema(connection)
        with connection.cursor() as cursor:
            if force:
                cursor.execute(
                    """
                    DELETE FROM historical_document_chunks
                    WHERE historical_document_id = ANY(%s)
                    """,
                    (document_ids,),
                )
            cursor.executemany(
                """
                INSERT INTO historical_document_chunks (
                    historical_document_id,
                    chunk_index,
                    content,
                    token_count,
                    content_hash,
                    embedding_model,
                    embedding,
                    search_vector,
                    metadata
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    to_tsvector('simple', %s)
                        || to_tsvector('turkish', %s),
                    %s
                )
                ON CONFLICT (historical_document_id, chunk_index) DO UPDATE SET
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    content_hash = EXCLUDED.content_hash,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding = EXCLUDED.embedding,
                    search_vector = EXCLUDED.search_vector,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                insert_rows,
            )
    return len(insert_rows)


def main() -> None:
    load_dotenv()
    args = parse_args()
    documents = load_documents(args)
    print("Pipeline:", PIPELINE_VERSION)
    print("Mode:", "TEST" if args.test else "WRITE" if args.write else "DRY_RUN")
    print("Eligible historical documents:", len(documents))
    if not documents:
        print("Nothing to embed.")
        return
    if not args.test and not args.write:
        print("Database was not changed. Use --test or --write to load the model.")
        return

    model = load_model()
    total_chunks = 0
    for batch_index, rows in enumerate(
        batched(documents, args.document_batch),
        start=1,
    ):
        model_rows = [(row[0], row[1], row[2], row[3]) for row in rows]
        chunks = build_chunks(
            model_rows,
            model.tokenizer,
            args.max_tokens,
            args.overlap_tokens,
        )
        embeddings = encode_chunks(model, chunks, args.batch_size)
        print(
            f"Batch {batch_index}: documents={len(rows)} "
            f"chunks={len(chunks)} shape={tuple(embeddings.shape)}"
        )
        if args.write:
            total_chunks += save_chunks(rows, chunks, embeddings, args.force)
        else:
            total_chunks += len(chunks)

    summary = {
        "mode": "test" if args.test else "write",
        "documents": len(documents),
        "chunks": total_chunks,
        "model": MODEL_NAME,
    }
    print("Summary:", json.dumps(summary, sort_keys=True))
    if args.test:
        print("Test completed. Database was not changed.")


if __name__ == "__main__":
    main()
