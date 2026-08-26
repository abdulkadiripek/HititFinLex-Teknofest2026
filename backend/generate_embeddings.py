import argparse
import os
from dataclasses import dataclass

import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
DEFAULT_MAX_TOKENS = 384
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_BATCH_SIZE = 16


@dataclass
class Chunk:
    document_id: int
    chunk_index: int
    content: str
    token_count: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate BGE-M3 embeddings for PostgreSQL pgvector."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Encode three documents without changing the database.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
    )
    return parser.parse_args()


def get_connection():
    required_variables = [
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]
    missing_variables = [
        name for name in required_variables if not os.getenv(name)
    ]
    if missing_variables:
        names = ", ".join(missing_variables)
        raise RuntimeError(f"Missing environment variables: {names}")

    return psycopg.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


def validate_arguments(args):
    if args.batch_size < 1:
        raise ValueError("batch-size must be greater than zero")
    if args.max_tokens < 64:
        raise ValueError("max-tokens must be at least 64")
    if args.overlap_tokens < 0:
        raise ValueError("overlap-tokens cannot be negative")
    if args.overlap_tokens >= args.max_tokens:
        raise ValueError("overlap-tokens must be smaller than max-tokens")


def inspect_chunk_table(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'document_chunks'
            ORDER BY ordinal_position
            """
        )
        columns = {row[0] for row in cursor.fetchall()}

        if not columns:
            raise RuntimeError(
                "Table public.document_chunks was not found."
            )

        required_columns = {"document_id", "chunk_index", "embedding"}
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            names = ", ".join(missing_columns)
            raise RuntimeError(
                f"document_chunks is missing required columns: {names}"
            )

        text_column = next(
            (
                name
                for name in ("content", "chunk_text", "text")
                if name in columns
            ),
            None,
        )
        if text_column is None:
            names = ", ".join(sorted(columns))
            raise RuntimeError(
                "No supported text column was found in document_chunks. "
                f"Available columns: {names}"
            )

        cursor.execute(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation
              ON relation.oid = attribute.attrelid
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'document_chunks'
              AND attribute.attname = 'embedding'
              AND NOT attribute.attisdropped
            """
        )
        embedding_type_row = cursor.fetchone()
        embedding_type = (
            embedding_type_row[0] if embedding_type_row else None
        )
        if embedding_type != "vector(1024)":
            raise RuntimeError(
                "document_chunks.embedding must be vector(1024), "
                f"but found: {embedding_type}"
            )

    return text_column, columns


def load_documents(limit=None):
    query = """
        SELECT
            documents.id,
            COALESCE(banks.bank_name, banks.bank_key, ''),
            COALESCE(documents.page_title, ''),
            COALESCE(
                NULLIF(BTRIM(documents.raw_text), ''),
                NULLIF(BTRIM(documents.summary_text), ''),
                NULLIF(BTRIM(documents.page_title), ''),
                ''
            )
        FROM documents
        JOIN banks ON banks.id = documents.bank_id
        ORDER BY documents.id
    """

    with get_connection() as connection:
        register_vector(connection)
        text_column, table_columns = inspect_chunk_table(connection)
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    documents = [row for row in rows if row[3].strip()]
    if limit is not None:
        documents = documents[:limit]
    if not documents:
        raise RuntimeError("No non-empty documents were found.")

    return documents, text_column, table_columns


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script does not use CPU fallback."
        )

    torch.set_float32_matmul_precision("high")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.max_seq_length = 512
    model.half()

    print("Model:", MODEL_NAME)
    print("Device:", model.device)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Allocated VRAM (MB):",
        round(torch.cuda.memory_allocated(0) / 1024**2, 1),
    )
    return model


def build_chunks(
    documents,
    tokenizer,
    max_tokens,
    overlap_tokens,
):
    chunks = []

    for document_id, bank_name, page_title, raw_text in documents:
        header_parts = []
        if bank_name.strip():
            header_parts.append(f"Banka: {bank_name.strip()}")
        if page_title.strip():
            header_parts.append(f"Baslik: {page_title.strip()}")

        header = "\n".join(header_parts)
        header_token_ids = tokenizer.encode(
            header,
            add_special_tokens=False,
        )[:64]
        header = tokenizer.decode(
            header_token_ids,
            skip_special_tokens=True,
        ).strip()

        body_token_ids = tokenizer.encode(
            raw_text,
            add_special_tokens=False,
        )
        body_capacity = max_tokens - len(header_token_ids)
        if body_capacity <= overlap_tokens:
            raise RuntimeError(
                "Document header leaves no room for chunk content."
            )

        step = body_capacity - overlap_tokens
        chunk_index = 0

        for start in range(0, len(body_token_ids), step):
            window = body_token_ids[start : start + body_capacity]
            if not window:
                break

            body = tokenizer.decode(
                window,
                skip_special_tokens=True,
            ).strip()
            content = "\n".join(
                part for part in (header, body) if part
            ).strip()

            if content:
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=content,
                        token_count=len(header_token_ids) + len(window),
                    )
                )
                chunk_index += 1

            if start + body_capacity >= len(body_token_ids):
                break

    if not chunks:
        raise RuntimeError("No chunks were generated.")
    return chunks


def encode_chunks(model, chunks, batch_size):
    contents = [chunk.content for chunk in chunks]
    embeddings = model.encode(
        contents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise RuntimeError(
            f"Expected embeddings with shape (n, 1024), got {embeddings.shape}"
        )
    return embeddings


def save_chunks(chunks, embeddings, text_column, table_columns):
    insert_columns = [
        "document_id",
        "chunk_index",
        text_column,
        "embedding",
    ]
    include_token_count = "token_count" in table_columns
    include_model_name = "embedding_model" in table_columns

    if include_token_count:
        insert_columns.append("token_count")
    if include_model_name:
        insert_columns.append("embedding_model")

    insert_query = sql.SQL(
        "INSERT INTO document_chunks ({columns}) VALUES ({values})"
    ).format(
        columns=sql.SQL(", ").join(
            sql.Identifier(name) for name in insert_columns
        ),
        values=sql.SQL(", ").join(
            sql.Placeholder() for _ in insert_columns
        ),
    )

    rows = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        row = [
            chunk.document_id,
            chunk.chunk_index,
            chunk.content,
            embedding,
        ]
        if include_token_count:
            row.append(chunk.token_count)
        if include_model_name:
            row.append(MODEL_NAME)
        rows.append(tuple(row))

    with get_connection() as connection:
        register_vector(connection)
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM document_chunks")
                cursor.executemany(insert_query, rows)

                cursor.execute(
                    "SELECT COUNT(*), COUNT(embedding) FROM document_chunks"
                )
                total_count, embedded_count = cursor.fetchone()

    if total_count != len(chunks) or embedded_count != len(chunks):
        raise RuntimeError(
            "Database verification failed after inserting embeddings."
        )
    return total_count


def main():
    load_dotenv()
    args = parse_args()
    validate_arguments(args)

    document_limit = 3 if args.test else None
    documents, text_column, table_columns = load_documents(document_limit)
    model = load_model()
    chunks = build_chunks(
        documents,
        model.tokenizer,
        args.max_tokens,
        args.overlap_tokens,
    )

    print("Documents:", len(documents))
    print("Chunks:", len(chunks))
    print("Max tokens:", args.max_tokens)
    print("Overlap tokens:", args.overlap_tokens)

    embeddings = encode_chunks(model, chunks, args.batch_size)
    print("Embedding shape:", embeddings.shape)

    if args.test:
        print("Test completed. Database was not changed.")
        return

    saved_count = save_chunks(
        chunks,
        embeddings,
        text_column,
        table_columns,
    )
    print("Saved chunks:", saved_count)
    print("Embedding generation completed successfully.")


if __name__ == "__main__":
    main()