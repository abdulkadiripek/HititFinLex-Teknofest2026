import argparse
import atexit
import os
import re
import textwrap
from threading import Lock

import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = "BAAI/bge-m3"
RRF_CONSTANT = int(os.getenv("RAG_V2_RRF_K", "60"))
DENSE_WEIGHT = float(os.getenv("RAG_V2_DENSE_WEIGHT", "1.0"))
LEXICAL_WEIGHT = float(os.getenv("RAG_V2_LEXICAL_WEIGHT", "0.5"))
MAX_RESULTS = 50
PREVIEW_LENGTH = 700
_connection_pool = None
_connection_pool_lock = Lock()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid BGE-M3 and PostgreSQL full-text search."
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Natural-language search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of unique documents to return. Default: 5.",
    )
    return parser.parse_args()


def _connection_parameters():
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

    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ["DB_PORT"]),
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


def _configure_connection(connection):
    register_vector(connection)
    connection.commit()


def get_connection():
    global _connection_pool
    with _connection_pool_lock:
        if _connection_pool is None:
            minimum = max(1, int(os.getenv("DB_POOL_MIN_SIZE", "1")))
            maximum = max(minimum, int(os.getenv("DB_POOL_MAX_SIZE", "10")))
            _connection_pool = ConnectionPool(
                conninfo="",
                kwargs=_connection_parameters(),
                min_size=minimum,
                max_size=maximum,
                timeout=float(os.getenv("DB_POOL_TIMEOUT_SECONDS", "10")),
                configure=_configure_connection,
                open=False,
                name="hititfinlex-legacy",
            )
            _connection_pool.open(wait=True)
    return _connection_pool.connection()


def close_connection_pool():
    global _connection_pool
    with _connection_pool_lock:
        if _connection_pool is not None:
            _connection_pool.close()
            _connection_pool = None


atexit.register(close_connection_pool)


def inspect_chunk_table(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'document_chunks'
            """
        )
        columns = {row[0] for row in cursor.fetchall()}

    text_column = next(
        (
            name
            for name in ("content", "chunk_text", "text")
            if name in columns
        ),
        None,
    )
    if text_column is None:
        names = ", ".join(sorted(columns)) or "TABLE_NOT_FOUND"
        raise RuntimeError(
            "No supported text column was found in document_chunks. "
            f"Available columns: {names}"
        )
    if "search_vector" not in columns:
        raise RuntimeError(
            "document_chunks.search_vector was not found."
        )
    return text_column


def load_model():
    configured_device = os.getenv("LOCAL_EMBEDDING_DEVICE", "").strip().lower()
    if configured_device not in {"", "cpu", "cuda"}:
        raise ValueError("LOCAL_EMBEDDING_DEVICE must be cpu or cuda")
    device = configured_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 512
    if device == "cuda":
        model.half()
    return model


def encode_query(model, query):
    vector = model.encode(
        query,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if vector.ndim != 1 or vector.shape[0] != 1024:
        raise RuntimeError(
            f"Expected a 1024-dimensional query vector, got {vector.shape}"
        )
    return vector


def build_lexical_query(query):
    words = re.findall(r"\w+", query, flags=re.UNICODE)
    words = [word for word in words if len(word) >= 3]
    if not words:
        return query
    return " OR ".join(f'"{word}"' for word in words)


def search_database(
    connection,
    text_column,
    query_vector,
    lexical_query,
    top_k,
):
    candidate_count = min(max(top_k * 10, 50), 500)

    search_query = sql.SQL(
        """
        WITH query_input AS (
            SELECT
                %s::vector AS query_vector,
                websearch_to_tsquery('simple', %s)
                || websearch_to_tsquery('turkish', %s) AS text_query
        ),
        semantic_candidates AS (
            SELECT
                chunks.id,
                1 - (
                    chunks.embedding <=> query_input.query_vector
                ) AS semantic_similarity,
                ROW_NUMBER() OVER (
                    ORDER BY chunks.embedding <=> query_input.query_vector
                ) AS semantic_rank
            FROM document_chunks AS chunks
            CROSS JOIN query_input
            WHERE chunks.embedding IS NOT NULL
            ORDER BY chunks.embedding <=> query_input.query_vector
            LIMIT %s
        ),
        lexical_candidates AS (
            SELECT
                chunks.id,
                ts_rank_cd(
                    chunks.search_vector,
                    query_input.text_query
                ) AS lexical_score,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(
                        chunks.search_vector,
                        query_input.text_query
                    ) DESC
                ) AS lexical_rank
            FROM document_chunks AS chunks
            CROSS JOIN query_input
            WHERE chunks.search_vector @@ query_input.text_query
            ORDER BY lexical_score DESC
            LIMIT %s
        ),
        candidate_ids AS (
            SELECT id FROM semantic_candidates
            UNION
            SELECT id FROM lexical_candidates
        ),
        fused AS (
            SELECT
                candidate_ids.id,
                semantic_candidates.semantic_similarity,
                lexical_candidates.lexical_score,
                COALESCE(
                    %s / (%s + semantic_candidates.semantic_rank),
                    0.0
                )
                + COALESCE(
                    %s / (%s + lexical_candidates.lexical_rank),
                    0.0
                ) AS hybrid_score
            FROM candidate_ids
            LEFT JOIN semantic_candidates
                ON semantic_candidates.id = candidate_ids.id
            LEFT JOIN lexical_candidates
                ON lexical_candidates.id = candidate_ids.id
        ),
        ranked_documents AS (
            SELECT
                chunks.id AS chunk_id,
                documents.id AS document_id,
                banks.bank_name,
                documents.page_title,
                documents.source_url,
                chunks.{text_column} AS content,
                fused.semantic_similarity,
                fused.lexical_score,
                fused.hybrid_score,
                documents.verified,
                ROW_NUMBER() OVER (
                    PARTITION BY documents.id
                    ORDER BY fused.hybrid_score DESC
                ) AS document_rank
            FROM fused
            JOIN document_chunks AS chunks
                ON chunks.id = fused.id
            JOIN documents
                ON documents.id = chunks.document_id
            JOIN banks
                ON banks.id = documents.bank_id
        )
        SELECT
            chunk_id,
            bank_name,
            page_title,
            source_url,
            content,
            semantic_similarity,
            lexical_score,
            hybrid_score,
            verified
        FROM ranked_documents
        WHERE document_rank = 1
        ORDER BY hybrid_score DESC
        LIMIT %s
        """
    ).format(text_column=sql.Identifier(text_column))

    parameters = (
        query_vector,
        lexical_query,
        lexical_query,
        candidate_count,
        candidate_count,
        DENSE_WEIGHT,
        RRF_CONSTANT,
        LEXICAL_WEIGHT,
        RRF_CONSTANT,
        top_k,
    )

    with connection.cursor() as cursor:
        cursor.execute(search_query, parameters)
        return cursor.fetchall()


def print_results(query, rows):
    print()
    print("Query:", query)
    print("Mode: hybrid (semantic + full-text RRF)")
    print("Unique documents:", len(rows))

    if not rows:
        print("No matching documents were found.")
        return

    for rank, row in enumerate(rows, start=1):
        (
            _,
            bank_name,
            page_title,
            source_url,
            content,
            semantic_similarity,
            lexical_score,
            hybrid_score,
            verified,
        ) = row

        preview = " ".join(content.split())
        preview = textwrap.shorten(
            preview,
            width=PREVIEW_LENGTH,
            placeholder="...",
        )
        semantic_value = (
            float(semantic_similarity)
            if semantic_similarity is not None
            else 0.0
        )
        lexical_value = (
            float(lexical_score) if lexical_score is not None else 0.0
        )

        print()
        print(
            f"[{rank}] Hybrid: {float(hybrid_score):.6f} | "
            f"Semantic: {semantic_value:.4f} | "
            f"Lexical: {lexical_value:.4f}"
        )
        print("Banka:", bank_name or "-")
        print("Baslik:", page_title or "-")
        print("Kaynak:", source_url or "-")
        print("Dogrulandi:", "evet" if verified else "hayir")
        print("Metin:", preview)


def main():
    load_dotenv()
    args = parse_args()
    query = " ".join(args.query).strip()

    if not query:
        raise ValueError("Query cannot be empty.")
    if not 1 <= args.top_k <= MAX_RESULTS:
        raise ValueError(f"top-k must be between 1 and {MAX_RESULTS}")

    print("Loading embedding model...")
    model = load_model()
    device = getattr(model, "device", "unknown")
    print("Device:", device)

    query_vector = encode_query(model, query)
    lexical_query = build_lexical_query(query)

    with get_connection() as connection:
        text_column = inspect_chunk_table(connection)
        rows = search_database(
            connection,
            text_column,
            query_vector,
            lexical_query,
            args.top_k,
        )

    print_results(query, rows)


if __name__ == "__main__":
    main()
