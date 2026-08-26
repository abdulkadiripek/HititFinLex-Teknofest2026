import argparse
import os
import textwrap

import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
MAX_RESULTS = 50
PREVIEW_LENGTH = 700


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search document chunks with BGE-M3 and pgvector."
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
        help="Number of results to return. Default: 5.",
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

    connection = psycopg.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )
    register_vector(connection)
    return connection


def find_text_column(connection):
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

    for candidate in ("content", "chunk_text", "text"):
        if candidate in columns:
            return candidate

    names = ", ".join(sorted(columns)) or "TABLE_NOT_FOUND"
    raise RuntimeError(
        "No supported text column was found in document_chunks. "
        f"Available columns: {names}"
    )


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script does not use CPU fallback."
        )

    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.max_seq_length = 512
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


def search_database(connection, text_column, query_vector, top_k):
    search_query = sql.SQL(
        """
        SELECT
            chunks.id,
            banks.bank_name,
            documents.page_title,
            documents.source_url,
            chunks.{text_column},
            1 - (chunks.embedding <=> %s) AS similarity
        FROM document_chunks AS chunks
        JOIN documents ON documents.id = chunks.document_id
        JOIN banks ON banks.id = documents.bank_id
        WHERE chunks.embedding IS NOT NULL
        ORDER BY chunks.embedding <=> %s
        LIMIT %s
        """
    ).format(text_column=sql.Identifier(text_column))

    with connection.cursor() as cursor:
        cursor.execute(
            search_query,
            (query_vector, query_vector, top_k),
        )
        return cursor.fetchall()


def print_results(query, rows):
    print()
    print("Query:", query)
    print("Results:", len(rows))

    if not rows:
        print("No matching chunks were found.")
        return

    for rank, row in enumerate(rows, start=1):
        _, bank_name, page_title, source_url, content, similarity = row
        preview = " ".join(content.split())
        preview = textwrap.shorten(
            preview,
            width=PREVIEW_LENGTH,
            placeholder="...",
        )

        print()
        print(f"[{rank}] Similarity: {float(similarity):.4f}")
        print("Banka:", bank_name or "-")
        print("Baslik:", page_title or "-")
        print("Kaynak:", source_url or "-")
        print("Metin:", preview)


def main():
    load_dotenv()
    args = parse_args()
    query = " ".join(args.query).strip()

    if not query:
        raise ValueError("Query cannot be empty.")
    if not 1 <= args.top_k <= MAX_RESULTS:
        raise ValueError(f"top-k must be between 1 and {MAX_RESULTS}")

    print("Loading model on GPU...")
    model = load_model()
    print("GPU:", torch.cuda.get_device_name(0))

    query_vector = encode_query(model, query)
    with get_connection() as connection:
        text_column = find_text_column(connection)
        rows = search_database(
            connection,
            text_column,
            query_vector,
            args.top_k,
        )

    print_results(query, rows)


if __name__ == "__main__":
    main()