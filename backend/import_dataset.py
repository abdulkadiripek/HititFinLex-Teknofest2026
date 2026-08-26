import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb


DATASET_VERSION = "v1"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error


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


def check_required_files(data_dir: Path):
    required_files = [
        data_dir / "ara" / "dokumanlar.jsonl",
        data_dir / "ara" / "pasajlar.jsonl",
        data_dir / "final" / "siniflandirma" / "train.jsonl",
        data_dir / "final" / "siniflandirma" / "val.jsonl",
        data_dir / "final" / "siniflandirma" / "test.jsonl",
        data_dir / "final" / "bilgi_cikarim" / "train.jsonl",
        data_dir / "final" / "bilgi_cikarim" / "val.jsonl",
        data_dir / "final" / "bilgi_cikarim" / "test.jsonl",
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        missing = "\n".join(missing_files)
        raise FileNotFoundError(f"Required dataset files not found:\n{missing}")


def import_documents(connection, data_dir: Path):
    document_ids = {}
    bank_ids = {}
    path = data_dir / "ara" / "dokumanlar.jsonl"
    count = 0

    with connection.cursor() as cursor:
        for row in read_jsonl(path):
            bank_key = row["banka_key"]

            cursor.execute(
                """
                INSERT INTO banks (bank_key, bank_name)
                VALUES (%s, %s)
                ON CONFLICT (bank_key) DO UPDATE
                SET bank_name = EXCLUDED.bank_name
                RETURNING id
                """,
                (bank_key, row["banka_adi"]),
            )
            bank_id = cursor.fetchone()[0]
            bank_ids[bank_key] = bank_id

            cursor.execute(
                """
                INSERT INTO documents (
                    record_key,
                    bank_id,
                    source_url,
                    page_title,
                    raw_text,
                    summary_text,
                    campaign_type_code,
                    campaign_type,
                    confidence,
                    label_source,
                    rationale,
                    verified,
                    auto_accepted,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, NOW()
                )
                ON CONFLICT (record_key) DO UPDATE SET
                    bank_id = EXCLUDED.bank_id,
                    source_url = EXCLUDED.source_url,
                    page_title = EXCLUDED.page_title,
                    raw_text = EXCLUDED.raw_text,
                    summary_text = EXCLUDED.summary_text,
                    campaign_type_code = EXCLUDED.campaign_type_code,
                    campaign_type = EXCLUDED.campaign_type,
                    confidence = EXCLUDED.confidence,
                    label_source = EXCLUDED.label_source,
                    rationale = EXCLUDED.rationale,
                    verified = EXCLUDED.verified,
                    auto_accepted = EXCLUDED.auto_accepted,
                    updated_at = NOW()
                RETURNING id
                """,
                (
                    row["kayit_id"],
                    bank_id,
                    row["kaynak_url"],
                    row.get("sayfa_basligi"),
                    row["ham_metin"],
                    row.get("ozet_metin"),
                    row.get("kampanya_turu_kod"),
                    row.get("kampanya_turu"),
                    row.get("guven"),
                    row.get("kaynak"),
                    row.get("gerekce"),
                    bool(row.get("dogrulandi", False)),
                    bool(row.get("otomatik_kabul", False)),
                ),
            )
            document_ids[row["kayit_id"]] = cursor.fetchone()[0]
            count += 1

    return document_ids, bank_ids, count


def import_passages_and_entities(connection, data_dir: Path, document_ids):
    passage_ids = {}
    passage_count = 0
    entity_count = 0
    path = data_dir / "ara" / "pasajlar.jsonl"

    with connection.cursor() as cursor:
        for row in read_jsonl(path):
            record_key = row["kayit_id"]
            if record_key not in document_ids:
                raise KeyError(f"Document not found for passage: {record_key}")

            cursor.execute(
                """
                INSERT INTO passages (
                    passage_key,
                    document_id,
                    campaign_type_code,
                    content,
                    document_offset,
                    verified,
                    auto_accepted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (passage_key) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    campaign_type_code = EXCLUDED.campaign_type_code,
                    content = EXCLUDED.content,
                    document_offset = EXCLUDED.document_offset,
                    verified = EXCLUDED.verified,
                    auto_accepted = EXCLUDED.auto_accepted
                RETURNING id
                """,
                (
                    row["pasaj_id"],
                    document_ids[record_key],
                    row.get("kampanya_turu_kod"),
                    row["metin"],
                    row.get("belge_ofseti"),
                    bool(row.get("dogrulandi", False)),
                    bool(row.get("otomatik_kabul", False)),
                ),
            )
            passage_id = cursor.fetchone()[0]
            passage_ids[row["pasaj_id"]] = passage_id
            passage_count += 1

            cursor.execute(
                "DELETE FROM entities WHERE passage_id = %s",
                (passage_id,),
            )

            for span_index, span in enumerate(row.get("spanlar", [])):
                cursor.execute(
                    """
                    INSERT INTO entities (
                        passage_id,
                        span_index,
                        start_offset,
                        end_offset,
                        entity_label,
                        entity_text,
                        normalized_value,
                        confidence,
                        extraction_source,
                        verified
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        passage_id,
                        span_index,
                        span["baslangic"],
                        span["bitis"],
                        span["etiket"],
                        span["metin"],
                        Jsonb(span["normal_deger"])
                        if span.get("normal_deger") is not None
                        else None,
                        span.get("guven"),
                        span.get("kaynak"),
                        bool(row.get("dogrulandi", False)),
                    ),
                )
                entity_count += 1

    return passage_ids, passage_count, entity_count


def import_classification_samples(connection, data_dir: Path, document_ids):
    count = 0
    split_paths = {
        "train": data_dir / "final" / "siniflandirma" / "train.jsonl",
        "val": data_dir / "final" / "siniflandirma" / "val.jsonl",
        "test": data_dir / "final" / "siniflandirma" / "test.jsonl",
    }

    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM classification_samples WHERE dataset_version = %s",
            (DATASET_VERSION,),
        )

        for data_split, path in split_paths.items():
            for row in read_jsonl(path):
                record_key = row["id"]
                if record_key not in document_ids:
                    raise KeyError(
                        f"Document not found for classification sample: {record_key}"
                    )

                cursor.execute(
                    """
                    INSERT INTO classification_samples (
                        document_id,
                        dataset_version,
                        data_split,
                        label_name,
                        label_code,
                        verified
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        document_ids[record_key],
                        DATASET_VERSION,
                        data_split,
                        row["etiket"],
                        row["etiket_kod"],
                        bool(row.get("dogrulandi", False)),
                    ),
                )
                count += 1

    return count


def import_extraction_samples(connection, data_dir: Path, passage_ids):
    count = 0
    split_paths = {
        "train": data_dir / "final" / "bilgi_cikarim" / "train.jsonl",
        "val": data_dir / "final" / "bilgi_cikarim" / "val.jsonl",
        "test": data_dir / "final" / "bilgi_cikarim" / "test.jsonl",
    }

    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM extraction_samples WHERE dataset_version = %s",
            (DATASET_VERSION,),
        )

        for data_split, path in split_paths.items():
            for row in read_jsonl(path):
                passage_key = row["id"]
                if passage_key not in passage_ids:
                    raise KeyError(
                        f"Passage not found for extraction sample: {passage_key}"
                    )

                cursor.execute(
                    """
                    INSERT INTO extraction_samples (
                        passage_id,
                        dataset_version,
                        data_split,
                        verified
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        passage_ids[passage_key],
                        DATASET_VERSION,
                        data_split,
                        bool(row.get("dogrulandi", False)),
                    ),
                )
                count += 1

    return count


def verify_counts(connection):
    table_names = [
        "banks",
        "documents",
        "passages",
        "entities",
        "classification_samples",
        "extraction_samples",
    ]
    counts = {}

    with connection.cursor() as cursor:
        for table_name in table_names:
            cursor.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            )
            counts[table_name] = cursor.fetchone()[0]

    return counts


def main():
    load_dotenv()
    data_dir = Path(os.getenv("DATA_DIR", "data")).expanduser().resolve()
    check_required_files(data_dir)

    print("Dataset directory:", data_dir)
    print("Starting database import...")

    with get_connection() as connection:
        document_ids, _, document_count = import_documents(
            connection, data_dir
        )
        passage_ids, passage_count, entity_count = import_passages_and_entities(
            connection, data_dir, document_ids
        )
        classification_count = import_classification_samples(
            connection, data_dir, document_ids
        )
        extraction_count = import_extraction_samples(
            connection, data_dir, passage_ids
        )
        counts = verify_counts(connection)

    print("Import completed successfully")
    print("Imported documents:", document_count)
    print("Imported passages:", passage_count)
    print("Imported entities:", entity_count)
    print("Imported classification samples:", classification_count)
    print("Imported extraction samples:", extraction_count)
    print("Database counts:")
    for table_name, count in counts.items():
        print(f"  {table_name}: {count}")


if __name__ == "__main__":
    main()