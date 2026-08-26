from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from archive_common_v28 import ensure_archive_schema, open_connection, product_title


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply completed historical classification and fact review CSV files. "
            "Dry-run is the default."
        )
    )
    parser.add_argument(
        "--classification-csv",
        default="data/archive_v28_training/archive_classification_review.csv",
    )
    parser.add_argument(
        "--fact-csv",
        default="data/archive_v28_training/archive_fact_review.csv",
    )
    parser.add_argument("--write", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def answer(value: str | None) -> str:
    return (value or "").strip().upper()


def resolve_classifications(
    connection,
    rows: list[dict[str, str]],
    write: bool,
) -> Counter:
    counters = Counter()
    for row in rows:
        decision = answer(row.get("ONAY_E_H"))
        corrected = answer(row.get("DUZELTILMIS_ETIKET"))
        if decision not in {"E", "H"} and not corrected:
            counters["classification_unanswered"] += 1
            continue
        archive_key = (row.get("archive_key") or "").strip()
        suggested = answer(row.get("suggested_product_type"))
        if corrected or decision == "E":
            product_code = corrected or suggested
            if not product_code:
                counters["classification_invalid"] += 1
                continue
            action = "approved"
            if write:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE historical_documents
                        SET product_type_code = %s,
                            product_type = %s,
                            classification_confidence = 1.0,
                            classification_decision = 'ACCEPTED',
                            classification_basis = 'human_review',
                            quality_status = 'accepted',
                            searchable = TRUE,
                            verified = TRUE,
                            verification_source = 'archive_review_v28',
                            updated_at = NOW()
                        WHERE archive_key = %s
                        """,
                        (product_code, product_title(product_code), archive_key),
                    )
                    counters["classification_rows_updated"] += cursor.rowcount
        else:
            action = "rejected"
            if write:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE historical_documents
                        SET classification_decision = 'FAILED',
                            classification_basis = 'human_reject',
                            quality_status = 'failed',
                            searchable = FALSE,
                            verified = TRUE,
                            verification_source = 'archive_review_v28',
                            updated_at = NOW()
                        WHERE archive_key = %s
                        """,
                        (archive_key,),
                    )
                    counters["classification_rows_updated"] += cursor.rowcount
        counters["classification_" + action] += 1
    return counters


def resolve_facts(
    connection,
    rows: list[dict[str, str]],
    write: bool,
) -> Counter:
    counters = Counter()
    for row in rows:
        decision = answer(row.get("ONAY_E_H"))
        corrected_label = answer(row.get("DUZELTILMIS_ETIKET"))
        corrected_value = (row.get("DUZELTILMIS_DEGER") or "").strip()
        if decision not in {"E", "H"} and not corrected_label and not corrected_value:
            counters["fact_unanswered"] += 1
            continue
        approved = bool(corrected_label or corrected_value or decision == "E")
        archive_key = (row.get("archive_key") or "").strip()
        old_label = answer(row.get("fact_type"))
        old_text = (row.get("fact_text") or "").strip()
        evidence = (row.get("evidence_text") or "").strip()
        if write:
            with connection.cursor() as cursor:
                if approved:
                    cursor.execute(
                        """
                        UPDATE historical_facts f
                        SET fact_type = %s,
                            fact_text = %s,
                            decision = 'accepted',
                            decision_reason = 'human_review',
                            review_status = 'approved',
                            confidence = 1.0,
                            updated_at = NOW()
                        FROM historical_documents d
                        WHERE f.historical_document_id = d.id
                          AND d.archive_key = %s
                          AND f.fact_type = %s
                          AND f.fact_text = %s
                          AND f.evidence_text = %s
                          AND f.review_status = 'pending'
                        """,
                        (
                            corrected_label or old_label,
                            corrected_value or old_text,
                            archive_key,
                            old_label,
                            old_text,
                            evidence,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE historical_facts f
                        SET decision = 'rejected',
                            decision_reason = 'human_reject',
                            review_status = 'rejected',
                            updated_at = NOW()
                        FROM historical_documents d
                        WHERE f.historical_document_id = d.id
                          AND d.archive_key = %s
                          AND f.fact_type = %s
                          AND f.fact_text = %s
                          AND f.evidence_text = %s
                          AND f.review_status = 'pending'
                        """,
                        (archive_key, old_label, old_text, evidence),
                    )
                counters["fact_rows_updated"] += cursor.rowcount
        counters["fact_approved" if approved else "fact_rejected"] += 1
    return counters


def main() -> None:
    load_dotenv()
    args = parse_args()
    classification_rows = read_csv(Path(args.classification_csv))
    fact_rows = read_csv(Path(args.fact_csv))
    with open_connection() as connection:
        if args.write:
            ensure_archive_schema(connection)
        summary = Counter()
        summary.update(
            resolve_classifications(
                connection,
                classification_rows,
                args.write,
            )
        )
        summary.update(resolve_facts(connection, fact_rows, args.write))
    result = {
        "mode": "write" if args.write else "dry_run",
        "classification_csv_rows": len(classification_rows),
        "fact_csv_rows": len(fact_rows),
        **dict(sorted(summary.items())),
    }
    print("Summary:", json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
