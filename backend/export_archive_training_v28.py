from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from archive_common_v28 import canonicalize_url, open_connection, sha256_text
from archive_training_utils_v28 import (
    TOKEN_PATTERN,
    annotate_evidence,
    choose_versions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export leakage-controlled silver training rows from V2.8. "
            "Archive labels are exported to train only, never to validation/test."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="data/archive_v28_training",
    )
    parser.add_argument("--min-classification-confidence", type=float, default=0.90)
    parser.add_argument("--min-fact-confidence", type=float, default=0.90)
    parser.add_argument("--max-versions-per-url", type=int, default=2)
    parser.add_argument("--negative-ratio", type=float, default=0.50)
    parser.add_argument(
        "--protected-split-dir",
        default="data_extracted/data/curated_v3/training",
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_classification_confidence <= 1.0:
        parser.error("invalid classification confidence")
    if not 0.0 <= args.min_fact_confidence <= 1.0:
        parser.error("invalid fact confidence")
    if args.max_versions_per_url < 1:
        parser.error("--max-versions-per-url must be positive")
    if not 0.0 <= args.negative_ratio <= 2.0:
        parser.error("--negative-ratio must be between 0 and 2")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def protected_group_keys(split_dir: Path) -> tuple[set[str], str]:
    protected_ids = set()
    for filename in ("classification_val.jsonl", "classification_test.jsonl"):
        for row in read_jsonl(split_dir / filename):
            value = row.get("source_document_id", row.get("id"))
            if value:
                protected_ids.add(str(value).split("::")[0])

    with open_connection() as connection:
        with connection.cursor() as cursor:
            if protected_ids:
                cursor.execute(
                    """
                    SELECT source_url
                    FROM documents
                    WHERE record_key = ANY(%s)
                    """,
                    (sorted(protected_ids),),
                )
                mode = "current_validation_and_test_urls"
            else:
                cursor.execute("SELECT source_url FROM documents")
                mode = "all_current_urls_fallback"
            urls = [str(row[0]) for row in cursor.fetchall() if row[0]]
    return {sha256_text(canonicalize_url(url)) for url in urls}, mode


def load_classification_candidates(min_confidence: float) -> list[dict[str, Any]]:
    with open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.id,
                    d.archive_key,
                    d.canonical_group_key,
                    b.bank_name,
                    d.page_title,
                    d.raw_text,
                    d.source_url,
                    d.snapshot_date,
                    d.campaign_label,
                    d.product_type_code,
                    d.classification_confidence
                FROM historical_documents d
                JOIN banks b ON b.id = d.bank_id
                WHERE d.quality_status = 'accepted'
                  AND d.searchable IS TRUE
                  AND d.classification_confidence >= %s
                  AND d.product_type_code IS NOT NULL
                ORDER BY d.canonical_group_key, d.snapshot_date NULLS LAST, d.id
                """,
                (min_confidence,),
            )
            columns = [item.name for item in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def classification_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        title = str(row["page_title"] or "").strip()
        text = "\n".join(part for part in (title, str(row["raw_text"])) if part)
        output.append(
            {
                "id": "history::" + str(row["archive_key"]),
                "source_document_id": "history::" + str(row["archive_key"]),
                "canonical_group_key": str(row["canonical_group_key"]).strip(),
                "text": text,
                "is_campaign": str(row["campaign_label"] or "HAYIR"),
                "product_type": str(row["product_type_code"]),
                "bank_name": str(row["bank_name"]),
                "source_url": str(row["source_url"]),
                "snapshot_date": (
                    row["snapshot_date"].isoformat()
                    if row["snapshot_date"]
                    else None
                ),
                "confidence": float(row["classification_confidence"]),
                "label_source": "historical_v2_8_silver_train_only",
            }
        )
    return output


def load_fact_groups(
    document_ids: list[int],
    min_confidence: float,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    if not document_ids:
        return {}
    with open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    f.historical_document_id,
                    d.archive_key,
                    f.fact_type,
                    f.fact_text,
                    f.evidence_text,
                    f.confidence
                FROM historical_facts f
                JOIN historical_documents d
                    ON d.id = f.historical_document_id
                WHERE f.historical_document_id = ANY(%s)
                  AND f.decision = 'accepted'
                  AND f.confidence >= %s
                  AND f.extraction_method LIKE %s
                ORDER BY f.historical_document_id, f.evidence_text, f.id
                """,
                (document_ids, min_confidence, "%_ner"),
            )
            rows = cursor.fetchall()
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for document_id, archive_key, label, text, evidence, confidence in rows:
        groups[(int(document_id), str(evidence))].append(
            {
                "archive_key": str(archive_key),
                "label": str(label),
                "text": str(text),
                "confidence": float(confidence),
            }
        )
    return groups


def positive_ner_rows(
    selected_rows: list[dict[str, Any]],
    min_confidence: float,
) -> list[dict[str, Any]]:
    groups = load_fact_groups(
        [int(row["id"]) for row in selected_rows],
        min_confidence,
    )
    output = []
    for index, ((document_id, evidence), facts) in enumerate(
        sorted(groups.items()),
        start=1,
    ):
        archive_key = str(facts[0]["archive_key"])
        row = annotate_evidence(
            f"history-{archive_key}-fact-{index:05d}",
            "history::" + archive_key,
            evidence,
            facts,
        )
        if row is not None:
            output.append(row)
    return output


def negative_ner_rows(
    selected_rows: list[dict[str, Any]],
    positive_rows: list[dict[str, Any]],
    ratio: float,
) -> list[dict[str, Any]]:
    target = int(round(len(positive_rows) * ratio))
    if target == 0:
        return []
    positive_texts = {str(row["text"]).strip() for row in positive_rows}
    candidates = []
    for row in selected_rows:
        archive_key = str(row["archive_key"])
        raw_text = str(row["raw_text"])
        for segment in re.split(r"(?<=[.!?])\s+|\n+", raw_text):
            segment = " ".join(segment.split())
            if not 40 <= len(segment) <= 420:
                continue
            if not re.search(r"\d", segment):
                continue
            if segment in positive_texts:
                continue
            digest = hashlib.sha256(
                f"{archive_key}|{segment}".encode("utf-8")
            ).hexdigest()
            matches = list(TOKEN_PATTERN.finditer(segment))
            candidates.append(
                (
                    digest,
                    {
                        "id": f"history-{archive_key}-negative-{digest[:12]}",
                        "document_id": "history::" + archive_key,
                        "tokens": [match.group(0) for match in matches],
                        "ner_tags": ["O"] * len(matches),
                        "text": segment,
                        "augmentation": "historical_v2_8_hard_negative",
                    },
                )
            )
    candidates.sort(key=lambda item: item[0])
    return [row for _, row in candidates[:target]]


def export_review_csv(output_path: Path) -> int:
    with open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.archive_key,
                    b.bank_name,
                    d.snapshot_date,
                    d.page_title,
                    d.source_url,
                    d.product_type_code,
                    d.classification_confidence,
                    d.classification_basis,
                    d.classification_payload
                FROM historical_documents d
                JOIN banks b ON b.id = d.bank_id
                WHERE d.quality_status = 'review'
                ORDER BY d.classification_confidence DESC NULLS LAST, d.id
                """
            )
            rows = cursor.fetchall()
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "archive_key",
                "bank_name",
                "snapshot_date",
                "page_title",
                "source_url",
                "suggested_product_type",
                "confidence",
                "decision_basis",
                "review_reasons",
                "ONAY_E_H",
                "DUZELTILMIS_ETIKET",
            ]
        )
        for row in rows:
            payload = row[8] or {}
            writer.writerow(
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    ",".join(payload.get("review_reasons", [])),
                    "",
                    "",
                ]
            )
    return len(rows)


def export_fact_review_csv(output_path: Path) -> int:
    with open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    d.archive_key,
                    b.bank_name,
                    d.snapshot_date,
                    d.page_title,
                    d.source_url,
                    d.product_type_code,
                    f.fact_type,
                    f.fact_text,
                    f.evidence_text,
                    f.confidence,
                    f.decision_reason
                FROM historical_facts f
                JOIN historical_documents d
                    ON d.id = f.historical_document_id
                JOIN banks b ON b.id = d.bank_id
                WHERE f.decision = 'review'
                  AND f.review_status = 'pending'
                ORDER BY f.confidence DESC, f.id
                """
            )
            rows = cursor.fetchall()
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "archive_key",
                "bank_name",
                "snapshot_date",
                "page_title",
                "source_url",
                "product_type",
                "fact_type",
                "fact_text",
                "evidence_text",
                "confidence",
                "review_reason",
                "ONAY_E_H",
                "DUZELTILMIS_ETIKET",
                "DUZELTILMIS_DEGER",
            ]
        )
        for row in rows:
            writer.writerow([*row, "", "", ""])
    return len(rows)


def main() -> None:
    load_dotenv()
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protected_groups, protection_mode = protected_group_keys(
        Path(args.protected_split_dir)
    )
    candidates = load_classification_candidates(
        args.min_classification_confidence
    )
    leakage_excluded = [
        row
        for row in candidates
        if str(row["canonical_group_key"]).strip() in protected_groups
    ]
    safe_candidates = [
        row
        for row in candidates
        if str(row["canonical_group_key"]).strip() not in protected_groups
    ]
    selected = choose_versions(safe_candidates, args.max_versions_per_url)

    class_rows = classification_rows(selected)
    positive_rows = positive_ner_rows(selected, args.min_fact_confidence)
    negative_rows = negative_ner_rows(
        selected,
        positive_rows,
        args.negative_ratio,
    )
    ner_rows = positive_rows + negative_rows

    write_jsonl(output_dir / "classification_archive_train.jsonl", class_rows)
    write_jsonl(output_dir / "ner_archive_train_bio.jsonl", ner_rows)
    review_count = export_review_csv(
        output_dir / "archive_classification_review.csv"
    )
    fact_review_count = export_fact_review_csv(
        output_dir / "archive_fact_review.csv"
    )

    summary = {
        "label_policy": "silver_train_only",
        "protection_mode": protection_mode,
        "protected_canonical_groups": len(protected_groups),
        "classification_candidates": len(candidates),
        "leakage_excluded": len(leakage_excluded),
        "version_limited_excluded": len(safe_candidates) - len(selected),
        "classification_train_rows": len(class_rows),
        "classification_distribution": dict(
            Counter(row["product_type"] for row in class_rows).most_common()
        ),
        "ner_positive_rows": len(positive_rows),
        "ner_negative_rows": len(negative_rows),
        "ner_train_rows": len(ner_rows),
        "classification_review_rows": review_count,
        "fact_review_rows": fact_review_count,
        "minimum_classification_confidence": args.min_classification_confidence,
        "minimum_fact_confidence": args.min_fact_confidence,
        "max_versions_per_url": args.max_versions_per_url,
    }
    with (output_dir / "archive_training_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("Output directory:", output_dir)
    print("Summary:", json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
