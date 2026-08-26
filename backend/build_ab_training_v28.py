from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build A/B training inputs by adding archive silver rows to train only. "
            "Existing validation and test files are copied without changes."
        )
    )
    parser.add_argument(
        "--archive-dir",
        default="data/archive_v28_training",
    )
    parser.add_argument(
        "--classification-dir",
        default="output/classifier_v2_data",
    )
    parser.add_argument(
        "--ner-dir",
        default="output/ner_v4",
    )
    parser.add_argument(
        "--output-dir",
        default="data/ab_v28",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file was not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def deduplicate_rows(
    base_rows: list[dict[str, Any]],
    archive_rows: list[dict[str, Any]],
    text_field: str,
) -> tuple[list[dict[str, Any]], int]:
    seen = {normalize(str(row[text_field])) for row in base_rows}
    accepted = list(base_rows)
    duplicate_count = 0
    for row in archive_rows:
        signature = normalize(str(row[text_field]))
        if not signature or signature in seen:
            duplicate_count += 1
            continue
        seen.add(signature)
        accepted.append(row)
    return accepted, duplicate_count


def main() -> None:
    args = parse_args()
    archive_dir = Path(args.archive_dir).resolve()
    classification_dir = Path(args.classification_dir).resolve()
    ner_dir = Path(args.ner_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    class_output = output_dir / "classification_v28"
    ner_output = output_dir / "ner_v28"
    class_output.mkdir(parents=True, exist_ok=True)
    ner_output.mkdir(parents=True, exist_ok=True)

    current_class_train = read_jsonl(
        classification_dir / "classification_train_augmented.jsonl"
    )
    archive_class_train = read_jsonl(
        archive_dir / "classification_archive_train.jsonl"
    )
    combined_class_train, class_duplicates = deduplicate_rows(
        current_class_train,
        archive_class_train,
        "text",
    )
    write_jsonl(
        class_output / "classification_train_augmented.jsonl",
        combined_class_train,
    )
    for source_name, target_name in (
        ("classification_validation.jsonl", "classification_validation.jsonl"),
        ("classification_test.jsonl", "classification_test.jsonl"),
    ):
        shutil.copy2(classification_dir / source_name, class_output / target_name)

    current_ner_train = read_jsonl(ner_dir / "ner_train_bio.jsonl")
    archive_ner_train = read_jsonl(archive_dir / "ner_archive_train_bio.jsonl")
    combined_ner_train, ner_duplicates = deduplicate_rows(
        current_ner_train,
        archive_ner_train,
        "text",
    )
    write_jsonl(ner_output / "ner_train_bio.jsonl", combined_ner_train)
    for filename in ("ner_val_bio.jsonl", "ner_test_bio.jsonl"):
        shutil.copy2(ner_dir / filename, ner_output / filename)

    summary = {
        "policy": "archive_rows_added_to_train_only",
        "classification": {
            "base_train": len(current_class_train),
            "archive_candidates": len(archive_class_train),
            "archive_exact_duplicates_removed": class_duplicates,
            "combined_train": len(combined_class_train),
            "archive_distribution": dict(
                Counter(
                    str(row["product_type"])
                    for row in archive_class_train
                ).most_common()
            ),
        },
        "ner": {
            "base_train": len(current_ner_train),
            "archive_candidates": len(archive_ner_train),
            "archive_exact_duplicates_removed": ner_duplicates,
            "combined_train": len(combined_ner_train),
        },
        "validation_and_test": "copied_unchanged",
    }
    with (output_dir / "ab_training_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print("Output directory:", output_dir)
    print("Summary:", json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
