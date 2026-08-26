from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


DEFAULT_API_URL = "http://127.0.0.1:8000"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preview or write new participation-finance documents through "
            "the classifier, NER, rule, embedding, and database pipeline."
        )
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Input .json or .jsonl file.",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Commit accepted documents and queue review records.",
    )
    parser.add_argument(
        "--allow-update",
        action="store_true",
        help="Replace an existing record_key when its content changed.",
    )
    args = parser.parse_args()
    if args.allow_update and not args.write:
        parser.error("--allow-update requires --write.")
    if not args.file.is_file():
        parser.error(f"Input file not found: {args.file}")
    args.api_url = args.api_url.rstrip("/")
    return args


def load_records(path: Path) -> list[dict]:
    if path.suffix.casefold() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL line {line_number} must contain an object."
                    )
                records.append(record)
        return records

    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError("JSON must contain one object or a list of objects.")


def main():
    args = parse_args()
    records = load_records(args.file)
    if not records:
        raise ValueError("Input file contains no records.")

    mode = "DATABASE_WRITE" if args.write else "DRY_RUN"
    print("Mode:", mode)
    print("Records:", len(records))
    summary: dict[str, int] = {}
    timeout = httpx.Timeout(300.0, connect=10.0)
    with httpx.Client(base_url=args.api_url, timeout=timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        if not health.json().get("classifier_ready"):
            raise RuntimeError("Classifier is not ready.")
        if not health.json().get("ner_model_ready"):
            raise RuntimeError("NER model is not ready.")
        if not health.json().get("model_ready"):
            raise RuntimeError("Embedding model is not ready.")

        for index, record in enumerate(records, start=1):
            payload = dict(record)
            payload["write"] = args.write
            payload["allow_update"] = args.allow_update
            response = client.post("/intake", json=payload)
            if not response.is_success:
                print(f"[{index}/{len(records)}] ERROR {response.status_code}")
                print(response.text)
                summary["ERROR"] = summary.get("ERROR", 0) + 1
                continue

            result = response.json()
            status = str(result["status"])
            action = str(result["database"]["action"])
            summary[status] = summary.get(status, 0) + 1
            print(
                f"[{index}/{len(records)}] {status:10} "
                f"action={action} record_key={result['record_key']}"
            )
            print(
                "  product="
                f"{result['classification']['product_type']['label']} "
                f"score={result['classification']['product_type']['score']:.4f}"
            )
            print(
                "  facts="
                f"accepted:{result['ner']['accepted_count']} "
                f"review:{result['ner']['review_count']} "
                f"rejected:{result['ner']['rejected_count']}"
            )
            if args.write:
                print(
                    "  database="
                    f"document:{result['database']['document_id']} "
                    f"chunks:{result['database']['chunks_written']} "
                    f"facts:{result['database']['facts_written']} "
                    f"fact_reviews:{result['database']['fact_reviews_queued']}"
                )

    print("Summary:", json.dumps(summary, ensure_ascii=True, sort_keys=True))
    if summary.get("ERROR"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
