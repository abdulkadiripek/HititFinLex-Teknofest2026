from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from archive_ingest_v28 import call_classifier, load_and_filter_documents


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_ZIP = "HititFinLex_VeriSeti_2.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose V2.8 historical classification review reasons. "
            "This command is read-only and never writes to the database."
        )
    )
    parser.add_argument("--dataset-zip", default=DEFAULT_ZIP)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--classification-threshold", type=float, default=0.80)
    parser.add_argument("--sample-per-reason", type=int, default=5)
    args = parser.parse_args()

    args.api_url = args.api_url.rstrip("/")
    args.force = False
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 0.0 <= args.classification_threshold <= 1.0:
        parser.error("--classification-threshold must be between 0 and 1")
    if args.sample_per_reason < 1 or args.sample_per_reason > 25:
        parser.error("--sample-per-reason must be between 1 and 25")
    return args


def compact(value: Any, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> None:
    load_dotenv()
    args = parse_args()
    documents, counters = load_and_filter_documents(args)

    print("Diagnostic: historical_v2_8_classification")
    print("Mode: READ_ONLY")
    print("Dataset ZIP:", Path(args.dataset_zip).resolve())
    print("Archive rows:", counters["archive_rows"])
    print("Eligible before window:", counters["eligible_before_window"])
    print("Selected documents:", len(documents))

    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    basis_counts: Counter[str] = Counter()
    review_product_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures = 0

    timeout = httpx.Timeout(180.0, connect=10.0)
    with httpx.Client(base_url=args.api_url, timeout=timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        if not health.json().get("classifier_ready"):
            raise RuntimeError("Classifier is not ready in the API.")

        for index, document in enumerate(documents, start=1):
            try:
                result = call_classifier(
                    client,
                    document,
                    args.classification_threshold,
                )
                product = result.get("product_type") or {}
                campaign = result.get("is_campaign") or {}
                rule = result.get("strong_rule") or {}
                product_label = str(product.get("label") or "")
                product_score = float(product.get("score") or 0.0)
                campaign_label = str(campaign.get("label") or "")
                campaign_score = float(campaign.get("score") or 0.0)
                decision = str(result.get("decision") or "UNKNOWN").upper()
                basis = str(result.get("decision_basis") or "unknown")
                reasons = [
                    str(value)
                    for value in (result.get("review_reasons") or [])
                ]

                decision_counts[decision] += 1
                basis_counts[basis] += 1
                if decision == "REVIEW":
                    review_product_counts[product_label] += 1
                    if not reasons:
                        reasons = ["missing_review_reason"]
                    for reason in reasons:
                        reason_counts[reason] += 1
                        if len(samples[reason]) < args.sample_per_reason:
                            samples[reason].append(
                                {
                                    "index": index,
                                    "key": document.archive_key,
                                    "date": str(document.snapshot_date or ""),
                                    "title": compact(document.page_title),
                                    "url": compact(document.source_url, 220),
                                    "source_category": compact(
                                        document.source_category
                                    ),
                                    "campaign_hint": document.is_campaign_hint,
                                    "campaign": campaign_label,
                                    "campaign_score": round(campaign_score, 4),
                                    "product": product_label,
                                    "product_score": round(product_score, 4),
                                    "rule_label": rule.get("label"),
                                    "rule_reason": rule.get("reason"),
                                    "basis": basis,
                                }
                            )
            except Exception as error:
                failures += 1
                print(
                    f"ERROR index={index} key={document.archive_key} "
                    f"type={type(error).__name__} message={error}"
                )

    print("Decision counts:", json.dumps(dict(decision_counts), sort_keys=True))
    print("Review reason counts:", json.dumps(dict(reason_counts), sort_keys=True))
    print("Review product counts:", json.dumps(dict(review_product_counts), sort_keys=True))
    print("Decision basis counts:", json.dumps(dict(basis_counts), sort_keys=True))
    print("Failures:", failures)

    for reason in sorted(samples):
        print(f"\nREASON: {reason} count={reason_counts[reason]}")
        for sample in samples[reason]:
            print("  SAMPLE:", json.dumps(sample, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
