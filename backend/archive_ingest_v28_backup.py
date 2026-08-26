from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from archive_common_v28 import (
    PIPELINE_VERSION,
    ArchiveDocument,
    ensure_archive_schema,
    iter_archive_documents,
    load_live_fingerprints,
    load_processed_archive_keys,
    open_connection,
    upsert_historical_result,
)
from archive_quality_v28 import classification_quality, deduplicate_facts
from coverage_rules_v27 import extract_coverage_facts
from ingest_ner_facts import (
    API_PREDICTION_THRESHOLD,
    decide_candidate,
    deduplicate_candidates,
    evidence_window,
    split_text,
)
from intake_service import ALLOWED_ENTITY_LABELS_BY_PRODUCT


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_ZIP = "HititFinLex_VeriSeti_2.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify and extract facts from historical HititFinLex snapshots. "
            "Dry-run is the default; --write is required for database changes."
        )
    )
    parser.add_argument("--dataset-zip", default=DEFAULT_ZIP)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of archive records to process. Use 0 for all records.",
    )
    parser.add_argument("--classification-threshold", type=float, default=0.80)
    parser.add_argument("--min-product-confidence", type=float, default=0.80)
    parser.add_argument("--review-threshold", type=float, default=0.60)
    parser.add_argument("--min-rule-confidence", type=float, default=0.90)
    parser.add_argument("--chunk-chars", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--skip-ner", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--show-evidence", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    args.api_url = args.api_url.rstrip("/")
    if args.offset < 0:
        parser.error("--offset cannot be negative")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    for name in (
        "classification_threshold",
        "min_product_confidence",
        "review_threshold",
        "min_rule_confidence",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.chunk_chars < 400 or args.chunk_chars > 8000:
        parser.error("--chunk-chars must be between 400 and 8000")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_chars:
        parser.error("--chunk-overlap must be smaller than --chunk-chars")
    return args


def call_classifier(
    client: httpx.Client,
    document: ArchiveDocument,
    threshold: float,
) -> dict[str, Any]:
    classification_text = "\n".join(
        part for part in (document.page_title, document.raw_text) if part
    )[:10000]
    response = client.post(
        "/classify",
        json={
            "text": classification_text,
            "page_title": document.page_title,
            "source_url": document.source_url,
            "threshold": threshold,
        },
    )
    response.raise_for_status()
    return response.json()


def call_ner(client: httpx.Client, text: str) -> list[dict[str, Any]]:
    response = client.post(
        "/ner",
        json={"text": text, "threshold": API_PREDICTION_THRESHOLD},
    )
    response.raise_for_status()
    return list(response.json().get("entities", []))


def ner_facts(
    client: httpx.Client,
    document: ArchiveDocument,
    product_code: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    allowed_labels = ALLOWED_ENTITY_LABELS_BY_PRODUCT.get(product_code)
    if not allowed_labels or args.skip_ner:
        return [], 0

    candidates = []
    chunks = split_text(document.raw_text, args.chunk_chars, args.chunk_overlap)
    filtered_out = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        for entity in call_ner(client, chunk):
            label = str(entity["label"])
            if label not in allowed_labels:
                filtered_out += 1
                continue
            evidence = evidence_window(
                chunk,
                int(entity["start"]),
                int(entity["end"]),
            )
            candidates.append(
                decide_candidate(
                    label=label,
                    value=str(entity["text"]),
                    evidence=evidence,
                    confidence=float(entity["score"]),
                    source_chunk=chunk_index,
                    review_threshold=args.review_threshold,
                )
            )

    output = []
    for candidate in deduplicate_candidates(candidates):
        output.append(
            {
                "fact_type": candidate.fact_type,
                "fact_text": candidate.fact_text,
                "normalized_value": candidate.normalized_value,
                "evidence_text": candidate.evidence_text,
                "confidence": candidate.confidence,
                "source_chunk": candidate.source_chunk,
                "decision": candidate.decision,
                "decision_reason": candidate.reason,
                "extraction_method": PIPELINE_VERSION + "_ner",
            }
        )
    return output, filtered_out


def rule_facts(
    document: ArchiveDocument,
    product_code: str,
    min_confidence: float,
) -> list[dict[str, Any]]:
    source_text = "\n".join(
        part for part in (document.page_title, document.raw_text) if part
    )
    output = []
    for fact in extract_coverage_facts(
        source_text,
        product_code,
        page_title=document.page_title,
        source_url=document.source_url,
    ):
        if fact.confidence < min_confidence:
            continue
        output.append(
            {
                "fact_type": fact.fact_type,
                "fact_text": fact.fact_text,
                "normalized_value": fact.normalized_value,
                "evidence_text": fact.evidence_text,
                "confidence": fact.confidence,
                "source_chunk": 0,
                "decision": "accepted",
                "decision_reason": fact.rule_name,
                "extraction_method": PIPELINE_VERSION + "_rule",
            }
        )
    return output


def load_and_filter_documents(
    args: argparse.Namespace,
) -> tuple[list[ArchiveDocument], Counter]:
    counters = Counter()
    with open_connection() as connection:
        live_keys, live_urls, live_hashes = load_live_fingerprints(connection)
        processed = load_processed_archive_keys(connection)

    selected = []
    seen_hashes = set()
    for document in iter_archive_documents(Path(args.dataset_zip)):
        counters["archive_rows"] += 1
        if document.content_hash in seen_hashes:
            counters["duplicate_inside_archive"] += 1
            continue
        seen_hashes.add(document.content_hash)
        if document.content_hash in live_hashes:
            counters["duplicate_live_text"] += 1
            continue
        if document.archive_key in live_keys:
            counters["duplicate_live_key"] += 1
            continue
        if document.canonical_url in live_urls:
            counters["historical_versions_of_live_urls"] += 1

        state = processed.get(document.archive_key)
        if (
            state is not None
            and state[0] == document.content_hash
            and state[1] == PIPELINE_VERSION
            and not args.force
        ):
            counters["unchanged_skipped"] += 1
            continue
        selected.append(document)

    counters["eligible_before_window"] = len(selected)
    stop = None if args.limit == 0 else args.offset + args.limit
    selected = selected[args.offset:stop]
    counters["selected"] = len(selected)
    return selected, counters


def print_candidate(fact: dict[str, Any], show_evidence: bool) -> None:
    print(
        f"  {str(fact['decision']).upper():8} "
        f"{str(fact['fact_type']):28} "
        f"{float(fact['confidence']):.4f} | {fact['fact_text']} "
        f"| {fact['decision_reason']}"
    )
    if show_evidence:
        print("    EVIDENCE |", fact["evidence_text"])


def main() -> None:
    load_dotenv()
    args = parse_args()
    documents, counters = load_and_filter_documents(args)

    print("Pipeline:", PIPELINE_VERSION)
    print("Mode:", "DATABASE_WRITE" if args.write else "DRY_RUN")
    print("Dataset ZIP:", Path(args.dataset_zip).resolve())
    print("Archive rows:", counters["archive_rows"])
    print("Duplicate live text:", counters["duplicate_live_text"])
    print("Historical versions of live URLs:", counters["historical_versions_of_live_urls"])
    print("Unchanged records skipped:", counters["unchanged_skipped"])
    print("Selected documents:", len(documents))
    if not documents:
        print("Summary:", json.dumps(dict(counters), sort_keys=True))
        return

    if args.write:
        with open_connection() as connection:
            ensure_archive_schema(connection)

    timeout = httpx.Timeout(180.0, connect=10.0)
    with httpx.Client(base_url=args.api_url, timeout=timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        health_payload = health.json()
        if not health_payload.get("classifier_ready"):
            raise RuntimeError("Classifier is not ready in the API.")
        if not args.classification_only and not args.skip_ner:
            if not health_payload.get("ner_model_ready"):
                raise RuntimeError("NER model is not ready in the API.")

        for index, document in enumerate(documents, start=1):
            try:
                classification = call_classifier(
                    client,
                    document,
                    args.classification_threshold,
                )
                quality_status = classification_quality(
                    classification,
                    args.min_product_confidence,
                )
                product_code = str(
                    classification.get("product_type", {}).get("label") or ""
                )
                product_score = float(
                    classification.get("product_type", {}).get("score") or 0.0
                )
                print(
                    f"[{index}/{len(documents)}] key={document.archive_key} "
                    f"date={document.snapshot_date or '-'} "
                    f"bank={document.bank_name} product={product_code} "
                    f"score={product_score:.4f} quality={quality_status}"
                )

                facts: list[dict[str, Any]] = []
                if quality_status == "accepted" and not args.classification_only:
                    extracted, filtered_out = ner_facts(
                        client,
                        document,
                        product_code,
                        args,
                    )
                    counters["ner_filtered_out"] += filtered_out
                    facts.extend(extracted)
                    facts.extend(
                        rule_facts(
                            document,
                            product_code,
                            args.min_rule_confidence,
                        )
                    )
                    facts = deduplicate_facts(facts)

                fact_counts = Counter(str(fact["decision"]) for fact in facts)
                counters["classification_" + quality_status] += 1
                counters["facts_accepted"] += fact_counts["accepted"]
                counters["facts_review"] += fact_counts["review"]
                counters["facts_rejected"] += fact_counts["rejected"]
                counters["documents_with_accepted_facts"] += int(
                    fact_counts["accepted"] > 0
                )
                if facts:
                    print(
                        "  FACTS "
                        f"accepted={fact_counts['accepted']} "
                        f"review={fact_counts['review']} "
                        f"rejected={fact_counts['rejected']}"
                    )
                    for fact in facts:
                        print_candidate(fact, args.show_evidence)

                if args.write:
                    with open_connection() as connection:
                        upsert_historical_result(
                            connection,
                            document,
                            classification,
                            quality_status,
                            facts,
                        )
                    counters["documents_written"] += 1
            except Exception as error:
                counters["failed"] += 1
                print(
                    f"[{index}/{len(documents)}] key={document.archive_key} "
                    f"ERROR: {type(error).__name__}: {error}"
                )

    summary = dict(sorted(counters.items()))
    summary["mode"] = "write" if args.write else "dry_run"
    print("Summary:", json.dumps(summary, ensure_ascii=True, sort_keys=True))
    print("Completed at:", datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
