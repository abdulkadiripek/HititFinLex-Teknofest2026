from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import httpx
from dotenv import load_dotenv

from archive_common_v28 import (
    ArchiveDocument,
    ensure_archive_schema,
    open_connection,
    upsert_historical_result,
)
from archive_ingest_v28 import ner_facts, rule_facts
from archive_quality_v28 import deduplicate_facts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill facts for accepted historical documents, including "
            "human-approved classification reviews. Dry-run is the default."
        )
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--include-covered", action="store_true")
    parser.add_argument("--skip-ner", action="store_true")
    parser.add_argument("--review-threshold", type=float, default=0.60)
    parser.add_argument("--min-rule-confidence", type=float, default=0.90)
    parser.add_argument("--chunk-chars", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--show-evidence", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    args.api_url = args.api_url.rstrip("/")
    if args.limit < 1:
        parser.error("--limit must be positive")
    return args


def load_documents(args: argparse.Namespace) -> list[tuple[ArchiveDocument, dict]]:
    fact_filter = "" if args.include_covered else "AND COALESCE(f.fact_count, 0) = 0"
    with open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                WITH fact_counts AS (
                    SELECT historical_document_id, COUNT(*) AS fact_count
                    FROM historical_facts
                    WHERE decision = 'accepted'
                    GROUP BY historical_document_id
                )
                SELECT
                    d.archive_key,
                    b.bank_key,
                    b.bank_name,
                    d.source_url,
                    d.canonical_url,
                    d.canonical_group_key,
                    COALESCE(d.archive_url, ''),
                    COALESCE(d.page_title, ''),
                    d.raw_text,
                    d.content_hash,
                    d.snapshot_date,
                    d.collected_at,
                    COALESCE(d.source_category, ''),
                    d.is_campaign_hint,
                    d.product_type_code,
                    d.classification_confidence,
                    d.campaign_label,
                    d.campaign_confidence,
                    d.classification_basis,
                    d.classification_payload
                FROM historical_documents d
                JOIN banks b ON b.id = d.bank_id
                LEFT JOIN fact_counts f ON f.historical_document_id = d.id
                WHERE d.quality_status = 'accepted'
                  AND d.searchable IS TRUE
                  AND d.product_type_code IS NOT NULL
                  {fact_filter}
                ORDER BY d.verified DESC, d.id
                LIMIT %s
                """,
                (args.limit,),
            )
            rows = cursor.fetchall()

    output = []
    for row in rows:
        document = ArchiveDocument(
            archive_key=str(row[0]),
            bank_key=str(row[1]),
            bank_name=str(row[2]),
            source_url=str(row[3]),
            canonical_url=str(row[4]),
            canonical_group_key=str(row[5]).strip(),
            archive_url=str(row[6]),
            page_title=str(row[7]),
            raw_text=str(row[8]),
            content_hash=str(row[9]).strip(),
            snapshot_date=row[10],
            collected_at=row[11],
            source_category=str(row[12]),
            is_campaign_hint=bool(row[13]),
        )
        payload: dict[str, Any] = dict(row[19] or {})
        payload["decision"] = "ACCEPTED"
        payload["decision_basis"] = str(row[18] or "historical_fact_backfill")
        payload["product_type"] = {
            "label": str(row[14]),
            "score": float(row[15] or 1.0),
        }
        payload["is_campaign"] = {
            "label": str(row[16] or "HAYIR"),
            "score": float(row[17] or 1.0),
        }
        payload.setdefault("strong_rule", {"label": None, "reason": None})
        output.append((document, payload))
    return output


def main() -> None:
    load_dotenv()
    args = parse_args()
    documents = load_documents(args)
    print("Mode:", "DATABASE_WRITE" if args.write else "DRY_RUN")
    print("Selected documents:", len(documents))
    counters = Counter()
    timeout = httpx.Timeout(180.0, connect=10.0)
    with httpx.Client(base_url=args.api_url, timeout=timeout) as client:
        if not args.skip_ner:
            health = client.get("/health")
            health.raise_for_status()
            if not health.json().get("ner_model_ready"):
                raise RuntimeError("NER model is not ready in the API.")
        for index, (document, classification) in enumerate(documents, start=1):
            product_code = str(classification["product_type"]["label"])
            try:
                facts, filtered = ner_facts(
                    client,
                    document,
                    product_code,
                    args,
                )
                facts.extend(
                    rule_facts(document, product_code, args.min_rule_confidence)
                )
                facts = deduplicate_facts(facts)
                decisions = Counter(str(fact["decision"]) for fact in facts)
                counters["filtered_out"] += filtered
                for key, value in decisions.items():
                    counters["facts_" + key] += value
                counters["processed"] += 1
                print(
                    f"[{index}/{len(documents)}] key={document.archive_key} "
                    f"product={product_code} accepted={decisions['accepted']} "
                    f"review={decisions['review']} rejected={decisions['rejected']}"
                )
                if args.show_evidence:
                    for fact in facts:
                        print(
                            f"  {fact['decision'].upper():8} {fact['fact_type']} "
                            f"| {fact['fact_text']} | {fact['evidence_text']}"
                        )
                if args.write:
                    with open_connection() as connection:
                        ensure_archive_schema(connection)
                        upsert_historical_result(
                            connection,
                            document,
                            classification,
                            "accepted",
                            facts,
                        )
                    counters["written"] += 1
            except Exception as error:
                counters["failed"] += 1
                print(
                    f"[{index}/{len(documents)}] key={document.archive_key} "
                    f"ERROR: {type(error).__name__}: {error}"
                )
    print("Summary:", json.dumps(dict(counters), sort_keys=True))


if __name__ == "__main__":
    main()
