from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from fact_context_rules import campaign_amount_roles
from fact_surface_rules import validate_entity_surface


HEADER_PATTERN = re.compile(r"^\[(\d+)/(\d+)\].*?product=([^ ]+)")
FACT_PATTERN = re.compile(
    r"^  (ACCEPTED|REVIEW|REJECTED)\s+([A-Z_]+)\s+"
    r"([0-9.]+) \| (.*?) \| (.*)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a V3.1 historical NER dry-run report."
    )
    parser.add_argument(
        "report",
        nargs="?",
        default="archive_ner_v31_dry_run.txt",
    )
    return parser.parse_args()


def amount_key(value: str) -> str:
    folded = value.casefold().replace(".", "").replace(" ", "")
    return folded.replace(",", ".")


def main() -> None:
    args = parse_args()
    path = Path(args.report)
    lines = path.read_text(encoding="utf-8").splitlines()

    document_index = None
    decisions: Counter[str] = Counter()
    accepted_labels: Counter[str] = Counter()
    review_labels: Counter[str] = Counter()
    rejected_labels: Counter[str] = Counter()
    review_reasons: Counter[str] = Counter()
    accepted_amount_entries: dict[
        tuple[int, str], list[dict[str, str]]
    ] = defaultdict(list)
    pending_amount_entry = None
    incomplete_accepted_dates = []
    error_lines = []
    summary = None

    for line in lines:
        header = HEADER_PATTERN.match(line)
        if header:
            document_index = int(header.group(1))
        if "ERROR:" in line:
            error_lines.append(line)
        if line.startswith("Summary:"):
            summary = json.loads(line.split(":", 1)[1].strip())

        fact = FACT_PATTERN.match(line)
        if fact is None:
            if (
                pending_amount_entry is not None
                and line.startswith("    EVIDENCE |")
            ):
                pending_amount_entry["evidence"] = line.split("|", 1)[1].strip()
                pending_amount_entry = None
            continue
        if document_index is None:
            continue
        decision, label, _, value, reason = fact.groups()
        decisions[decision] += 1
        if decision == "ACCEPTED":
            accepted_labels[label] += 1
            if label in {"HARCAMA_ESIGI", "ODUL_TUTARI"}:
                entry = {
                    "label": label,
                    "value": value,
                    "evidence": "",
                }
                accepted_amount_entries[
                    (document_index, amount_key(value))
                ].append(entry)
                pending_amount_entry = entry
            if (
                label == "KAMPANYA_TARIH_ARALIGI"
                and validate_entity_surface(label, value) is not None
            ):
                incomplete_accepted_dates.append(
                    {"document": document_index, "value": value}
                )
        elif decision == "REVIEW":
            review_labels[label] += 1
            review_reasons[reason] += 1
        else:
            rejected_labels[label] += 1

    role_mismatches = []
    invalid_cross_role = []
    valid_dual_role = []
    for key, entries in accepted_amount_entries.items():
        labels = {entry["label"] for entry in entries}
        entry_mismatches = []
        for entry in entries:
            roles = campaign_amount_roles(entry["value"], entry["evidence"])
            expected = (
                "spend_threshold"
                if entry["label"] == "HARCAMA_ESIGI"
                else "reward_amount"
            )
            if expected not in roles:
                mismatch = {
                    "document": key[0],
                    "label": entry["label"],
                    "value": entry["value"],
                    "roles": sorted(roles),
                    "evidence": entry["evidence"],
                }
                role_mismatches.append(mismatch)
                entry_mismatches.append(mismatch)
        if len(labels) > 1:
            summary_item = {
                "document": key[0],
                "value": key[1],
                "roles": sorted(labels),
            }
            if entry_mismatches:
                summary_item["mismatches"] = entry_mismatches
                invalid_cross_role.append(summary_item)
            else:
                valid_dual_role.append(summary_item)

    output = {
        "report": str(path.resolve()),
        "summary": summary,
        "parsed_decisions": dict(sorted(decisions.items())),
        "accepted_labels": dict(sorted(accepted_labels.items())),
        "review_labels": dict(sorted(review_labels.items())),
        "rejected_labels": dict(sorted(rejected_labels.items())),
        "review_reasons": dict(sorted(review_reasons.items())),
        "accepted_cross_role_amount_count": len(invalid_cross_role),
        "accepted_cross_role_amount_samples": invalid_cross_role[:20],
        "accepted_valid_dual_role_amount_count": len(valid_dual_role),
        "accepted_valid_dual_role_amount_samples": valid_dual_role[:20],
        "accepted_amount_role_mismatch_count": len(role_mismatches),
        "accepted_amount_role_mismatch_samples": role_mismatches[:20],
        "accepted_incomplete_date_count": len(incomplete_accepted_dates),
        "accepted_incomplete_date_samples": incomplete_accepted_dates[:20],
        "error_count": len(error_lines),
        "error_samples": error_lines[:20],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
