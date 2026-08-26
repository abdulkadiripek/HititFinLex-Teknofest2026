from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from fact_surface_rules import validate_entity_surface


HEADER_PATTERN = re.compile(r"^\[(\d+)/(\d+)\].*?product=([^ ]+)")
FACT_PATTERN = re.compile(
    r"^  (ACCEPTED|REVIEW|REJECTED)\s+([A-Z_]+)\s+"
    r"([0-9.]+) \| (.*?) \| (.*)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a V3.0 historical NER dry-run report."
    )
    parser.add_argument(
        "report",
        nargs="?",
        default="archive_ner_v30_dry_run.txt",
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
    accepted_amount_roles: dict[tuple[int, str], set[str]] = defaultdict(set)
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
        if fact is None or document_index is None:
            continue
        decision, label, _, value, reason = fact.groups()
        decisions[decision] += 1
        if decision == "ACCEPTED":
            accepted_labels[label] += 1
            if label in {"HARCAMA_ESIGI", "ODUL_TUTARI"}:
                accepted_amount_roles[
                    (document_index, amount_key(value))
                ].add(label)
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

    cross_role = [
        {"document": key[0], "value": key[1], "roles": sorted(roles)}
        for key, roles in accepted_amount_roles.items()
        if len(roles) > 1
    ]

    output = {
        "report": str(path.resolve()),
        "summary": summary,
        "parsed_decisions": dict(sorted(decisions.items())),
        "accepted_labels": dict(sorted(accepted_labels.items())),
        "review_labels": dict(sorted(review_labels.items())),
        "rejected_labels": dict(sorted(rejected_labels.items())),
        "review_reasons": dict(sorted(review_reasons.items())),
        "accepted_cross_role_amount_count": len(cross_role),
        "accepted_cross_role_amount_samples": cross_role[:20],
        "accepted_incomplete_date_count": len(incomplete_accepted_dates),
        "accepted_incomplete_date_samples": incomplete_accepted_dates[:20],
        "error_count": len(error_lines),
        "error_samples": error_lines[:20],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
