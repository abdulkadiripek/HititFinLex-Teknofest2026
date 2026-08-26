from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


TOKEN_PATTERN = re.compile(r"\w+(?:[.,/]\w+)*|[^\w\s]", flags=re.UNICODE)


def choose_versions(
    rows: list[dict[str, Any]],
    max_versions: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["canonical_group_key"]).strip()].append(row)
    chosen = []
    for group_rows in groups.values():
        if len(group_rows) <= max_versions:
            chosen.extend(group_rows)
            continue
        if max_versions == 1:
            chosen.append(group_rows[-1])
            continue
        indices = {
            round(index * (len(group_rows) - 1) / (max_versions - 1))
            for index in range(max_versions)
        }
        chosen.extend(group_rows[index] for index in sorted(indices))
    return sorted(chosen, key=lambda row: int(row["id"]))


def annotate_evidence(
    record_id: str,
    document_id: str,
    evidence: str,
    facts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    spans = []
    occupied: list[tuple[int, int]] = []
    for fact in sorted(facts, key=lambda item: -len(item["text"])):
        start = evidence.find(fact["text"])
        if start < 0:
            continue
        end = start + len(fact["text"])
        if any(start < old_end and old_start < end for old_start, old_end in occupied):
            continue
        occupied.append((start, end))
        spans.append((start, end, fact["label"]))
    if not spans:
        return None

    matches = list(TOKEN_PATTERN.finditer(evidence))
    tags = ["O"] * len(matches)
    for start, end, label in sorted(spans):
        covered = [
            index
            for index, match in enumerate(matches)
            if match.start() < end and start < match.end()
        ]
        for offset, token_index in enumerate(covered):
            if tags[token_index] != "O":
                continue
            tags[token_index] = ("B-" if offset == 0 else "I-") + label
    if all(tag == "O" for tag in tags):
        return None
    return {
        "id": record_id,
        "document_id": document_id,
        "tokens": [match.group(0) for match in matches],
        "ner_tags": tags,
        "text": evidence,
        "augmentation": "historical_v2_8_silver",
    }
