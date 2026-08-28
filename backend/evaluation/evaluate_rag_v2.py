"""Command line entry point for paired legacy and RAG V2 evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.rag_v2_metrics import (  # noqa: E402
    DatasetValidationError,
    compare_records,
    load_comparison,
    load_scenarios,
    merge_dataset_and_comparison,
)


DEFAULT_DATASET = Path(__file__).with_name("multiturn_scenarios.silver_unverified.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy and RAG V2 outputs on one shared silver_unverified dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--comparison",
        type=Path,
        help="JSON or JSONL with paired legacy/v2 results. If omitted, metrics are unavailable.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        scenarios = load_scenarios(args.dataset)
        comparison = load_comparison(args.comparison) if args.comparison else None
        records = merge_dataset_and_comparison(scenarios, comparison)
        report = compare_records(records)
    except (OSError, json.JSONDecodeError, DatasetValidationError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
