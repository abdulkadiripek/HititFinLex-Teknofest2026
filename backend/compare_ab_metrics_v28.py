from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare production and V2.8 A/B test metrics."
    )
    parser.add_argument(
        "--product-baseline",
        default="models/classifier_product_v2_best/test_results.json",
    )
    parser.add_argument(
        "--product-candidate",
        default=(
            "ab_models_v28/product/models/"
            "classifier_product_v2_best/test_results.json"
        ),
    )
    parser.add_argument(
        "--ner-baseline",
        default="models/ner_v4_best/test_results.json",
    )
    parser.add_argument(
        "--ner-candidate",
        default="ab_models_v28/ner/ner_v28_best/test_results.json",
    )
    parser.add_argument("--max-regression", type=float, default=0.005)
    parser.add_argument("--output", default="ab_comparison_v28.json")
    return parser.parse_args()


def read_json(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Metric file was not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(payload: dict[str, Any], names: tuple[str, ...]) -> float:
    metrics = payload.get("test_metrics", {})
    for name in names:
        if name in metrics:
            return float(metrics[name])
    raise KeyError(f"None of the metric keys were found: {names}")


def comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    definitions: dict[str, tuple[str, ...]],
    max_regression: float,
) -> dict[str, Any]:
    rows = {}
    passed = True
    for label, names in definitions.items():
        old = metric(baseline, names)
        new = metric(candidate, names)
        delta = new - old
        metric_passed = delta >= -max_regression
        passed = passed and metric_passed
        rows[label] = {
            "baseline": round(old, 6),
            "candidate": round(new, 6),
            "delta": round(delta, 6),
            "passed": metric_passed,
        }
    return {"passed": passed, "metrics": rows}


def main() -> None:
    args = parse_args()
    if args.max_regression < 0:
        raise ValueError("--max-regression cannot be negative")
    product = comparison(
        read_json(args.product_baseline),
        read_json(args.product_candidate),
        {
            "macro_f1": ("test_macro_f1", "eval_macro_f1", "macro_f1"),
            "accuracy": ("test_accuracy", "eval_accuracy", "accuracy"),
        },
        args.max_regression,
    )
    ner = comparison(
        read_json(args.ner_baseline),
        read_json(args.ner_candidate),
        {
            "strict_f1": ("test_f1", "eval_f1", "f1"),
            "precision": ("test_precision", "eval_precision", "precision"),
            "recall": ("test_recall", "eval_recall", "recall"),
        },
        args.max_regression,
    )
    result = {
        "promotion_ready": bool(product["passed"] and ner["passed"]),
        "max_allowed_regression": args.max_regression,
        "product": product,
        "ner": ner,
        "note": (
            "Passing this numeric gate does not replace manual error analysis. "
            "Production model files are not modified by this script."
        ),
    }
    output_path = Path(args.output).expanduser().resolve()
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    print("Output:", output_path)


if __name__ == "__main__":
    main()
