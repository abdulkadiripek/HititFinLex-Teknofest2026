from fact_surface_rules import validate_entity_surface


CASES = [
    ("TAKSIT_SAYISI", "20.000 TL", "count_entity_has_invalid_unit"),
    ("TAKSIT_SAYISI", "12", None),
    ("INDIRIM_TUTARI", "20 puan", "money_entity_missing_currency"),
    ("INDIRIM_TUTARI", "20 TL", None),
    ("INDIRIM_ORANI", "%10", None),
    ("INDIRIM_ORANI", "10", "percent_entity_missing_marker"),
    ("VADE_SURESI", "45 gun", None),
    ("VADE_SURESI", "36", "duration_entity_missing_unit"),
    ("FINANSMAN_TUTARI", "7.500 TL", None),
]


def main() -> None:
    failures = []
    for label, value, expected in CASES:
        actual = validate_entity_surface(label, value)
        if actual != expected:
            failures.append(
                {
                    "label": label,
                    "value": value,
                    "expected": expected,
                    "actual": actual,
                }
            )

    if failures:
        raise RuntimeError(f"Surface rule failures: {failures}")

    print(f"Fact surface rules: OK ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
