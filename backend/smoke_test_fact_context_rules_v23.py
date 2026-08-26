from __future__ import annotations

from fact_context_rules import (
    AUTO_THRESHOLDS,
    campaign_date_context_pass,
    excluded_context_reason,
)


def expect(label: str, evidence: str, expected: str | None):
    actual = excluded_context_reason(label, evidence)
    assert actual == expected, (
        f"Expected {expected!r}, got {actual!r}: {evidence}"
    )


def main():
    expect(
        "FINANSMAN_TUTARI",
        "Odenecek toplam tutar: 10.787,28 TL.",
        "excluded_total_repayment_context",
    )
    expect(
        "FINANSMAN_TUTARI",
        "800.000 TL - 1.200.000 TL arasi vade 24 aydir.",
        "excluded_asset_value_tier_context",
    )
    expect(
        "VADE_SURESI",
        "Taahhut suresini 6 ay uzatabilirler.",
        "excluded_commitment_extension_context",
    )
    expect(
        "VADE_SURESI",
        "Riskli yapilarda en az 1 yil oturan kiracilar.",
        "excluded_residency_duration_context",
    )

    expect(
        "FINANSMAN_TUTARI",
        "Azami finansman tutari 3.000.000 TL olabilir.",
        None,
    )
    expect(
        "VADE_SURESI",
        "Finansman 6 ay vade ile sunulur.",
        None,
    )
    assert campaign_date_context_pass(
        "ampanya Araligi: 20 Ocak 2026 - 31 Aralik 2026"
    )
    assert AUTO_THRESHOLDS["FINANSMAN_TUTARI"] == 0.75
    assert AUTO_THRESHOLDS["HARCAMA_UST_LIMITI"] == 0.70
    assert AUTO_THRESHOLDS["KAR_PAYI_ORANI"] == 0.80
    print("Fact context rules V2.3: OK (10 cases)")


if __name__ == "__main__":
    main()
