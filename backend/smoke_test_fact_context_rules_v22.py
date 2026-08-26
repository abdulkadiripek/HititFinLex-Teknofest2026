from __future__ import annotations

from fact_context_rules import excluded_context_reason


def expect(label: str, evidence: str, expected: str | None):
    actual = excluded_context_reason(label, evidence)
    assert actual == expected, (
        f"Expected {expected!r}, got {actual!r}: {evidence}"
    )


def main():
    expect(
        "VADE_SURESI",
        (
            "En az 10 yil sistemde kalanlar devlet katkisi ve "
            "getirilerinin yuzde 60'ina hak kazanir."
        ),
        "excluded_membership_duration_context",
    )
    expect(
        "VADE_SURESI",
        (
            "lk taksit icin en gec 45 gun sonrasina kadar tarih "
            "secebilirsiniz."
        ),
        "excluded_first_payment_timing_context",
    )
    expect(
        "FINANSMAN_TUTARI",
        (
            "Finansman tutari 7500 TL'yi asmayan islemlerde "
            "gelir belgesi beyani talep edilmeyecektir."
        ),
        "excluded_document_requirement_threshold_context",
    )
    expect(
        "VADE_SURESI",
        "Kalan vadesi 36 ayi asan erken odemelerde tazminat uygulanir.",
        "excluded_early_payment_context",
    )
    expect(
        "FINANSMAN_TUTARI",
        "Gayrimenkul degeri 5.000.000 TL seviyesindedir.",
        "conflicting_property_value_context",
    )

    # Valid product facts must not be excluded.
    expect(
        "VADE_SURESI",
        "Finansman 10 yil vade ile kullanilabilir.",
        None,
    )
    expect(
        "VADE_SURESI",
        "Urun 45 gun vade ile sunulmaktadir.",
        None,
    )
    expect(
        "FINANSMAN_TUTARI",
        "Musteriler 7500 TL finansman kullanabilir.",
        None,
    )
    print("Fact context rules V2.2: OK (8 cases)")


if __name__ == "__main__":
    main()
