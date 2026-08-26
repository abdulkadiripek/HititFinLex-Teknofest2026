from coverage_rules_v27 import extract_coverage_facts


def fact_types(
    text: str,
    product_code: str,
    page_title: str = "",
) -> set[str]:
    return {
        fact.fact_type
        for fact in extract_coverage_facts(
            text,
            product_code,
            page_title=page_title,
        )
    }


def main() -> None:
    cases = [
        (
            "Kampanya 1 Agustos 2026 - 31 Agustos 2026 tarihlerinde gecerlidir.",
            "KART_KAMPANYASI",
            "KAMPANYA_TARIH_ARALIGI",
        ),
        (
            "Yapilan harcamalarda %20 indirim kazanabilirsiniz.",
            "KART",
            "INDIRIM_ORANI",
        ),
        (
            "Kampanya kapsaminda 500 TL iade kazanabilirsiniz.",
            "KART",
            "ODUL_TUTARI",
        ),
        (
            "En az 2.500 TL harcama yapan musteriler yararlanabilir.",
            "KART_KAMPANYASI",
            "HARCAMA_ESIGI",
        ),
        (
            "Alisverislerinize 6 taksit imkani sunulur.",
            "KART_KAMPANYASI",
            "TAKSIT_SAYISI",
        ),
        (
            "Basvurunuzu Mobil Sube veya internet subemizden yapabilirsiniz.",
            "KATILMA_HESABI",
            "BASVURU_KANALI",
        ),
        (
            "Butcenize uygun esnek odeme plani olusturabilirsiniz.",
            "KONUT_FINANSMANI",
            "ODEME_PLANI",
        ),
        (
            "Bu finansmanda KKDF ve BSMV alinmaz.",
            "IHTIYAC_FINANSMANI",
            "VERGI_MUAFIYETI",
        ),
        (
            "Hesabiniz TMSF guvencesi kapsamindadir.",
            "KATILMA_HESABI",
            "MEVDUAT_GUVENCESI",
        ),
        (
            "KOBI'lere yonelik finansman destegi sunulmaktadir.",
            "TICARI_FINANSMAN",
            "HEDEF_KITLE",
        ),
        (
            "Azami finansman tutari 3.000.000 TL olabilir.",
            "TICARI_FINANSMAN",
            "FINANSMAN_TUTARI",
        ),
        (
            "120 ay vade ile finansman kullanabilirsiniz.",
            "KONUT_FINANSMANI",
            "VADE_SURESI",
        ),
    ]
    for text, product_code, expected in cases:
        actual = fact_types(text, product_code)
        assert expected in actual, (text, expected, actual)

    assert not extract_coverage_facts(
        "Tum ziyaretcilerimize ucretsiz cay sunulur.",
        "DIGER",
    )
    assert "FINANSMAN_TUTARI" not in fact_types(
        "Satisa konu konutun degeri 3.000.000 TL'dir.",
        "KONUT_FINANSMANI",
    )
    assert "MASRAF_DURUMU" not in fact_types(
        "SMS bildirimleri ucretsizdir.",
        "KART_URUNU",
    )
    assert "FINANSMAN_AMACI" in fact_types(
        "Egitim ihtiyaclarina yonelik urundur.",
        "IHTIYAC_FINANSMANI",
        "Egitim Finansmani",
    )
    assert "YATIRIM_ARACI" in fact_types(
        "Birikimlerinizi degerlendirin.",
        "YATIRIM_URUNU",
        "Kira Sertifikasi (Sukuk)",
    )
    assert "KART_TURU" in fact_types(
        "Guvenli alisveris yapin.",
        "KART_URUNU",
        "Sanal Kart",
    )

    evidence_text = "Basvurunuzu Mobil Sube uzerinden yapabilirsiniz."
    facts = extract_coverage_facts(evidence_text, "KATILMA_HESABI")
    assert all(fact.fact_text in fact.evidence_text for fact in facts)
    assert all(fact.confidence >= 0.90 for fact in facts)

    print("Coverage rules V2.7: OK (18 cases)")


if __name__ == "__main__":
    main()
