from __future__ import annotations

from archive_quality_v28 import deduplicate_facts
from coverage_rules_v27 import extract_coverage_facts
from fact_context_rules import (
    campaign_amount_roles,
    campaign_percent_roles,
    excluded_context_reason,
)
from fact_surface_rules import validate_entity_surface


def values(
    text: str,
    fact_type: str,
    product_code: str = "KART_KAMPANYASI",
) -> set[str]:
    return {
        fact.fact_text
        for fact in extract_coverage_facts(text, product_code)
        if fact.fact_type == fact_type
    }


def main() -> None:
    simple_campaign = (
        "Tek seferde yapacaginiz 2.000 TL ve uzeri harcamaya "
        "250 TL Worldpuan hediye!"
    )
    assert values(simple_campaign, "HARCAMA_ESIGI") == {"2.000 TL"}
    assert values(simple_campaign, "ODUL_TUTARI") == {"250 TL"}

    multi_reward = (
        "1.000 TL ve uzeri ucuncu harcamaya 100 TL, dorduncu "
        "harcamaya 150 TL, toplamda 250 TL Worldpuan hediye!"
    )
    assert values(multi_reward, "HARCAMA_ESIGI") == {"1.000 TL"}
    assert values(multi_reward, "ODUL_TUTARI") == {
        "100 TL",
        "150 TL",
        "250 TL",
    }

    implicit_threshold = (
        "Kartinizla 150 TL degerinde harcama yapmaniz halinde "
        "40 TL hediye bonus kazanabilirsiniz."
    )
    assert values(implicit_threshold, "HARCAMA_ESIGI") == {"150 TL"}
    assert values(implicit_threshold, "ODUL_TUTARI") == {"40 TL"}

    mixed_campaign = (
        "Aylik %1,37'den baslayan oranlarla Ihtiyac ve Tasit "
        "Finansmani kullanabilir, e-ticaret alisverislerinizden "
        "100 TL bonus kazanabilir, indirimli sigortadan "
        "yararlanabilirsiniz."
    )
    assert values(
        mixed_campaign,
        "KAR_PAYI_ORANI",
        "DIGER_KAMPANYA",
    ) == {"%1,37"}
    assert values(
        mixed_campaign,
        "ODUL_TUTARI",
        "DIGER_KAMPANYA",
    ) == {"100 TL"}
    assert values(
        mixed_campaign,
        "INDIRIM_ORANI",
        "DIGER_KAMPANYA",
    ) == set()
    assert values(
        mixed_campaign,
        "INDIRIM_TUTARI",
        "DIGER_KAMPANYA",
    ) == set()
    assert campaign_percent_roles("%1,37", mixed_campaign) == {
        "finance_rate"
    }

    promotion = "Emeklilerimize 3.500 TL'ye varan nakit promosyon."
    assert values(promotion, "ODUL_TUTARI") == {"3.500 TL"}
    monthly_bonus = (
        "Kartiniz ile 250 TL ve uzeri harcamalar icin aylik "
        "100 TL, toplam 300 TL bonus kazanin."
    )
    assert values(monthly_bonus, "HARCAMA_ESIGI") == {"250 TL"}
    assert values(monthly_bonus, "ODUL_TUTARI") == {"100 TL", "300 TL"}
    tiered_promotion = (
        "Maas 1.499 TL'ye kadarsa 1.150 TL, 1.500 TL-2.499 TL "
        "arasindaysa 1.425 TL, 2.500 TL'nin uzerindeyse "
        "2.250 TL nakit promosyon odenir."
    )
    assert values(tiered_promotion, "ODUL_TUTARI") == {
        "1.150 TL",
        "1.425 TL",
        "2.250 TL",
    }
    withdrawal = "ATM'lerden gunluk 2.000 TL'ye kadar ucretsiz para cekin."
    assert values(withdrawal, "ODUL_TUTARI") == set()
    assert excluded_context_reason(
        "ODUL_TUTARI",
        withdrawal,
        "2.000 TL",
    ) == "excluded_withdrawal_limit_context"

    legitimate_dual_role = (
        "Davet ettiginiz her emekli icin 250 TL nakit odul kazanin. "
        "Kartinizla 250 TL ve uzeri harcama yapin."
    )
    assert values(legitimate_dual_role, "ODUL_TUTARI") == {"250 TL"}
    assert values(legitimate_dual_role, "HARCAMA_ESIGI") == {"250 TL"}

    channels = (
        "Kampanya basvurunuzu Internet Subesi ya da Mobil Sube "
        "uzerinden yapabilirsiniz."
    )
    assert values(channels, "BASVURU_KANALI") == {
        "Internet Subesi",
        "Mobil Sube",
    }

    assert campaign_amount_roles("250 TL", simple_campaign) == {
        "reward_amount"
    }
    assert campaign_amount_roles("2.000 TL", simple_campaign) == {
        "spend_threshold"
    }
    assert excluded_context_reason(
        "HARCAMA_ESIGI",
        "250 TL'ye varan Worldpuan kazanabilirsiniz.",
        "250 TL",
    ) == "excluded_reward_amount_context"
    assert excluded_context_reason(
        "ODUL_TUTARI",
        "En az 2.000 TL ve uzeri harcama yapilmalidir.",
        "2.000 TL",
    ) == "excluded_spend_threshold_context"

    date_text = "Kampanya 9 Mayis-9 Haziran tarihleri arasinda gecerlidir."
    assert values(date_text, "KAMPANYA_TARIH_ARALIGI") == {
        "9 Mayis-9 Haziran"
    }
    assert validate_entity_surface(
        "KAMPANYA_TARIH_ARALIGI",
        "- 9 Haziran",
    ) == "date_range_incomplete"
    assert validate_entity_surface(
        "KAMPANYA_TARIH_ARALIGI",
        "9 Mayis - 9 Haziran 2022",
    ) is None

    cashback = "Market harcamalarinizda %10 iade kazanin."
    assert values(cashback, "INDIRIM_ORANI") == {"%10"}

    assert excluded_context_reason(
        "FINANSMAN_TUTARI",
        "Abonman policemizin maksimum limiti gemi icin 2.000.000 USD'dir.",
        "2.000.000 USD",
    ) == "excluded_insurance_policy_limit_context"

    incomplete_eligibility = (
        "Kampanyadan faydalanabilmek icin Albaraka Mobil uzerinden"
    )
    assert "UYGUNLUK_KOSULU" not in {
        fact.fact_type
        for fact in extract_coverage_facts(
            incomplete_eligibility,
            "KART_KAMPANYASI",
        )
    }
    complete_eligibility = (
        "Kampanyadan faydalanabilmek icin kartinizi kullanmaniz "
        "yeterlidir."
    )
    assert "UYGUNLUK_KOSULU" in {
        fact.fact_type
        for fact in extract_coverage_facts(
            complete_eligibility,
            "KART_KAMPANYASI",
        )
    }

    duplicate_facts = [
        {
            "fact_type": "TAKSIT_SAYISI",
            "fact_text": "3",
            "normalized_value": {"value": 3, "unit": "count"},
            "confidence": 0.99,
            "decision": "accepted",
        },
        {
            "fact_type": "TAKSIT_SAYISI",
            "fact_text": "3 taksit",
            "normalized_value": {"value": 3, "unit": "count"},
            "confidence": 0.97,
            "decision": "accepted",
        },
        {
            "fact_type": "KAMPANYA_TARIH_ARALIGI",
            "fact_text": "12 Subat 2026 - 22 Mart 2026",
            "normalized_value": {
                "raw": "12 Subat 2026 - 22 Mart 2026"
            },
            "confidence": 0.99,
            "decision": "accepted",
        },
        {
            "fact_type": "KAMPANYA_TARIH_ARALIGI",
            "fact_text": "12 Subat 2026- 22 Mart 2026",
            "normalized_value": {
                "raw": "12 Subat 2026- 22 Mart 2026"
            },
            "confidence": 0.98,
            "decision": "accepted",
        },
        {
            "fact_type": "KAMPANYA_TARIH_ARALIGI",
            "fact_text": "12 Subat - 22 Mart",
            "normalized_value": {"raw": "12 Subat - 22 Mart"},
            "confidence": 0.99,
            "decision": "accepted",
        },
    ]
    deduplicated = deduplicate_facts(duplicate_facts)
    assert len(deduplicated) == 2, deduplicated

    channel_facts = [
        {
            "fact_type": "BASVURU_KANALI",
            "fact_text": value,
            "normalized_value": None,
            "evidence_text": channels,
            "confidence": 0.96,
            "decision": "accepted",
        }
        for value in ("Internet Subesi", "Mobil Sube", "Sube")
    ]
    assert {
        fact["fact_text"] for fact in deduplicate_facts(channel_facts)
    } == {"Internet Subesi", "Mobil Sube"}

    print("NER quality V3.1: OK (31 cases)")


if __name__ == "__main__":
    main()
