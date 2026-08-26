from __future__ import annotations

import sys
from types import ModuleType


try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ModuleNotFoundError:
    torch_stub = ModuleType("torch")
    transformers_stub = ModuleType("transformers")
    transformers_stub.AutoModelForSequenceClassification = object
    transformers_stub.AutoTokenizer = object
    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("transformers", transformers_stub)

from classifier_service import resolve_classification


def resolve(
    *,
    title: str,
    source_url: str,
    campaign_label: str,
    campaign_score: float,
    product_label: str,
    product_score: float,
    body: str = "Related products and navigation links.",
):
    product = {"label": product_label, "score": product_score}
    return resolve_classification(
        text=f"{title}\n{body}",
        page_title=title,
        source_url=source_url,
        campaign={"label": campaign_label, "score": campaign_score},
        product=product,
        product_ranked=[product],
        threshold=0.80,
    )


def main():
    cases = [
        (
            "generic_finance_landing",
            "Dunya Katilim",
            "https://dunyakatilim.com.tr/kendim-icin/finansmanlar",
            "HAYIR",
            0.9993,
            "DIGER_FINANSMAN",
            0.4619,
        ),
        (
            "education_finance",
            "Egitim Finansmani Sistemi",
            "https://hayatfinans.com.tr/krediler/egitim-finansmani-sistemi",
            "EVET",
            0.9836,
            "IHTIYAC_FINANSMANI",
            0.9897,
        ),
        (
            "campaign_path",
            "Emlak Konut Asansor Isbirligi",
            "https://example.com/tr/kurumsal/kampanyalar/asansor-isbirligi",
            "HAYIR",
            0.7884,
            "DIGER_KAMPANYA",
            0.9774,
        ),
        (
            "foreign_trade",
            "Dis Ticaret Odeme Yontemleri",
            "https://example.com/tr-tr/ticari/dis-ticaret-ve-finansmani/yontemler",
            "HAYIR",
            0.9992,
            "TICARI_FINANSMAN",
            0.8081,
        ),
        (
            "fast_finance_portal",
            "Ana Sayfa - Hizli Finansman",
            "https://www.hizlifinansman.com.tr/Sayfalar/default.aspx",
            "EVET",
            0.9913,
            "DIGER_FINANSMAN",
            0.4938,
        ),
        (
            "pilgrimage_finance",
            "Ziraat Katilim Bankasi",
            "https://example.com/ihtiyac-finansmani/hac-ve-umre-finansmani",
            "HAYIR",
            0.9990,
            "IHTIYAC_FINANSMANI",
            0.7434,
        ),
        (
            "private_banking_finance",
            "Ziraat Katilim Bankasi",
            "https://example.com/ozel-bankacilik/finansman-urunleri",
            "HAYIR",
            0.9992,
            "DIGER_FINANSMAN",
            0.8128,
        ),
    ]

    for (
        name,
        title,
        source_url,
        campaign_label,
        campaign_score,
        product_label,
        product_score,
    ) in cases:
        result = resolve(
            title=title,
            source_url=source_url,
            campaign_label=campaign_label,
            campaign_score=campaign_score,
            product_label=product_label,
            product_score=product_score,
        )
        assert result["decision"] == "ACCEPTED", f"{name}: {result}"

    conflict = resolve(
        title="Konut Finansmani",
        source_url="https://example.com/konut-finansmani",
        campaign_label="HAYIR",
        campaign_score=0.99,
        product_label="IHTIYAC_FINANSMANI",
        product_score=0.99,
    )
    assert conflict["decision"] == "REVIEW", conflict
    assert conflict["review_reasons"] == [
        "url_rule_model_conflict"
    ], conflict
    print("Classifier resolution V2.3: OK (8 cases)")


if __name__ == "__main__":
    main()
