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
    product_label: str,
    product_score: float,
    campaign_label: str,
    campaign_score: float,
):
    product = {"label": product_label, "score": product_score}
    return resolve_classification(
        text=f"{title}\nRelated products and navigation links.",
        page_title=title,
        source_url=source_url,
        campaign={"label": campaign_label, "score": campaign_score},
        product=product,
        product_ranked=[product],
        threshold=0.80,
    )


def main() -> None:
    card_campaign = resolve(
        title=(
            "Business Kart ile Elektronik Urun Harcamaniza "
            "500 TL Worldpuan"
        ),
        source_url=(
            "https://example.com/tr/kampanyalar/detay/"
            "business-kart-elektronik-worldpuan"
        ),
        product_label="KART_KAMPANYASI",
        product_score=0.9962,
        campaign_label="EVET",
        campaign_score=0.9981,
    )
    assert card_campaign["decision"] == "ACCEPTED", card_campaign
    assert card_campaign["decision_basis"] == (
        "title_rule_model_agreement"
    ), card_campaign
    assert card_campaign["strong_rule"]["reason"] == (
        "title:explicit_card_campaign"
    ), card_campaign

    model_specializes_generic_path = resolve(
        title="Yaza Ozel Avantajlar",
        source_url="https://example.com/kampanyalar/detay/yaza-ozel",
        product_label="KART_KAMPANYASI",
        product_score=0.9960,
        campaign_label="EVET",
        campaign_score=0.9980,
    )
    assert model_specializes_generic_path["decision"] == (
        "ACCEPTED"
    ), model_specializes_generic_path
    assert model_specializes_generic_path["decision_basis"] == (
        "high_confidence_model_over_url_advisory"
    ), model_specializes_generic_path
    assert model_specializes_generic_path["strong_rule"]["reason"] == (
        "url_advisory:campaign_path"
    ), model_specializes_generic_path

    historical_vehicle_conflict = resolve(
        title="ARAC FINANSMANI KAMPANYASI",
        source_url="https://example.com/arac-finansmani-kampanyasi.aspx",
        product_label="DIGER",
        product_score=0.9554,
        campaign_label="HAYIR",
        campaign_score=0.9985,
    )
    assert historical_vehicle_conflict["decision"] == (
        "REVIEW"
    ), historical_vehicle_conflict
    assert historical_vehicle_conflict["review_reasons"] == [
        "title_rule_model_conflict"
    ], historical_vehicle_conflict
    assert historical_vehicle_conflict["strong_rule"]["reason"] == (
        "title:vehicle_finance_phrase"
    ), historical_vehicle_conflict

    generic_old_campaign = resolve(
        title="DISHEKIMI FINANSMAN KAMPANYASI",
        source_url=(
            "https://example.com/dis-hekimi-finansman-kampanyasi.aspx"
        ),
        product_label="DIGER",
        product_score=0.9319,
        campaign_label="HAYIR",
        campaign_score=0.9990,
    )
    assert generic_old_campaign["decision"] == (
        "REVIEW"
    ), generic_old_campaign
    assert generic_old_campaign["review_reasons"] == [
        "campaign_url_non_campaign_model_conflict"
    ], generic_old_campaign
    assert generic_old_campaign["strong_rule"]["reason"] == (
        "url_advisory:campaign_path"
    ), generic_old_campaign

    low_confidence = resolve(
        title="Banka Hesabinizi Subeye Gitmeden Acin",
        source_url="https://example.com/tr/bireysel/banka-hesabi-acma",
        product_label="DIGER_KAMPANYA",
        product_score=0.7486,
        campaign_label="EVET",
        campaign_score=0.9934,
    )
    assert low_confidence["decision"] == "REVIEW", low_confidence
    assert low_confidence["review_reasons"] == [
        "low_product_confidence"
    ], low_confidence

    print("Classifier resolution V2.9: OK (5 cases)")


if __name__ == "__main__":
    main()
