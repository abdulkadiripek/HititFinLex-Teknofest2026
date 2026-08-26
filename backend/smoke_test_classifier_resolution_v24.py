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
    campaign_label: str = "HAYIR",
    campaign_score: float = 0.999,
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


def assert_accepted(
    *,
    name: str,
    title: str,
    source_url: str,
    product_label: str,
    product_score: float,
    expected_basis: str | None = None,
    expected_rule: str | None = None,
):
    result = resolve(
        title=title,
        source_url=source_url,
        product_label=product_label,
        product_score=product_score,
    )
    assert result["decision"] == "ACCEPTED", f"{name}: {result}"
    if expected_basis is not None:
        assert result["decision_basis"] == expected_basis, (
            f"{name}: {result}"
        )
    if expected_rule is not None:
        assert result["strong_rule"]["reason"] == expected_rule, (
            f"{name}: {result}"
        )


def main():
    assert_accepted(
        name="commercial_life_insurance",
        title="Ticari Kredili Hayat Sigortalari",
        source_url=(
            "https://example.com/isim-icin/sigortalar/"
            "hayat-sigortalari/ticari-kredili-hayat-sigortalari"
        ),
        product_label="SIGORTA_TEKAFUL_URUNU",
        product_score=0.9932,
        expected_basis="url_rule_model_agreement",
        expected_rule="url:insurance_path",
    )
    assert_accepted(
        name="corporate_accident_insurance",
        title="Finansman Ferdi Kaza Sigortasi",
        source_url=(
            "https://example.com/kurumsal/sigorta-ve-emeklilik/"
            "kurumsal-sigorta-urunleri/finansman-ferdi-kaza-sigortasi"
        ),
        product_label="SIGORTA_TEKAFUL_URUNU",
        product_score=0.9932,
        expected_basis="url_rule_model_agreement",
        expected_rule="url:insurance_path",
    )
    assert_accepted(
        name="generic_landing_agreement",
        title="Dunya Katilim",
        source_url="https://example.com/kendim-icin/finansmanlar",
        product_label="DIGER_FINANSMAN",
        product_score=0.4619,
        expected_basis="url_advisory_model_agreement",
        expected_rule="url_advisory:generic_finance_path",
    )

    advisory_override_cases = [
        (
            "generic_commercial",
            "Finansman Is Birlikleri",
            "https://example.com/finansman-urunleri/finansman-is-birlikleri",
            "TICARI_FINANSMAN",
            0.9946,
        ),
        (
            "generic_housing",
            "Arsa Finansmani",
            "https://example.com/kendim-icin/finansmanlar/arsa-finansmani",
            "KONUT_FINANSMANI",
            0.9925,
        ),
        (
            "generic_personal",
            "Alisveris Finansmani",
            "https://example.com/bireysel/finansman-urunleri/alisveris-finansmani",
            "IHTIYAC_FINANSMANI",
            0.9810,
        ),
    ]
    for name, title, source_url, product_label, product_score in (
        advisory_override_cases
    ):
        assert_accepted(
            name=name,
            title=title,
            source_url=source_url,
            product_label=product_label,
            product_score=product_score,
            expected_basis="high_confidence_model_over_url_advisory",
            expected_rule="url_advisory:generic_finance_path",
        )

    assert_accepted(
        name="top_level_commercial_path",
        title="Ziraat Katilim Bankasi",
        source_url="https://example.com/ticari/finansman-urunleri",
        product_label="TICARI_FINANSMAN",
        product_score=0.9952,
        expected_basis="url_rule_model_agreement",
        expected_rule="url:commercial_finance_path",
    )
    assert_accepted(
        name="agricultural_finance_path",
        title="Su Urunleri Finansmani",
        source_url=(
            "https://example.com/tarim/tarimsal-finansman-urunleri/"
            "su-urunleri-finansmani"
        ),
        product_label="TICARI_FINANSMAN",
        product_score=0.9941,
        expected_basis="url_rule_model_agreement",
        expected_rule="url:commercial_finance_path",
    )
    assert_accepted(
        name="title_beats_generic_url",
        title="Egitim Finansmani Sistemi",
        source_url="https://example.com/bireysel/finansman-urunleri",
        product_label="IHTIYAC_FINANSMANI",
        product_score=0.7434,
        expected_basis="title_rule_model_agreement",
        expected_rule="title:personal_finance_phrase",
    )

    low_conflict = resolve(
        title="Genel Sayfa",
        source_url="https://example.com/bireysel/finansman-urunleri",
        product_label="IHTIYAC_FINANSMANI",
        product_score=0.72,
    )
    assert low_conflict["decision"] == "REVIEW", low_conflict
    assert low_conflict["review_reasons"] == [
        "low_product_confidence"
    ], low_conflict

    strong_conflict = resolve(
        title="Konut Finansmani",
        source_url="https://example.com/konut-finansmani",
        product_label="IHTIYAC_FINANSMANI",
        product_score=0.99,
    )
    assert strong_conflict["decision"] == "REVIEW", strong_conflict
    assert strong_conflict["review_reasons"] == [
        "url_rule_model_conflict"
    ], strong_conflict

    print("Classifier resolution V2.4: OK (11 cases)")


if __name__ == "__main__":
    main()
