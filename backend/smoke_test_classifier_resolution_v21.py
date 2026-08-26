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
    title: str | None,
    body: str,
    product_label: str,
    product_score: float,
    campaign_label: str = "HAYIR",
    campaign_score: float = 0.999,
    source_url: str | None = None,
):
    product = {"label": product_label, "score": product_score}
    return resolve_classification(
        text="\n".join(part for part in (title, body) if part),
        page_title=title,
        campaign={"label": campaign_label, "score": campaign_score},
        product=product,
        product_ranked=[product],
        threshold=0.80,
        source_url=source_url,
    )


def expect(
    name: str,
    result: dict[str, object],
    decision: str,
    basis: str | None = None,
):
    assert result["decision"] == decision, (
        f"{name}: expected {decision}, got {result}"
    )
    if basis is not None:
        assert result["decision_basis"] == basis, (
            f"{name}: expected basis {basis}, got {result}"
        )


def main():
    observed_cases = [
        (
            "kira_odeyen_hesap",
            "Kira Odeyen Hesap",
            "Katilma hesabi bilgileri ve ilgili urunler.",
            "IHTIYAC_FINANSMANI",
            0.5845,
            "title_rule_model_agreement",
        ),
        (
            "bes_teminatli",
            "BES Teminatli Finansman",
            "Ihtiyac finansmani hakkinda ek bilgiler.",
            "DIGER_FINANSMAN",
            0.9920,
            "title_rule_model_agreement",
        ),
        (
            "generic_finance",
            "Finansmanlar",
            "Konut finansmani dahil farkli urunler.",
            "DIGER_FINANSMAN",
            0.9914,
            "title_rule_model_agreement",
        ),
        (
            "bayide_kobi",
            "Bayide Finansman",
            "Ihtiyac finansmani ifadesi yan metinde geciyor.",
            "TICARI_FINANSMAN",
            0.9931,
            "high_confidence_model_over_body_rule",
        ),
        (
            "bayide_bireysel",
            "Bayide Finansman ile Satis Hacminize Katkida Bulunun",
            "Kredi karti menusu sayfada tekrar ediyor.",
            "DIGER_FINANSMAN",
            0.9914,
            "high_confidence_model_over_body_rule",
        ),
        (
            "kobi_nakdi",
            "Kobi Nakdi Finansman",
            "EFT ve havale secenekleri de listelenir.",
            "TICARI_FINANSMAN",
            0.9950,
            "title_rule_model_agreement",
        ),
        (
            "findeks",
            "Findeks Kredi Notu Ogrenme",
            "Sigorta menusu sayfada tekrar ediyor.",
            "DIGER_FINANSMAN",
            0.9912,
            "title_rule_model_agreement",
        ),
        (
            "home_needs",
            "Evinizin Ihtiyaclari Icin Avantajli Finansmanlari Kacirmayin",
            "Konut finansmani baglantisi yan alanda bulunuyor.",
            "IHTIYAC_FINANSMANI",
            0.9692,
            "high_confidence_model_over_body_rule",
        ),
        (
            "jet",
            "Jet Finansman",
            "Kredi karti limitini kullanmadan alisveris yapin.",
            "IHTIYAC_FINANSMANI",
            0.9919,
            "title_rule_model_agreement",
        ),
        (
            "leasing",
            "Leasing - Finansal Kiralama",
            "Sigorta secenekleri de anlatilmaktadir.",
            "TICARI_FINANSMAN",
            0.9804,
            "title_rule_model_agreement",
        ),
    ]

    for name, title, body, label, score, basis in observed_cases:
        result = resolve(
            title=title,
            body=body,
            product_label=label,
            product_score=score,
        )
        expect(name, result, "ACCEPTED", basis)

    expect(
        "mid_confidence_body_conflict",
        resolve(
            title="Yeni Finansman Urunu",
            body="Sigorta secenekleri de sunulur.",
            product_label="DIGER_FINANSMAN",
            product_score=0.90,
        ),
        "REVIEW",
        "model_body_rule_conflict",
    )
    expect(
        "title_conflict",
        resolve(
            title="Konut Finansmani",
            body="Bireysel finansman urunudur.",
            product_label="IHTIYAC_FINANSMANI",
            product_score=0.99,
        ),
        "REVIEW",
        "title_rule_model_conflict",
    )
    expect(
        "free_text_stays_conservative",
        resolve(
            title=None,
            body="Konut finansmani ve ihtiyac cozumleri.",
            product_label="IHTIYAC_FINANSMANI",
            product_score=0.99,
        ),
        "REVIEW",
        "rule_model_conflict",
    )
    expect(
        "low_confidence_without_rule",
        resolve(
            title="Avantajli Cozumler",
            body="Musterilerimize esnek cozumler sunuyoruz.",
            product_label="DIGER",
            product_score=0.55,
        ),
        "REVIEW",
        "low_product_confidence",
    )
    print("Classifier resolution V2.1: OK (14 cases)")


if __name__ == "__main__":
    main()
