from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN_MODEL_DIR = (
    BASE_DIR / "models" / "classifier_campaign_v1_best"
)
DEFAULT_PRODUCT_MODEL_DIR = (
    BASE_DIR / "models" / "classifier_product_v2_best"
)
MAX_LENGTH = 384
BODY_RULE_OVERRIDE_THRESHOLD = 0.95


@dataclass
class SequenceClassifier:
    tokenizer: object
    model: object
    model_dir: Path


@dataclass
class ClassifierBundle:
    campaign: SequenceClassifier
    product: SequenceClassifier
    device: torch.device


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )
    value = value.replace("ı", "i")
    return re.sub(r"[^a-z0-9%]+", " ", value).strip()


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def strong_product_rule(
    text: str,
    is_campaign: str,
) -> tuple[str | None, str | None]:
    value = normalize(text)
    if is_campaign == "EVET":
        if contains_any(
            value,
            (
                "kredi kart",
                "banka kart",
                "mastercard",
                "worldpuan",
                "bankkart",
                "paraf",
                "business kart",
                "debit kart",
            ),
        ):
            return "KART_KAMPANYASI", "explicit_card_campaign"
        return "DIGER_KAMPANYA", "non_card_campaign"

    if contains_any(
        value,
        (
            "kira odeyen hesap",
            "fatura odeyen hesap",
            "aidat odeyen hesap",
            "jet finansman",
        ),
    ):
        return "IHTIYAC_FINANSMANI", "named_personal_finance_product"
    if contains_any(
        value,
        (
            "bes teminatli finansman",
            "findeks kredi notu",
        ),
    ):
        return "DIGER_FINANSMAN", "named_other_finance_product"
    if value in {"finansman", "finansmanlar"}:
        return "DIGER_FINANSMAN", "generic_finance_title"
    if contains_any(
        value,
        (
            "leasing",
            "finansal kiralama",
            "kobi nakdi finans",
            "kobi gayri nakdi finans",
        ),
    ):
        return "TICARI_FINANSMAN", "named_commercial_finance_product"

    if contains_any(
        value,
        ("sigorta", "dask", "kasko", "tekaf", "ferdi kaza"),
    ):
        return "SIGORTA_TEKAFUL_URUNU", "insurance_phrase"
    if contains_any(value, ("konut finans", "mortgage")) or (
        "konut" in value
        and contains_any(value, ("finansman", "ev sahibi"))
    ):
        return "KONUT_FINANSMANI", "housing_finance_phrase"
    if contains_any(
        value,
        (
            "tasit finans",
            "arac finans",
            "motosiklet finans",
            "togg finans",
        ),
    ):
        return "TASIT_FINANSMANI", "vehicle_finance_phrase"
    if contains_any(
        value,
        (
            "ihtiyac finans",
            "alisveris kredi",
            "egitim finans",
            "hazir limit",
        ),
    ):
        return "IHTIYAC_FINANSMANI", "personal_finance_phrase"
    if contains_any(
        value,
        (
            "kredi kart",
            "banka kart",
            "sanal kart",
            "ek kart",
            "business kart",
        ),
    ):
        return "KART_URUNU", "card_product_phrase"
    if contains_any(
        value,
        (
            "katilma hesabi",
            "kar payi odemeli hesap",
            "gunluk kazandiran hesap",
        ),
    ):
        return "KATILMA_HESABI", "participation_account_phrase"
    if contains_any(
        value,
        (
            "para transfer",
            "moneygram",
            "swift",
            "eft",
            "havale",
            "fatura odeme",
            "sanal pos",
        ),
    ):
        return "ODEME_TRANSFER_HIZMETI", "payment_transfer_phrase"
    if contains_any(
        value,
        (
            "yatirim fon",
            "hisse sened",
            "kira sertifika",
            "sukuk",
            "kiymetli maden",
            "altin hesabi",
        ),
    ):
        return "YATIRIM_URUNU", "investment_phrase"
    if contains_any(
        value,
        (
            "ticari finans",
            "kobi finans",
            "isletme finans",
            "tedarikci finans",
            "teminat mektubu",
        ),
    ):
        return "TICARI_FINANSMAN", "commercial_finance_phrase"
    return None, None


def load_sequence_classifier(
    model_dir: Path,
    device: torch.device,
) -> SequenceClassifier:
    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Classifier model directory not found: {model_dir}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return SequenceClassifier(
        tokenizer=tokenizer,
        model=model,
        model_dir=model_dir,
    )


def load_classifiers(
    campaign_model_dir: Path = DEFAULT_CAMPAIGN_MODEL_DIR,
    product_model_dir: Path = DEFAULT_PRODUCT_MODEL_DIR,
) -> ClassifierBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    campaign = load_sequence_classifier(campaign_model_dir, device)
    product = load_sequence_classifier(product_model_dir, device)
    return ClassifierBundle(
        campaign=campaign,
        product=product,
        device=device,
    )


def predict_ranked(
    text: str,
    classifier: SequenceClassifier,
    device: torch.device,
) -> list[dict[str, object]]:
    encoded = classifier.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded = {
        name: value.to(device)
        for name, value in encoded.items()
    }
    with torch.inference_mode():
        logits = classifier.model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    ranked = torch.argsort(probabilities, descending=True)
    return [
        {
            "label": str(classifier.model.config.id2label[index]),
            "score": round(float(probabilities[index]), 4),
        }
        for index in ranked.tolist()
    ]


def resolve_classification(
    *,
    text: str,
    page_title: str | None,
    campaign: dict[str, object],
    product: dict[str, object],
    product_ranked: list[dict[str, object]],
    threshold: float,
) -> dict[str, object]:
    """Resolve model and rule evidence without trusting page-wide keywords.

    A title rule is strong evidence. A body-only rule is advisory because bank
    pages frequently repeat navigation menus and related product links. When
    a high-confidence model conflicts only with a body rule, the model wins.
    Mid-confidence conflicts remain in human review.
    """

    campaign_label = str(campaign["label"])
    product_label = str(product["label"])
    product_score = float(product["score"])
    clean_title = (page_title or "").strip()

    body_rule_label, body_rule_reason = strong_product_rule(
        text,
        campaign_label,
    )
    title_rule_label = None
    title_rule_reason = None
    if clean_title:
        title_rule_label, title_rule_reason = strong_product_rule(
            clean_title,
            campaign_label,
        )

    if title_rule_label is not None:
        rule_label = title_rule_label
        rule_reason = f"title:{title_rule_reason}"
        rule_scope = "title"
    elif body_rule_label is not None:
        rule_label = body_rule_label
        rule_reason = f"body_advisory:{body_rule_reason}"
        rule_scope = "body_advisory"
    else:
        rule_label = None
        rule_reason = None
        rule_scope = None

    review_reasons: list[str] = []
    decision_basis = "model_confidence"
    if float(campaign["score"]) < threshold:
        review_reasons.append("low_campaign_confidence")

    if clean_title:
        if rule_scope == "title":
            if rule_label == product_label:
                decision_basis = "title_rule_model_agreement"
            else:
                review_reasons.append("title_rule_model_conflict")
                decision_basis = "title_rule_model_conflict"
        elif product_score < threshold:
            review_reasons.append("low_product_confidence")
            decision_basis = "low_product_confidence"
        elif (
            rule_scope == "body_advisory"
            and rule_label != product_label
        ):
            if product_score >= max(
                threshold,
                BODY_RULE_OVERRIDE_THRESHOLD,
            ):
                decision_basis = "high_confidence_model_over_body_rule"
            else:
                review_reasons.append("model_body_rule_conflict")
                decision_basis = "model_body_rule_conflict"
        elif rule_scope == "body_advisory":
            decision_basis = "model_body_rule_agreement"
        else:
            decision_basis = "high_confidence_model"
    else:
        # Free-text API calls have no independent title signal. Preserve the
        # conservative behavior: a strong rule conflict requires review.
        if rule_label is not None:
            if rule_label == product_label:
                decision_basis = "rule_model_agreement"
            else:
                review_reasons.append("rule_model_conflict")
                decision_basis = "rule_model_conflict"
        elif product_score < threshold:
            review_reasons.append("low_product_confidence")

    return {
        "is_campaign": campaign,
        "product_type": product,
        "product_top3": product_ranked[:3],
        "strong_rule": {
            "label": rule_label,
            "reason": rule_reason,
        },
        "decision": "REVIEW" if review_reasons else "ACCEPTED",
        "decision_basis": decision_basis,
        "review_reasons": review_reasons,
        "model_threshold": threshold,
    }


def classify_text(
    text: str,
    bundle: ClassifierBundle,
    threshold: float = 0.80,
    page_title: str | None = None,
) -> dict[str, object]:
    campaign_ranked = predict_ranked(
        text,
        bundle.campaign,
        bundle.device,
    )
    product_ranked = predict_ranked(
        text,
        bundle.product,
        bundle.device,
    )
    result = resolve_classification(
        text=text,
        page_title=page_title,
        campaign=campaign_ranked[0],
        product=product_ranked[0],
        product_ranked=product_ranked,
        threshold=threshold,
    )
    result["campaign_model"] = bundle.campaign.model_dir.name
    result["product_model"] = bundle.product.model_dir.name
    return result
