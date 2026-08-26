from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
CAMPAIGN_MODEL_DIR = BASE_DIR / "models" / "classifier_campaign_v1_best"
PRODUCT_MODEL_DIR = BASE_DIR / "models" / "classifier_product_v2_best"
MAX_LENGTH = 384
AUTO_ACCEPT_THRESHOLD = 0.80


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.replace("ı", "i")
    return re.sub(r"[^a-z0-9%]+", " ", value).strip()


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def strong_product_rule(text: str, is_campaign: str) -> tuple[str | None, str | None]:
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

    if contains_any(value, ("sigorta", "dask", "kasko", "tekaf", "ferdi kaza")):
        return "SIGORTA_TEKAFUL_URUNU", "insurance_phrase"
    if contains_any(value, ("konut finans", "mortgage")) or (
        "konut" in value and contains_any(value, ("finansman", "ev sahibi"))
    ):
        return "KONUT_FINANSMANI", "housing_finance_phrase"
    if contains_any(value, ("tasit finans", "arac finans", "motosiklet finans", "togg finans")):
        return "TASIT_FINANSMANI", "vehicle_finance_phrase"
    if contains_any(value, ("ihtiyac finans", "alisveris kredi", "egitim finans", "hazir limit")):
        return "IHTIYAC_FINANSMANI", "personal_finance_phrase"
    if contains_any(
        value,
        ("kredi kart", "banka kart", "sanal kart", "ek kart", "business kart"),
    ):
        return "KART_URUNU", "card_product_phrase"
    if contains_any(
        value,
        ("katilma hesabi", "kar payi odemeli hesap", "gunluk kazandiran hesap"),
    ):
        return "KATILMA_HESABI", "participation_account_phrase"
    if contains_any(
        value,
        ("para transfer", "moneygram", "swift", "eft", "havale", "fatura odeme", "sanal pos"),
    ):
        return "ODEME_TRANSFER_HIZMETI", "payment_transfer_phrase"
    if contains_any(
        value,
        ("yatirim fon", "hisse sened", "kira sertifika", "sukuk", "kiymetli maden", "altin hesabi"),
    ):
        return "YATIRIM_URUNU", "investment_phrase"
    if contains_any(
        value,
        ("ticari finans", "kobi finans", "isletme finans", "tedarikci finans", "teminat mektubu"),
    ):
        return "TICARI_FINANSMAN", "commercial_finance_phrase"
    return None, None


def load_model(model_dir: Path, device: torch.device):
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Run python train_product_v2.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_ranked(text: str, tokenizer, model, device: torch.device) -> list[dict]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits[0], dim=-1)
    ranked = torch.argsort(probabilities, descending=True)
    return [
        {
            "label": str(model.config.id2label[index]),
            "score": round(float(probabilities[index]), 4),
        }
        for index in ranked.tolist()
    ]


def main() -> None:
    text = input("Text: ").strip()
    if not text:
        raise SystemExit("Text cannot be empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    campaign_tokenizer, campaign_model = load_model(CAMPAIGN_MODEL_DIR, device)
    product_tokenizer, product_model = load_model(PRODUCT_MODEL_DIR, device)

    campaign_ranked = predict_ranked(
        text, campaign_tokenizer, campaign_model, device
    )
    product_ranked = predict_ranked(text, product_tokenizer, product_model, device)
    campaign = campaign_ranked[0]
    product = product_ranked[0]
    rule_label, rule_reason = strong_product_rule(text, campaign["label"])

    review_reasons = []
    decision_basis = "model_confidence"
    if campaign["score"] < AUTO_ACCEPT_THRESHOLD:
        review_reasons.append("low_campaign_confidence")
    if rule_label is not None:
        if rule_label == product["label"]:
            decision_basis = "rule_model_agreement"
        else:
            review_reasons.append("rule_model_conflict")
            decision_basis = "rule_model_conflict"
    elif product["score"] < AUTO_ACCEPT_THRESHOLD:
        review_reasons.append("low_product_confidence")

    result = {
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
        "model_threshold": AUTO_ACCEPT_THRESHOLD,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
