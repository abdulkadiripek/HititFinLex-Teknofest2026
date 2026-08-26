from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
CAMPAIGN_MODEL_DIR = BASE_DIR / "models" / "classifier_campaign_v1_best"
PRODUCT_MODEL_DIR = BASE_DIR / "models" / "classifier_product_v1_best"
MAX_LENGTH = 384
AUTO_ACCEPT_THRESHOLD = 0.80


def load_model(model_dir: Path, device: torch.device):
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Run python train_classifier.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_one(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> list[dict]:
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
    results = []
    for index in ranked.tolist():
        results.append(
            {
                "label": str(model.config.id2label[index]),
                "score": round(float(probabilities[index]), 4),
            }
        )
    return results


def main() -> None:
    text = input("Text: ").strip()
    if not text:
        raise SystemExit("Text cannot be empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    campaign_tokenizer, campaign_model = load_model(CAMPAIGN_MODEL_DIR, device)
    product_tokenizer, product_model = load_model(PRODUCT_MODEL_DIR, device)

    campaign_ranked = predict_one(
        text,
        campaign_tokenizer,
        campaign_model,
        device,
    )
    product_ranked = predict_one(
        text,
        product_tokenizer,
        product_model,
        device,
    )
    campaign = campaign_ranked[0]
    product = product_ranked[0]

    review_reasons = []
    if campaign["score"] < AUTO_ACCEPT_THRESHOLD:
        review_reasons.append("low_campaign_confidence")
    if product["score"] < AUTO_ACCEPT_THRESHOLD:
        review_reasons.append("low_product_confidence")

    campaign_product_types = {"KART_KAMPANYASI", "DIGER_KAMPANYA"}
    if product["label"] in campaign_product_types and campaign["label"] == "HAYIR":
        review_reasons.append("campaign_product_conflict")

    result = {
        "is_campaign": campaign,
        "product_type": product,
        "product_top3": product_ranked[:3],
        "decision": "REVIEW" if review_reasons else "ACCEPTED",
        "review_reasons": review_reasons,
        "threshold": AUTO_ACCEPT_THRESHOLD,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
