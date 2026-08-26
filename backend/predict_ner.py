from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


DEFAULT_MODEL_DIR = Path("models/ner_v4_best")
DEFAULT_THRESHOLD = 0.40
MAX_LENGTH = 256
TOKEN_PATTERN = re.compile(r"\w+(?:[.,/]\w+)*|[^\w\s]", flags=re.UNICODE)
AMOUNT_PATTERN = re.compile(r"^\d[\d.,]*$")
CURRENCY_TOKENS = {"tl", "try", "₺", "usd", "eur", "dolar", "euro", "avro"}


def load_ner(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def split_bio(label: str) -> tuple[str, str]:
    if label == "O" or "-" not in label:
        return "O", ""
    prefix, entity_type = label.split("-", 1)
    return prefix, entity_type


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0), match.start(), match.end())
        for match in TOKEN_PATTERN.finditer(text)
    ]


def normalize_currency_spans(words: list[dict[str, object]]) -> None:
    for index in range(len(words) - 1):
        amount_word = words[index]
        currency_word = words[index + 1]
        amount_text = str(amount_word["text"])
        currency_text = str(currency_word["text"]).casefold()

        if not AMOUNT_PATTERN.fullmatch(amount_text):
            continue
        if currency_text not in CURRENCY_TOKENS:
            continue

        amount_prefix, amount_type = split_bio(str(amount_word["label"]))
        currency_prefix, currency_type = split_bio(str(currency_word["label"]))

        if amount_prefix != "O":
            currency_word["label"] = f"I-{amount_type}"
            currency_word["score"] = amount_word["score"]
        elif currency_prefix != "O":
            amount_word["label"] = f"B-{currency_type}"
            amount_word["score"] = currency_word["score"]
            currency_word["label"] = f"I-{currency_type}"


def predict_entities(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    threshold: float,
) -> list[dict[str, object]]:
    token_offsets = tokenize_with_offsets(text)
    tokens = [token for token, _, _ in token_offsets]
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    word_ids = encoded.word_ids(batch_index=0)
    encoded = {name: value.to(device) for name, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        scores, label_ids = probabilities.max(dim=-1)

    id2label = model.config.id2label
    words: list[dict[str, object]] = []
    current_word_id = None

    for token_index, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        if word_id != current_word_id:
            _, start, end = token_offsets[word_id]
            label_id = int(label_ids[token_index].item())
            words.append(
                {
                    "word_id": word_id,
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "label": id2label[label_id],
                    "score": float(scores[token_index].item()),
                }
            )
            current_word_id = word_id

    normalize_currency_spans(words)

    entities: list[dict[str, object]] = []
    active: dict[str, object] | None = None

    def close_active() -> None:
        nonlocal active
        if active is None:
            return
        active["text"] = text[int(active["start"]): int(active["end"])]
        active["score"] = round(
            sum(active.pop("token_scores")) / int(active.pop("token_count")),
            4,
        )
        if float(active["score"]) >= threshold:
            entities.append(active)
        active = None

    for word in words:
        prefix, entity_type = split_bio(str(word["label"]))

        if prefix == "O":
            close_active()
            continue

        should_continue = (
            prefix == "I"
            and active is not None
            and active["label"] == entity_type
        )

        if should_continue:
            active["end"] = word["end"]
            active["token_scores"].append(float(word["score"]))
            active["token_count"] = int(active["token_count"]) + 1
        else:
            close_active()
            active = {
                "label": entity_type,
                "start": word["start"],
                "end": word["end"],
                "token_scores": [float(word["score"])],
                "token_count": 1,
            }

    close_active()
    return entities


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained Turkish NER model.")
    parser.add_argument("text", nargs="*", help="Text to analyze")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    if not text:
        text = input("Text: ").strip()
    if not text:
        raise SystemExit("Text cannot be empty.")

    tokenizer, model, device = load_ner(args.model_dir)
    entities = predict_entities(
        text=text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        threshold=args.threshold,
    )

    print(f"Device: {device}")
    print(json.dumps(entities, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()