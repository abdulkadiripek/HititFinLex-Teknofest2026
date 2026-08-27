from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


DEFAULT_NER_MODEL_DIR = (
    Path(__file__).resolve().parent / "models" / "ner_v4_best"
)
MAX_LENGTH = 256
NER_WINDOW_OVERLAP_WORDS = 32
TOKEN_PATTERN = re.compile(r"\w+(?:[.,/]\w+)*|[^\w\s]", flags=re.UNICODE)
AMOUNT_PATTERN = re.compile(r"^\d[\d.,]*$")
CURRENCY_TOKENS = {
    "tl",
    "try",
    "₺",
    "usd",
    "eur",
    "dolar",
    "euro",
    "avro",
}


@dataclass
class NerBundle:
    tokenizer: object
    model: object
    device: torch.device
    model_dir: Path


def load_ner(model_dir: Path = DEFAULT_NER_MODEL_DIR) -> NerBundle:
    model_dir = Path(model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"NER model directory not found: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return NerBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_dir=model_dir,
    )


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
        currency_prefix, currency_type = split_bio(
            str(currency_word["label"])
        )

        if amount_prefix != "O":
            currency_word["label"] = f"I-{amount_type}"
            currency_word["score"] = amount_word["score"]
        elif currency_prefix != "O":
            amount_word["label"] = f"B-{currency_type}"
            amount_word["score"] = currency_word["score"]
            currency_word["label"] = f"I-{currency_type}"


def predict_entities(
    text: str,
    bundle: NerBundle,
    threshold: float = 0.40,
) -> list[dict[str, object]]:
    entities, _ = predict_entities_with_metadata(
        text=text,
        bundle=bundle,
        threshold=threshold,
    )
    return entities


def predict_entities_with_metadata(
    text: str,
    bundle: NerBundle,
    threshold: float = 0.40,
) -> tuple[list[dict[str, object]], dict[str, int | bool]]:
    token_offsets = tokenize_with_offsets(text)
    if not token_offsets:
        return [], {
            "input_word_count": 0,
            "model_chunk_count": 0,
            "truncated": False,
        }

    # A tokenizer word can expand to multiple wordpieces, so a fixed number of
    # source words is not guaranteed to fit MAX_LENGTH. Adjacent windows retain
    # overlap; predictions nearest a window's center win for repeated words.
    # This prevents entities at a tokenizer boundary from losing their context.
    id2label = bundle.model.config.id2label
    predictions_by_word: dict[int, dict[str, object]] = {}
    next_word = 0
    model_chunk_count = 0

    while next_word < len(token_offsets):
        candidate_offsets = token_offsets[
            next_word: next_word + MAX_LENGTH
        ]
        encoded_batch = bundle.tokenizer(
            [token for token, _, _ in candidate_offsets],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        word_ids = encoded_batch.word_ids(batch_index=0)
        encoded = {
            name: value.to(bundle.device)
            for name, value in encoded_batch.items()
        }

        encoded_word_ids = [
            int(word_id) for word_id in word_ids if word_id is not None
        ]
        if not encoded_word_ids:
            raise RuntimeError("NER tokenizer did not encode any source words.")
        covered_word_count = max(encoded_word_ids) + 1
        if covered_word_count > len(candidate_offsets):
            raise RuntimeError("NER tokenizer returned an invalid source word id.")

        with torch.inference_mode():
            logits = bundle.model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            scores, label_ids = probabilities.max(dim=-1)

        current_word_id = None
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id == current_word_id:
                continue

            global_word_id = next_word + int(word_id)
            token_text, start, end = token_offsets[global_word_id]
            label_id = int(label_ids[token_index].item())
            score = float(scores[token_index].item())
            local_word_id = int(word_id)
            center_distance = min(
                local_word_id,
                covered_word_count - local_word_id - 1,
            )
            prediction = {
                "word_id": global_word_id,
                "start": start,
                "end": end,
                "text": token_text,
                "label": id2label[label_id],
                "score": score,
                "_center_distance": center_distance,
            }
            previous = predictions_by_word.get(global_word_id)
            if previous is None or (
                center_distance,
                score,
            ) > (
                int(previous["_center_distance"]),
                float(previous["score"]),
            ):
                predictions_by_word[global_word_id] = prediction
            current_word_id = word_id

        model_chunk_count += 1
        window_end = next_word + covered_word_count
        if window_end >= len(token_offsets):
            break
        overlap = min(
            NER_WINDOW_OVERLAP_WORDS,
            max(1, covered_word_count // 4),
            max(0, covered_word_count - 1),
        )
        following_word = window_end - overlap
        if following_word <= next_word:
            following_word = next_word + 1
        next_word = following_word

    missing_word_ids = [
        word_id
        for word_id in range(len(token_offsets))
        if word_id not in predictions_by_word
    ]
    if missing_word_ids:
        raise RuntimeError(
            "NER tokenizer skipped source words while processing windows."
        )
    words = [predictions_by_word[word_id] for word_id in range(len(token_offsets))]
    for word in words:
        word.pop("_center_distance", None)

    normalize_currency_spans(words)
    entities: list[dict[str, object]] = []
    active: dict[str, object] | None = None

    def close_active() -> None:
        nonlocal active
        if active is None:
            return

        active["text"] = text[
            int(active["start"]): int(active["end"])
        ]
        active["score"] = round(
            sum(active.pop("token_scores"))
            / int(active.pop("token_count")),
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
    return entities, {
        "input_word_count": len(token_offsets),
        "model_chunk_count": model_chunk_count,
        "truncated": False,
    }
