from __future__ import annotations

import inspect
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "classification_v2"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 384
SEED = 42


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = functional.cross_entropy(
            outputs.logits,
            labels,
            weight=weights,
        )
        return (loss, outputs) if return_outputs else loss


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_default(value):
    return value.item() if hasattr(value, "item") else str(value)


def class_weights(label_ids: list[int], number_of_labels: int) -> torch.Tensor:
    counts = Counter(label_ids)
    total = len(label_ids)
    weights = np.array(
        [
            np.sqrt(total / max(counts.get(index, 0), 1))
            for index in range(number_of_labels)
        ],
        dtype=np.float32,
    )
    weights /= weights.mean()
    weights = np.clip(weights, 0.35, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def validate_splits(rows_by_split: dict[str, list[dict]]) -> None:
    required_fields = {"id", "text", "is_campaign", "product_type"}
    normalized_by_split: dict[str, set[str]] = {}
    ids_by_split: dict[str, set[str]] = {}

    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"Split is empty: {split}")
        for index, row in enumerate(rows, start=1):
            missing = required_fields - set(row)
            if missing:
                raise ValueError(
                    f"Missing fields in {split} row {index}: {sorted(missing)}"
                )
            if not str(row["text"]).strip():
                raise ValueError(f"Empty text in {split} row {index}")
        normalized_by_split[split] = {
            normalize_text(str(row["text"])) for row in rows
        }
        ids_by_split[split] = {
            str(row.get("source_document_id", row["id"])).split("::")[0]
            for row in rows
        }

    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for first, second in pairs:
        text_overlap = normalized_by_split[first] & normalized_by_split[second]
        id_overlap = ids_by_split[first] & ids_by_split[second]
        if text_overlap:
            raise ValueError(
                f"Normalized text leakage between {first} and {second}: "
                f"{len(text_overlap)}"
            )
        if id_overlap:
            raise ValueError(
                f"Document id leakage between {first} and {second}: "
                f"{len(id_overlap)}"
            )
    print("Split validation: PASS (no id or normalized text leakage)")


def make_training_arguments(output_dir: Path) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 8,
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "save_strategy": "epoch",
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": 0,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "epoch"

    unsupported = [name for name in kwargs if name not in parameters]
    if unsupported:
        print("Unsupported TrainingArguments ignored:", ", ".join(unsupported))
    kwargs = {name: value for name, value in kwargs.items() if name in parameters}
    return TrainingArguments(**kwargs)


def train_task(
    task_name: str,
    label_field: str,
    rows_by_split: dict[str, list[dict]],
    preferred_labels: list[str] | None = None,
) -> dict:
    all_labels = {
        row[label_field]
        for rows in rows_by_split.values()
        for row in rows
    }
    if preferred_labels is not None:
        labels = [label for label in preferred_labels if label in all_labels]
        labels.extend(sorted(all_labels - set(labels)))
    else:
        labels = sorted(all_labels)

    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    datasets = {}
    encoded_label_ids = {}
    for split, rows in rows_by_split.items():
        label_ids = [label2id[row[label_field]] for row in rows]
        encoded_label_ids[split] = label_ids
        dataset = Dataset.from_dict(
            {
                "text": [row["text"] for row in rows],
                "labels": label_ids,
            }
        )

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=MAX_LENGTH,
            )

        datasets[split] = dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {task_name} {split}",
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )
    output_dir = BASE_DIR / "models" / f"classifier_{task_name}_checkpoints"
    best_dir = BASE_DIR / "models" / f"classifier_{task_name}_best"

    def compute_metrics(prediction):
        predicted = np.argmax(prediction.predictions, axis=-1)
        expected = prediction.label_ids
        return {
            "accuracy": accuracy_score(expected, predicted),
            "macro_f1": f1_score(
                expected,
                predicted,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                expected,
                predicted,
                average="weighted",
                zero_division=0,
            ),
        }

    trainer_kwargs = {
        "model": model,
        "args": make_training_arguments(output_dir),
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
        "class_weights": class_weights(
            encoded_label_ids["train"],
            len(labels),
        ),
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    print("\nTask:", task_name)
    print("Labels:", labels)
    print("Train distribution:")
    train_distribution = Counter(
        row[label_field] for row in rows_by_split["train"]
    )
    for label in labels:
        print(f"  {label}: {train_distribution[label]}")

    trainer = WeightedTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    test_output = trainer.predict(datasets["test"])
    predicted = np.argmax(test_output.predictions, axis=-1)
    expected = test_output.label_ids
    report = classification_report(
        expected,
        predicted,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "task": task_name,
        "model_name": MODEL_NAME,
        "best_model_dir": str(best_dir),
        "labels": labels,
        "test_metrics": test_output.metrics,
        "classification_report": report,
    }
    best_dir.mkdir(parents=True, exist_ok=True)
    with (best_dir / "test_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )

    print("Best model:", best_dir)
    print(json.dumps(test_output.metrics, indent=2, default=json_default))
    return result


def main() -> None:
    set_seed(SEED)
    rows_by_split = {
        "train": read_jsonl(DATA_DIR / "classification_train_augmented.jsonl"),
        "validation": read_jsonl(DATA_DIR / "classification_validation.jsonl"),
        "test": read_jsonl(DATA_DIR / "classification_test.jsonl"),
    }
    validate_splits(rows_by_split)
    print("Train documents:", len(rows_by_split["train"]))
    print("Validation documents:", len(rows_by_split["validation"]))
    print("Test documents:", len(rows_by_split["test"]))
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    results = {
        "product": train_task(
            task_name="product_v2",
            label_field="product_type",
            rows_by_split=rows_by_split,
        )
    }
    summary_path = (
        BASE_DIR / "models" / "classifier_product_v2_training_summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            results,
            handle,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    print("\nProduct V2 classification completed.")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
