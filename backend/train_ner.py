from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "ner_v4"
OUTPUT_DIR = BASE_DIR / "models" / "ner_v4_checkpoints"
BEST_MODEL_DIR = BASE_DIR / "models" / "ner_v4_best"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 256
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_labels(*splits: list[dict]) -> list[str]:
    entity_labels = {
        tag[2:]
        for split in splits
        for row in split
        for tag in row["ner_tags"]
        if tag != "O"
    }
    labels = ["O"]
    for entity in sorted(entity_labels):
        labels.extend([f"B-{entity}", f"I-{entity}"])
    return labels


def main() -> None:
    set_seed(SEED)

    train_rows = read_jsonl(DATA_DIR / "ner_train_bio.jsonl")
    val_rows = read_jsonl(DATA_DIR / "ner_val_bio.jsonl")
    test_rows = read_jsonl(DATA_DIR / "ner_test_bio.jsonl")

    label_list = collect_labels(train_rows, val_rows, test_rows)
    label2id = {label: index for index, label in enumerate(label_list)}
    id2label = {index: label for label, index in label2id.items()}

    print(f"Train passages: {len(train_rows)}")
    print(f"Validation passages: {len(val_rows)}")
    print(f"Test passages: {len(test_rows)}")
    print(f"BIO label count: {len(label_list)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_and_align(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            max_length=MAX_LENGTH,
            is_split_into_words=True,
        )
        aligned_labels = []
        for batch_index, word_labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label2id[word_labels[word_id]])
                else:
                    label_ids.append(-100)
                previous_word_id = word_id
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    raw_datasets = {
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(val_rows),
        "test": Dataset.from_list(test_rows),
    }
    tokenized_datasets = {
        name: dataset.map(
            tokenize_and_align,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {name}",
        )
        for name, dataset in raw_datasets.items()
    }

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def decode_predictions(predictions: np.ndarray, labels: np.ndarray):
        predicted_ids = np.argmax(predictions, axis=2)
        true_predictions = []
        true_labels = []
        for predicted_row, label_row in zip(predicted_ids, labels):
            row_predictions = []
            row_labels = []
            for predicted_id, label_id in zip(predicted_row, label_row):
                if label_id == -100:
                    continue
                row_predictions.append(id2label[int(predicted_id)])
                row_labels.append(id2label[int(label_id)])
            true_predictions.append(row_predictions)
            true_labels.append(row_labels)
        return true_predictions, true_labels

    def compute_metrics(eval_prediction) -> dict:
        predictions, labels = eval_prediction
        true_predictions, true_labels = decode_predictions(predictions, labels)
        return {
            "precision": precision_score(true_labels, true_predictions, zero_division=0),
            "recall": recall_score(true_labels, true_predictions, zero_division=0),
            "f1": f1_score(true_labels, true_predictions, zero_division=0),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    training_kwargs = {
        "output_dir": str(OUTPUT_DIR),
        "learning_rate": 3e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 8,
        "weight_decay": 0.01,
        "warmup_steps": 60,
        "save_strategy": "epoch",
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": 0,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
    }
    argument_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in argument_names:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in argument_names:
        training_kwargs["evaluation_strategy"] = "epoch"

    unsupported_args = [
        name for name in training_kwargs if name not in argument_names
    ]
    if unsupported_args:
        print(
            "Unsupported TrainingArguments ignored:",
            ", ".join(unsupported_args),
        )
    training_kwargs = {
        name: value
        for name, value in training_kwargs.items()
        if name in argument_names
    }
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(BEST_MODEL_DIR))

    test_output = trainer.predict(tokenized_datasets["test"])
    test_predictions, test_labels = decode_predictions(
        test_output.predictions,
        test_output.label_ids,
    )
    report = classification_report(
        test_labels,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "model_name": MODEL_NAME,
        "best_model_dir": str(BEST_MODEL_DIR),
        "labels": label_list,
        "test_metrics": test_output.metrics,
        "classification_report": report,
    }
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (BEST_MODEL_DIR / "test_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.item() if hasattr(value, "item") else str(value)
            ),
        )

    print("\nTraining completed.")
    print(f"Best model: {BEST_MODEL_DIR}")
    print(json.dumps(test_output.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
