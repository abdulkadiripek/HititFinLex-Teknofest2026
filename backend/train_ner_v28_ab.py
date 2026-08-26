from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V2.8 A/B NER model without replacing NER V4."
    )
    parser.add_argument(
        "--data-dir",
        default="data/ab_v28/ner_v28",
    )
    parser.add_argument(
        "--output-root",
        default="ab_models_v28/ner",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import train_ner as trainer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "train_ner.py was not found in the project root."
        ) from error

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trainer.DATA_DIR = data_dir
    trainer.OUTPUT_DIR = output_root / "ner_v28_checkpoints"
    trainer.BEST_MODEL_DIR = output_root / "ner_v28_best"
    print("A/B data directory:", data_dir)
    print("A/B output root:", output_root)
    print("Current production model will not be replaced.")
    trainer.main()


if __name__ == "__main__":
    main()
