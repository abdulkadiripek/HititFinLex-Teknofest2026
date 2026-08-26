from __future__ import annotations

import argparse
import tempfile
import zipfile
from datetime import date
from pathlib import Path

from archive_common_v28 import (
    canonicalize_url,
    content_digest,
    iter_archive_documents,
    validate_zip_members,
)
from archive_quality_v28 import classification_quality, deduplicate_facts
from archive_training_utils_v28 import annotate_evidence, choose_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-zip",
        default="HititFinLex_VeriSeti_2.zip",
    )
    return parser.parse_args()


def test_url_normalization() -> None:
    first = canonicalize_url(
        "http://www.example.com/Urunler/Konut-Finansmani/?utm_source=x#top"
    )
    second = canonicalize_url(
        "https://example.com/Urunler/Konut-Finansmani"
    )
    assert first == second, (first, second)


def test_zip_safety() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "unsafe.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../escape.txt", "blocked")
        with zipfile.ZipFile(path) as archive:
            try:
                validate_zip_members(archive)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsafe ZIP path was not rejected")


def test_classification_gate() -> None:
    high_model = {
        "decision": "ACCEPTED",
        "product_type": {"label": "TICARI_FINANSMAN", "score": 0.99},
        "strong_rule": {"label": None, "reason": None},
    }
    assert classification_quality(high_model, 0.80) == "accepted"
    metadata_rule = {
        "decision": "ACCEPTED",
        "product_type": {"label": "IHTIYAC_FINANSMANI", "score": 0.74},
        "strong_rule": {
            "label": "IHTIYAC_FINANSMANI",
            "reason": "title:personal_finance_phrase",
        },
    }
    assert classification_quality(metadata_rule, 0.80) == "accepted"
    body_only = {
        "decision": "ACCEPTED",
        "product_type": {"label": "DIGER_FINANSMAN", "score": 0.55},
        "strong_rule": {
            "label": "DIGER_FINANSMAN",
            "reason": "body_advisory:generic_finance_title",
        },
    }
    assert classification_quality(body_only, 0.80) == "review"
    model_review = {
        "decision": "REVIEW",
        "product_type": {"label": "DIGER", "score": 0.99},
        "strong_rule": {"label": None, "reason": None},
    }
    assert classification_quality(model_review, 0.80) == "review"


def test_fact_deduplication() -> None:
    facts = [
        {
            "fact_type": "VADE_SURESI",
            "fact_text": "24 ay",
            "normalized_value": None,
            "evidence_text": "Vade 24 ay.",
            "confidence": 0.80,
            "source_chunk": 1,
            "decision": "review",
            "decision_reason": "low",
            "extraction_method": "ner",
        },
        {
            "fact_type": "VADE_SURESI",
            "fact_text": "24 ay",
            "normalized_value": None,
            "evidence_text": "Azami vade 24 ay.",
            "confidence": 0.99,
            "source_chunk": 0,
            "decision": "accepted",
            "decision_reason": "rule",
            "extraction_method": "rule",
        },
    ]
    result = deduplicate_facts(facts)
    assert len(result) == 1
    assert result[0]["decision"] == "accepted"


def test_training_helpers() -> None:
    rows = [
        {
            "id": index,
            "canonical_group_key": "same",
            "snapshot_date": date(2020 + index, 1, 1),
        }
        for index in range(1, 5)
    ]
    selected = choose_versions(rows, 2)
    assert [row["id"] for row in selected] == [1, 4]
    annotated = annotate_evidence(
        "sample",
        "doc",
        "Azami finansman tutari 500.000 TL ve vade 24 aydir.",
        [
            {
                "text": "500.000 TL",
                "label": "FINANSMAN_TUTARI",
                "confidence": 0.99,
            },
            {
                "text": "24 ay",
                "label": "VADE_SURESI",
                "confidence": 0.99,
            },
        ],
    )
    assert annotated is not None
    assert "B-FINANSMAN_TUTARI" in annotated["ner_tags"]
    assert "B-VADE_SURESI" in annotated["ner_tags"]


def test_dataset(zip_path: Path) -> None:
    documents = list(iter_archive_documents(zip_path))
    assert len(documents) == 2580, len(documents)
    hashes = {document.content_hash for document in documents}
    assert len(hashes) == 2580, len(hashes)
    assert all(document.raw_text for document in documents)
    assert all(document.canonical_url for document in documents)
    assert content_digest(documents[0].raw_text) == documents[0].content_hash


def main() -> None:
    args = parse_args()
    test_url_normalization()
    test_zip_safety()
    test_classification_gate()
    test_fact_deduplication()
    test_training_helpers()
    test_dataset(Path(args.dataset_zip))
    print("HititFinLex V2.8 historical pipeline: OK (6 groups, 2580 rows)")


if __name__ == "__main__":
    main()
