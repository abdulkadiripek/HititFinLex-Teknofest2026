from __future__ import annotations

import httpx


API_URL = "http://127.0.0.1:8000"


def post_intake(client: httpx.Client, payload: dict) -> dict:
    response = client.post("/intake", json=payload)
    response.raise_for_status()
    return response.json()


def main():
    base = {
        "record_key": "smoke-test-konut-001",
        "bank_key": "smoke-test-bank",
        "bank_name": "Smoke Test Bank",
        "source_url": "https://example.com/smoke-test",
        "page_title": "Konut Finansmani",
        "classification_threshold": 0.8,
        "ner_threshold": 0.4,
        "review_threshold": 0.6,
        "write": False,
        "allow_update": False,
    }
    accepted_text = (
        "Konut finansmani kapsaminda 500.000 TL finansman 120 ay vade ve "
        "%2,79 kar payi orani ile sunuluyor. Tahsis ucreti 2.500 TL, "
        "ekspertiz ucreti 8.000 TL ve ipotek tesis ucreti 3.000 TL'dir."
    )
    review_text = (
        "Bankamiz musterilerine avantajli, hizli ve esnek cozumler "
        "sunmaktadir."
    )

    with httpx.Client(base_url=API_URL, timeout=180.0) as client:
        accepted = post_intake(
            client,
            {**base, "raw_text": accepted_text},
        )
        assert accepted["status"] == "ACCEPTED", accepted
        assert (
            accepted["classification"]["product_type"]["label"]
            == "KONUT_FINANSMANI"
        ), accepted
        assert accepted["ner"]["accepted_count"] == 6, accepted
        assert accepted["database"]["mode"] == "DRY_RUN", accepted

        review = post_intake(
            client,
            {
                **base,
                "record_key": "smoke-test-review-001",
                "page_title": "",
                "raw_text": review_text,
            },
        )
        assert review["status"] == "REVIEW", review
        assert review["ner"]["executed"] is False, review
        assert (
            review["ner"]["skip_reason"]
            == "classification_requires_review"
        ), review
        assert review["database"]["mode"] == "DRY_RUN", review

    print("ACCEPTED path: OK (6 facts)")
    print("REVIEW gate: OK (NER skipped)")
    print("Database safety: OK (DRY_RUN)")


if __name__ == "__main__":
    main()
