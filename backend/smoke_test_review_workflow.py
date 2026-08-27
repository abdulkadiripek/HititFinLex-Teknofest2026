from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv


API_URL = "http://127.0.0.1:8000"


def get_json(client: httpx.Client, path: str, **kwargs):
    response = client.get(path, **kwargs)
    response.raise_for_status()
    return response.json()


def main():
    load_dotenv()
    admin_api_key = os.getenv("HITITFINLEX_ADMIN_API_KEY", "").strip()
    if not admin_api_key:
        raise RuntimeError(
            "HITITFINLEX_ADMIN_API_KEY is required for this smoke test."
        )
    with httpx.Client(
        base_url=API_URL,
        timeout=60.0,
        headers={"X-API-Key": admin_api_key},
    ) as client:
        health = get_json(client, "/health")
        assert health["review_workflow"] == "human_review_v1", health
        summary = get_json(client, "/reviews/summary")
        documents = get_json(
            client,
            "/reviews/documents",
            params={"review_status": "pending", "limit": 50},
        )
        facts = get_json(
            client,
            "/reviews/facts",
            params={"review_status": "pending", "limit": 50},
        )

    assert documents["count"] == summary["document_reviews"]["pending"]
    assert facts["count"] == summary["fact_reviews"]["pending"]
    assert summary["pending_total"] == (
        documents["count"] + facts["count"]
    )
    print("Review API: OK")
    print("Pending document reviews:", documents["count"])
    print("Pending fact reviews:", facts["count"])
    print("Mutation performed: NO")


if __name__ == "__main__":
    main()
