from __future__ import annotations

import argparse
import json

import httpx


DEFAULT_API_URL = "http://127.0.0.1:8000"


def add_common_list_arguments(parser):
    parser.add_argument(
        "--status",
        choices=("pending", "approved", "rejected"),
        default="pending",
    )
    parser.add_argument("--limit", type=int, default=50)


def parse_args():
    parser = argparse.ArgumentParser(
        description="List and resolve document and fact review queues."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("summary")
    document_list = commands.add_parser("list-documents")
    add_common_list_arguments(document_list)
    fact_list = commands.add_parser("list-facts")
    add_common_list_arguments(fact_list)

    approve_document = commands.add_parser("approve-document")
    approve_document.add_argument("review_id", type=int)
    approve_document.add_argument("--product-type", required=True)
    approve_document.add_argument("--yes", action="store_true")

    reject_document = commands.add_parser("reject-document")
    reject_document.add_argument("review_id", type=int)
    reject_document.add_argument("--yes", action="store_true")

    approve_fact = commands.add_parser("approve-fact")
    approve_fact.add_argument("review_id", type=int)
    approve_fact.add_argument("--yes", action="store_true")

    reject_fact = commands.add_parser("reject-fact")
    reject_fact.add_argument("review_id", type=int)
    reject_fact.add_argument("--yes", action="store_true")

    args = parser.parse_args()
    args.api_url = args.api_url.rstrip("/")
    if hasattr(args, "review_id") and args.review_id < 1:
        parser.error("review_id must be positive.")
    if hasattr(args, "limit") and not 1 <= args.limit <= 200:
        parser.error("--limit must be between 1 and 200.")
    if args.command in {
        "approve-document",
        "reject-document",
        "approve-fact",
        "reject-fact",
    } and not args.yes:
        parser.error("Mutation commands require --yes.")
    return args


def request_json(client: httpx.Client, method: str, path: str, **kwargs):
    response = client.request(method, path, **kwargs)
    if not response.is_success:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API {response.status_code}: {detail}")
    return response.json()


def print_summary(payload):
    documents = payload["document_reviews"]
    facts = payload["fact_reviews"]
    print("Document reviews:")
    print("  pending :", documents["pending"])
    print("  approved:", documents["approved"])
    print("  rejected:", documents["rejected"])
    print("Fact reviews:")
    print("  pending :", facts["pending"])
    print("  approved:", facts["approved"])
    print("  rejected:", facts["rejected"])
    print("Pending total:", payload["pending_total"])
    print("Product types:")
    for product_type in payload["product_type_choices"]:
        print(" ", product_type)


def print_document_reviews(payload):
    print("Document review count:", payload["count"])
    for item in payload["items"]:
        product = item["classification"]["product_type"]
        reasons = ", ".join(
            item["classification"].get("review_reasons", [])
        )
        print()
        print("ID:", item["id"])
        print("Bank:", item["bank_name"])
        print("Title:", item["page_title"])
        print("URL:", item["source_url"])
        print(
            "Prediction:",
            product["label"],
            f"({float(product['score']):.4f})",
        )
        print("Reason:", item["review_reason"], reasons)
        print("Text:", item["raw_text_preview"])


def print_fact_reviews(payload):
    print("Fact review count:", payload["count"])
    for item in payload["items"]:
        print()
        print("ID:", item["id"])
        print("Bank:", item["bank_name"])
        print("Title:", item["page_title"])
        print("Fact:", item["fact_type"], "=", item["fact_text"])
        print("Confidence:", f"{float(item['confidence']):.4f}")
        print("Reason:", item["review_reason"])
        print("Evidence:", item["evidence_text"])


def main():
    args = parse_args()
    timeout = httpx.Timeout(300.0, connect=10.0)
    with httpx.Client(base_url=args.api_url, timeout=timeout) as client:
        health = request_json(client, "GET", "/health")
        if health.get("review_workflow") != "human_review_v1":
            raise RuntimeError(
                "API 0.9.0 human review workflow is not active."
            )

        if args.command == "summary":
            print_summary(
                request_json(client, "GET", "/reviews/summary")
            )
            return
        if args.command == "list-documents":
            payload = request_json(
                client,
                "GET",
                "/reviews/documents",
                params={"review_status": args.status, "limit": args.limit},
            )
            print_document_reviews(payload)
            return
        if args.command == "list-facts":
            payload = request_json(
                client,
                "GET",
                "/reviews/facts",
                params={"review_status": args.status, "limit": args.limit},
            )
            print_fact_reviews(payload)
            return

        if args.command == "approve-document":
            payload = {
                "review_id": args.review_id,
                "action": "approve",
                "product_type": args.product_type.strip().upper(),
                "ner_threshold": 0.4,
                "review_threshold": 0.6,
            }
            result = request_json(
                client,
                "POST",
                "/reviews/documents/resolve",
                json=payload,
            )
        elif args.command == "reject-document":
            result = request_json(
                client,
                "POST",
                "/reviews/documents/resolve",
                json={
                    "review_id": args.review_id,
                    "action": "reject",
                    "product_type": None,
                },
            )
        elif args.command == "approve-fact":
            result = request_json(
                client,
                "POST",
                "/reviews/facts/resolve",
                json={"review_id": args.review_id, "action": "approve"},
            )
        else:
            result = request_json(
                client,
                "POST",
                "/reviews/facts/resolve",
                json={"review_id": args.review_id, "action": "reject"},
            )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
