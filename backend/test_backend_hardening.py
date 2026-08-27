from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi import FastAPI, HTTPException, Security
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api_security import (
    ADMIN_API_KEY_ENV,
    BodySizeLimitMiddleware,
    SlidingWindowRateLimitMiddleware,
    ensure_admin_api_key,
    require_admin_api_key,
)
from classifier_service import (
    canonicalize_product_label,
    canonicalize_ranked_scores,
    resolve_classification,
)
from intake_service import (
    IntakeReviewAlreadyApprovedError,
    IntakeReviewConflictError,
    approve_locked_document_review,
    clear_stale_document_data,
    content_digest,
    lock_pending_document_review,
    persist_intake,
    queue_document_review,
    require_migrated_columns,
    require_migrated_tables,
    validate_review_document_baseline,
)
from ner_service import NerBundle, predict_entities_with_metadata


class FakeBatch(dict):
    def __init__(self, word_ids, input_ids=None):
        sequence_length = len(word_ids)
        if input_ids is None:
            input_ids = [0] * sequence_length
        super().__init__(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            attention_mask=torch.ones((1, sequence_length), dtype=torch.long),
        )
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids


class FakeTokenizer:
    def __init__(self, words_per_call=3):
        self.words_per_call = words_per_call
        self.calls = 0

    def __call__(self, tokens, **_kwargs):
        self.calls += 1
        covered = min(self.words_per_call, len(tokens))
        return FakeBatch([None, *range(covered), None])


class FakeNerModel:
    config = SimpleNamespace(id2label={0: "O"})

    def __call__(self, **encoded):
        sequence_length = encoded["input_ids"].shape[1]
        return SimpleNamespace(
            logits=torch.zeros((1, sequence_length, 1), dtype=torch.float32)
        )


class BoundaryTokenizer:
    def __call__(self, tokens, *, max_length, **_kwargs):
        covered = min(len(tokens), max_length - 2)
        token_ids = []
        for token in tokens[:covered]:
            if token.startswith("w") and token[1:].isdigit():
                token_ids.append(int(token[1:]) + 1)
            else:
                token_ids.append(5001)
        return FakeBatch(
            [None, *range(covered), None],
            [0, *token_ids, 0],
        )


class BoundaryNerModel:
    config = SimpleNamespace(
        id2label={
            0: "O",
            1: "B-VADE_SURESI",
            2: "I-VADE_SURESI",
        }
    )

    def __call__(self, **encoded):
        input_ids = encoded["input_ids"][0]
        sequence_length = int(input_ids.shape[0])
        logits = torch.full(
            (1, sequence_length, 3),
            -10.0,
            dtype=torch.float32,
        )
        logits[0, :, 0] = 10.0
        for token_index, token_id in enumerate(input_ids.tolist()):
            label_id = None
            if token_id == 254 and token_index < sequence_length - 2:
                label_id = 1
            elif token_id == 255:
                label_id = 2
            if label_id is not None:
                logits[0, token_index, :] = -10.0
                logits[0, token_index, label_id] = 10.0
        return SimpleNamespace(logits=logits)


class RecordingCursor:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = list(rows or [])

    def execute(self, query, parameters=None):
        self.calls.append((" ".join(str(query).split()), parameters))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else (True,)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class RecordingConnection:
    def __init__(self, cursor):
        self.cursor_instance = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self.cursor_instance

    def transaction(self):
        return self


class QueryCaptureCursor:
    def __init__(self):
        self.query = ""
        self.parameters = ()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, parameters=None):
        self.query = " ".join(str(query).split())
        self.parameters = tuple(parameters or ())

    def fetchall(self):
        return []


class QueryCaptureConnection:
    def __init__(self):
        self.cursor_instance = QueryCaptureCursor()

    def cursor(self):
        return self.cursor_instance


class SecurityTests(unittest.TestCase):
    def test_admin_key_fails_closed_and_uses_constant_time_comparison_path(self):
        with patch.dict(os.environ, {ADMIN_API_KEY_ENV: ""}, clear=False):
            with self.assertRaises(HTTPException) as missing:
                ensure_admin_api_key(None)
            self.assertEqual(missing.exception.status_code, 503)

        valid_key = "3m9!nV7#pQ2@xL8$wR5%tK1&zF6*hC4?"
        with patch.dict(os.environ, {ADMIN_API_KEY_ENV: valid_key}, clear=False):
            with self.assertRaises(HTTPException) as invalid:
                ensure_admin_api_key("wrong")
            self.assertEqual(invalid.exception.status_code, 401)
            self.assertIsNone(ensure_admin_api_key(valid_key))

        for weak_key in (
            "CHANGE_ME_TO_A_LONG_RANDOM_VALUE",
            "change_me_local_only",
            "password",
            "a" * 64,
        ):
            with self.subTest(weak_key=weak_key):
                with patch.dict(
                    os.environ,
                    {ADMIN_API_KEY_ENV: weak_key},
                    clear=False,
                ):
                    with self.assertRaises(HTTPException) as insecure:
                        ensure_admin_api_key(weak_key)
                    self.assertEqual(insecure.exception.status_code, 503)

    def test_openapi_exposes_admin_api_key_scheme(self):
        app = FastAPI()

        @app.get("/admin", dependencies=[Security(require_admin_api_key)])
        def admin_endpoint():
            return {"ok": True}

        schema = app.openapi()
        security_scheme = schema["components"]["securitySchemes"]["AdminApiKey"]
        self.assertEqual(security_scheme["type"], "apiKey")
        self.assertEqual(security_scheme["name"], "X-API-Key")

    def test_intake_requires_key_only_when_write_is_true(self):
        from api import IntakeRequest, intake

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        base_payload = {
            "bank_key": "bank",
            "bank_name": "Bank",
            "source_url": "https://example.test/document",
            "raw_text": "This document has enough text for intake.",
        }

        with patch.dict(os.environ, {ADMIN_API_KEY_ENV: ""}, clear=False):
            with self.assertRaises(HTTPException) as dry_run:
                intake(IntakeRequest(**base_payload), request, api_key=None)
            self.assertEqual(dry_run.exception.status_code, 503)
            self.assertIn("models", str(dry_run.exception.detail).casefold())

            with self.assertRaises(HTTPException) as write:
                intake(
                    IntakeRequest(**base_payload, write=True),
                    request,
                    api_key=None,
                )
            self.assertEqual(write.exception.status_code, 503)
            self.assertIn("admin api", str(write.exception.detail).casefold())

    def test_body_size_limit_checks_actual_body(self):
        app = FastAPI()
        app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=4)

        @app.post("/body")
        async def body_endpoint():
            return {"ok": True}

        response = TestClient(app).post(
            "/body",
            content=b"12345",
            headers={"Transfer-Encoding": "chunked"},
        )
        self.assertEqual(response.status_code, 413)

    def test_rate_limit_returns_retry_after(self):
        app = FastAPI()
        app.add_middleware(
            SlidingWindowRateLimitMiddleware,
            requests_per_minute=2,
            admin_requests_per_minute=2,
        )

        @app.get("/limited")
        async def limited_endpoint():
            return {"ok": True}

        client = TestClient(app)
        self.assertEqual(client.get("/limited").status_code, 200)
        self.assertEqual(client.get("/limited").status_code, 200)
        limited = client.get("/limited")
        self.assertEqual(limited.status_code, 429)
        self.assertIn("Retry-After", limited.headers)

    def test_intake_uses_the_stricter_admin_rate_bucket(self):
        app = FastAPI()
        app.add_middleware(
            SlidingWindowRateLimitMiddleware,
            requests_per_minute=2,
            admin_requests_per_minute=1,
        )

        @app.post("/intake")
        async def intake_endpoint():
            return {"ok": True}

        @app.get("/public")
        async def public_endpoint():
            return {"ok": True}

        client = TestClient(app)
        self.assertEqual(client.post("/intake").status_code, 200)
        self.assertEqual(client.post("/intake").status_code, 429)
        self.assertEqual(client.get("/public").status_code, 200)
        self.assertEqual(client.get("/public").status_code, 200)

    def test_intake_keys_match_clean_migration_column_limits(self):
        from api import IntakeRequest

        base = {
            "bank_name": "Bank",
            "source_url": "https://example.test/document",
            "raw_text": "This document has enough text for intake.",
        }
        accepted = IntakeRequest(
            **base,
            bank_key="b" * 100,
            record_key="r" * 200,
        )
        self.assertEqual(len(accepted.bank_key), 100)
        self.assertEqual(len(accepted.record_key or ""), 200)

        for field, value in (
            ("bank_key", "b" * 101),
            ("record_key", "r" * 201),
        ):
            payload = {**base, "bank_key": "bank", field: value}
            with self.subTest(field=field), self.assertRaises(ValidationError):
                IntakeRequest(**payload)

    def test_intake_accepts_only_absolute_credential_free_http_urls(self):
        from api import IntakeRequest

        base = {
            "bank_key": "bank",
            "bank_name": "Bank",
            "raw_text": "This document has enough text for intake.",
        }
        accepted = IntakeRequest(
            **base,
            source_url="https://example.test/document?version=1",
        )
        self.assertEqual(
            accepted.source_url,
            "https://example.test/document?version=1",
        )

        for source_url in (
            "javascript:alert(1)",
            "file:///etc/passwd",
            "//example.test/document",
            "https://user:password@example.test/document",
        ):
            with self.subTest(source_url=source_url), self.assertRaises(
                ValidationError
            ):
                IntakeRequest(**base, source_url=source_url)


class NerChunkingTests(unittest.TestCase):
    def test_ner_processes_every_word_across_model_windows(self):
        tokenizer = FakeTokenizer(words_per_call=3)
        bundle = NerBundle(
            tokenizer=tokenizer,
            model=FakeNerModel(),
            device=torch.device("cpu"),
            model_dir=Path("fake-ner"),
        )
        text = " ".join(f"word{index}" for index in range(10))

        entities, metadata = predict_entities_with_metadata(text, bundle)

        self.assertEqual(entities, [])
        self.assertEqual(metadata["input_word_count"], 10)
        self.assertEqual(metadata["model_chunk_count"], 5)
        self.assertFalse(metadata["truncated"])
        self.assertEqual(tokenizer.calls, 5)

    def test_ner_keeps_an_entity_across_a_window_boundary_at_10k_limit(self):
        from api import NerRequest

        text = " ".join(f"w{index:04d}" for index in range(1600))
        text += " " + ("x" * 400)
        self.assertEqual(len(text), 10000)
        self.assertEqual(NerRequest(text=text).text, text)
        bundle = NerBundle(
            tokenizer=BoundaryTokenizer(),
            model=BoundaryNerModel(),
            device=torch.device("cpu"),
            model_dir=Path("boundary-ner"),
        )

        entities, metadata = predict_entities_with_metadata(text, bundle)

        self.assertGreater(metadata["model_chunk_count"], 1)
        self.assertFalse(metadata["truncated"])
        self.assertEqual(
            entities,
            [
                {
                    "label": "VADE_SURESI",
                    "start": text.index("w0253"),
                    "end": text.index("w0254") + len("w0254"),
                    "text": "w0253 w0254",
                    "score": 1.0,
                }
            ],
        )


class CanonicalizationTests(unittest.TestCase):
    def test_legacy_kart_label_is_canonicalized_and_deduplicated(self):
        self.assertEqual(
            canonicalize_product_label("kart"),
            "KART_KAMPANYASI",
        )
        ranked = canonicalize_ranked_scores(
            [
                {"label": "KART", "score": 0.4},
                {"label": "KART_KAMPANYASI", "score": 0.3},
                {"label": "DIGER", "score": 0.3},
            ]
        )
        self.assertEqual(ranked[0], {"label": "KART_KAMPANYASI", "score": 0.7})
        self.assertEqual(
            [item["label"] for item in ranked].count("KART_KAMPANYASI"),
            1,
        )

    def test_resolution_returns_canonical_product_label(self):
        result = resolve_classification(
            text="Kredi kart kampanyasi",
            page_title="Kredi kart kampanyasi",
            campaign={"label": "EVET", "score": 0.99},
            product={"label": "KART", "score": 0.99},
            product_ranked=[{"label": "KART", "score": 0.99}],
            threshold=0.8,
        )
        self.assertEqual(result["product_type"]["label"], "KART_KAMPANYASI")
        self.assertEqual(result["decision"], "ACCEPTED")


class IntegritySqlTests(unittest.TestCase):
    def test_update_cleanup_removes_stale_entities_and_facts(self):
        cursor = RecordingCursor(rows=[(True,)])

        clear_stale_document_data(cursor, 42)

        statements = "\n".join(query for query, _ in cursor.calls)
        self.assertIn("DELETE FROM entities", statements)
        self.assertIn("DELETE FROM passages", statements)
        self.assertIn("DELETE FROM comparison_fact_review_queue", statements)
        self.assertIn("DELETE FROM comparison_facts", statements)
        for query, parameters in cursor.calls[:4]:
            self.assertEqual(parameters, (42,), query)

    def test_review_lock_and_approval_are_scoped_to_id_key_and_hash(self):
        digest = "a" * 64
        revision = object()
        cursor = RecordingCursor(
            rows=[
                (7, "record-1", digest, revision, False, None),
                None,
                (7, "record-1", "approved"),
            ]
        )

        locked = lock_pending_document_review(
            cursor,
            review_id=7,
            record_key="record-1",
            digest=digest,
            review_updated_at=revision,
        )
        approved = approve_locked_document_review(
            cursor,
            review_id=7,
            record_key="record-1",
            digest=digest,
        )

        self.assertEqual(locked["id"], 7)
        self.assertIs(locked["updated_at"], revision)
        self.assertFalse(locked["base_document_exists"])
        self.assertIsNone(locked["base_document_hash"])
        self.assertEqual(approved["review_status"], "approved")
        self.assertEqual(
            cursor.calls[0][1],
            (7, "record-1", digest, revision),
        )
        self.assertEqual(cursor.calls[1][1], ("record-1", 7))
        self.assertEqual(cursor.calls[2][1], (7, "record-1", digest))
        self.assertIn("WHERE id = %s", cursor.calls[2][0])

    def test_stale_review_is_blocked_when_newer_review_is_active(self):
        digest = "a" * 64
        revision = object()
        cursor = RecordingCursor(
            rows=[
                (7, "record-1", digest, revision, False, None),
                (8, "b" * 64, "approved"),
            ]
        )

        with self.assertRaises(IntakeReviewConflictError) as conflict:
            lock_pending_document_review(
                cursor,
                review_id=7,
                record_key="record-1",
                digest=digest,
                review_updated_at=revision,
            )

        self.assertIn("newer review 8", str(conflict.exception))
        self.assertIn("id > %s", cursor.calls[1][0])
        self.assertIn("'pending', 'approved'", cursor.calls[1][0])

    def test_same_hash_review_refresh_invalidates_loaded_revision(self):
        revision = object()
        cursor = RecordingCursor(rows=[None])

        with self.assertRaises(IntakeReviewConflictError):
            lock_pending_document_review(
                cursor,
                review_id=7,
                record_key="record-1",
                digest="a" * 64,
                review_updated_at=revision,
            )

        self.assertIn("updated_at = %s", cursor.calls[0][0])
        self.assertEqual(cursor.calls[0][1][-1], revision)

    def test_review_queue_serializes_updates_by_record_key(self):
        cursor = RecordingCursor(
            rows=[(True,), (True, True), None, (11,)]
        )

        review_id = queue_document_review(
            cursor,
            record_key="record-1",
            bank_key="bank",
            bank_name="Bank",
            source_url="https://example.test",
            page_title="Title",
            raw_text="Document text",
            digest="a" * 64,
            classification={"decision": "REVIEW"},
            reason="classification_review",
        )

        self.assertEqual(review_id, 11)
        self.assertIn("pg_advisory_xact_lock", cursor.calls[2][0])
        self.assertEqual(cursor.calls[2][1], ("record-1",))
        self.assertIn("FOR SHARE", cursor.calls[3][0])
        self.assertIn("INSERT INTO document_intake_review_queue", cursor.calls[4][0])
        self.assertIn("base_document_exists", cursor.calls[4][0])
        self.assertNotIn(
            "base_document_exists = EXCLUDED.base_document_exists",
            cursor.calls[4][0],
        )
        self.assertIn(
            "document_intake_review_queue.base_document_exists IS NULL",
            cursor.calls[4][0],
        )
        self.assertIn(
            "ELSE document_intake_review_queue.base_document_hash",
            cursor.calls[4][0],
        )
        self.assertEqual(cursor.calls[4][1][-2:], (False, None))

    def test_review_queue_captures_existing_document_hash(self):
        accepted_text = "Existing accepted document"
        cursor = RecordingCursor(
            rows=[
                (True,),
                (True, True),
                (accepted_text,),
                (11,),
            ]
        )

        queue_document_review(
            cursor,
            record_key="record-1",
            bank_key="bank",
            bank_name="Bank",
            source_url="https://example.test",
            page_title="Title",
            raw_text="Pending candidate document",
            digest="a" * 64,
            classification={"decision": "REVIEW"},
            reason="classification_review",
        )

        self.assertEqual(
            cursor.calls[4][1][-2:],
            (True, content_digest(accepted_text)),
        )

    def test_review_queue_fails_closed_without_migrated_table(self):
        cursor = RecordingCursor(rows=[(False,)])

        with self.assertRaises(RuntimeError) as missing:
            queue_document_review(
                cursor,
                record_key="record-1",
                bank_key="bank",
                bank_name="Bank",
                source_url="https://example.test",
                page_title="Title",
                raw_text="Document text",
                digest="a" * 64,
                classification={"decision": "REVIEW"},
                reason="classification_review",
            )

        self.assertIn("public.document_intake_review_queue", str(missing.exception))
        statements = "\n".join(query for query, _ in cursor.calls)
        self.assertNotIn("CREATE TABLE", statements)
        self.assertNotIn("INSERT INTO", statements)

    def test_required_migration_check_reports_every_missing_table(self):
        cursor = RecordingCursor(rows=[(True, False, False)])

        with self.assertRaises(RuntimeError) as missing:
            require_migrated_tables(
                cursor,
                (
                    "document_intake_review_queue",
                    "document_intake_state",
                    "comparison_fact_review_queue",
                ),
            )

        message = str(missing.exception)
        self.assertNotIn("document_intake_review_queue", message)
        self.assertIn("public.document_intake_state", message)
        self.assertIn("public.comparison_fact_review_queue", message)

    def test_required_migration_columns_fail_closed(self):
        cursor = RecordingCursor(rows=[(True, False)])

        with self.assertRaises(RuntimeError) as missing:
            require_migrated_columns(
                cursor,
                "document_intake_review_queue",
                ("base_document_exists", "base_document_hash"),
            )

        self.assertNotIn("base_document_exists", str(missing.exception))
        self.assertIn(
            "public.document_intake_review_queue.base_document_hash",
            str(missing.exception),
        )

    def test_review_document_baseline_allows_only_unchanged_or_candidate(self):
        base_digest = content_digest("accepted base")
        candidate_digest = content_digest("pending candidate")

        allowed = (
            (True, base_digest, base_digest),
            (False, None, None),
            (False, None, candidate_digest),
            (True, base_digest, candidate_digest),
        )
        for base_exists, base_hash, current_digest in allowed:
            with self.subTest(
                base_exists=base_exists,
                current_digest=current_digest,
            ):
                validate_review_document_baseline(
                    review_id=7,
                    candidate_digest=candidate_digest,
                    base_document_exists=base_exists,
                    base_document_hash=base_hash,
                    current_document_digest=current_digest,
                )

        blocked = (
            (True, base_digest, content_digest("newer accepted")),
            (True, base_digest, None),
            (False, None, content_digest("newly accepted")),
            (None, None, base_digest),
        )
        for base_exists, base_hash, current_digest in blocked:
            with self.subTest(
                base_exists=base_exists,
                current_digest=current_digest,
            ):
                with self.assertRaises(IntakeReviewConflictError):
                    validate_review_document_baseline(
                        review_id=7,
                        candidate_digest=candidate_digest,
                        base_document_exists=base_exists,
                        base_document_hash=base_hash,
                        current_document_digest=current_digest,
                    )

    def test_resolved_same_hash_review_is_never_reported_as_requeued(self):
        approved_cursor = RecordingCursor(
            rows=[
                (True,),
                (True, True),
                None,
                None,
                (11, "approved"),
            ]
        )
        with self.assertRaises(IntakeReviewAlreadyApprovedError) as unchanged:
            queue_document_review(
                approved_cursor,
                record_key="record-1",
                bank_key="bank",
                bank_name="Bank",
                source_url="https://example.test",
                page_title="Title",
                raw_text="Document text",
                digest="a" * 64,
                classification={"decision": "REVIEW"},
                reason="classification_review",
            )
        self.assertEqual(unchanged.exception.review_id, 11)
        self.assertIn("already approved", str(unchanged.exception))
        self.assertIn("FOR UPDATE", approved_cursor.calls[-1][0])

        rejected_cursor = RecordingCursor(
            rows=[
                (True,),
                (True, True),
                None,
                None,
                (12, "rejected"),
            ]
        )
        with self.assertRaises(IntakeReviewConflictError) as conflict:
            queue_document_review(
                rejected_cursor,
                record_key="record-1",
                bank_key="bank",
                bank_name="Bank",
                source_url="https://example.test",
                page_title="Title",
                raw_text="Document text",
                digest="a" * 64,
                classification={"decision": "REVIEW"},
                reason="classification_review",
            )
        self.assertIn("rejected", str(conflict.exception))
        self.assertIn("not requeued", str(conflict.exception))
        self.assertIn("FOR UPDATE", rejected_cursor.calls[-1][0])

    def test_approved_same_hash_review_returns_unchanged_when_still_current(self):
        raw_text = "Approved document text"
        digest = content_digest(raw_text)
        queue_cursor = RecordingCursor(
            rows=[
                (True,),
                (True, True),
                (raw_text,),
                None,
                (11, "approved"),
            ]
        )
        current_cursor = RecordingCursor(
            rows=[(42, raw_text, "bank")]
        )
        connections = [
            RecordingConnection(queue_cursor),
            RecordingConnection(current_cursor),
        ]
        analysis = {
            "status": "REVIEW",
            "classification": {
                "review_reasons": ["classification_review"],
            },
        }

        with patch("intake_service.get_connection", side_effect=connections):
            result = persist_intake(
                record_key="record-1",
                bank_key="bank",
                bank_name="Bank",
                source_url="https://example.test",
                page_title="Title",
                raw_text=raw_text,
                digest=digest,
                analysis=analysis,
                embedding_model=object(),
                embedding_lock=object(),
                allow_update=False,
            )

        self.assertEqual(result["action"], "unchanged_skipped")
        self.assertEqual(result["document_id"], 42)
        self.assertIsNone(result["document_review_id"])

    def test_pending_review_loader_returns_updated_at_revision(self):
        from review_service import load_pending_document_review

        revision = object()
        cursor = RecordingCursor(
            rows=[
                (True,),
                (
                    7,
                    "record-1",
                    "bank",
                    "Bank",
                    "https://example.test",
                    "Title",
                    "Document text",
                    "a" * 64,
                    {"decision": "REVIEW"},
                    "classification_review",
                    revision,
                ),
            ]
        )
        connection = RecordingConnection(cursor)

        with patch("review_service.get_connection", return_value=connection):
            review = load_pending_document_review(7)

        self.assertIs(review["updated_at"], revision)
        self.assertIn("updated_at", cursor.calls[1][0])

    def test_review_resolution_forwards_loaded_revision_to_atomic_write(self):
        from api import DocumentReviewResolutionRequest, resolve_document_review

        revision = object()
        review = {
            "record_key": "record-1",
            "bank_key": "bank",
            "bank_name": "Bank",
            "source_url": "https://example.test",
            "page_title": "Title",
            "raw_text": "Document text",
            "content_hash": "a" * 64,
            "classification": {"decision": "REVIEW"},
            "updated_at": revision,
        }
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    ner_bundle=object(),
                    embedding_model=object(),
                    ner_lock=object(),
                    model_lock=object(),
                )
            )
        )
        payload = DocumentReviewResolutionRequest(
            review_id=7,
            action="approve",
            product_type="KART_KAMPANYASI",
        )

        with (
            patch("api.load_pending_document_review", return_value=review),
            patch(
                "api.analyze_reviewed_intake",
                return_value={"ner": {"candidates": []}},
            ),
            patch(
                "api.persist_intake",
                return_value={
                    "review_resolution": {
                        "id": 7,
                        "review_status": "approved",
                    }
                },
            ) as persist,
        ):
            result = resolve_document_review(payload, request)

        self.assertEqual(result["action"], "approved")
        self.assertIs(
            persist.call_args.kwargs["review_updated_at"],
            revision,
        )

    def test_stale_review_resolution_returns_http_409(self):
        from api import DocumentReviewResolutionRequest, resolve_document_review

        review = {
            "record_key": "record-1",
            "bank_key": "bank",
            "bank_name": "Bank",
            "source_url": "https://example.test",
            "page_title": "Title",
            "raw_text": "Pending candidate",
            "content_hash": "a" * 64,
            "classification": {"decision": "REVIEW"},
            "updated_at": object(),
        }
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    ner_bundle=object(),
                    embedding_model=object(),
                    ner_lock=object(),
                    model_lock=object(),
                )
            )
        )
        payload = DocumentReviewResolutionRequest(
            review_id=7,
            action="approve",
            product_type="KART_KAMPANYASI",
        )

        with (
            patch("api.load_pending_document_review", return_value=review),
            patch(
                "api.analyze_reviewed_intake",
                return_value={"ner": {"candidates": []}},
            ),
            patch(
                "api.persist_intake",
                side_effect=IntakeReviewConflictError("superseded"),
            ),
            self.assertRaises(HTTPException) as response,
        ):
            resolve_document_review(payload, request)

        self.assertEqual(response.exception.status_code, 409)
        self.assertEqual(response.exception.detail, "superseded")


class ResponseContractTests(unittest.TestCase):
    def test_unverified_search_and_fact_results_have_explicit_warnings(self):
        from api import rows_to_comparison_items, rows_to_search_results

        search = rows_to_search_results(
            [
                (
                    1,
                    "Bank",
                    "Title",
                    "https://example.test",
                    "content",
                    0.9,
                    0.2,
                    0.03,
                    False,
                )
            ]
        )[0]
        self.assertFalse(search.verified)
        self.assertIsNotNone(search.verification_warning)

        comparison = rows_to_comparison_items(
            [
                (
                    1,
                    "Bank",
                    "Title",
                    "https://example.test",
                    "KART",
                    "Kart Kampanyasi",
                    None,
                    0.9,
                    False,
                    "INDIRIM_ORANI",
                    "%10",
                    {"value": 10},
                    "ner_v4_rules_v3_1",
                    0.8,
                    "%10 indirim",
                    False,
                )
            ]
        )[0]
        self.assertEqual(comparison.campaign_type_code, "KART_KAMPANYASI")
        self.assertIsNotNone(comparison.verification_warning)
        fact = comparison.attributes["INDIRIM_ORANI"][0]
        self.assertFalse(fact.verified)
        self.assertIsNotNone(fact.verification_warning)

    def test_historical_filters_and_verification_contract_are_enforced(self):
        from api import (
            HistoricalComparisonRequest,
            HistoricalSearchRequest,
            history_comparison,
            rows_to_historical_results,
        )
        from historical_search_v28 import search_historical_database

        payload = HistoricalSearchRequest(
            query="konut finansmanı",
            has_facts=True,
            min_confidence=0.75,
        )
        self.assertTrue(payload.has_facts)
        self.assertEqual(payload.min_confidence, 0.75)

        result = rows_to_historical_results(
            [
                (
                    1,
                    42,
                    "archive-key",
                    "Bank",
                    "Title",
                    "https://example.test",
                    "https://archive.example.test",
                    None,
                    "KART",
                    "content",
                    0.9,
                    0.2,
                    0.03,
                    False,
                )
            ]
        )[0]
        self.assertEqual(result.product_type_code, "KART_KAMPANYASI")
        self.assertFalse(result.verified)
        self.assertIsNotNone(result.verification_warning)

        connection = QueryCaptureConnection()
        with patch(
            "historical_search_v28.archive_search_ready",
            return_value=True,
        ):
            rows = search_historical_database(
                connection,
                [0.1, 0.2],
                "konut",
                5,
                has_facts=True,
                min_confidence=0.75,
            )
        self.assertEqual(rows, [])
        self.assertIn("EXISTS (SELECT 1 FROM historical_facts", connection.cursor_instance.query)
        self.assertIn("COALESCE(d.classification_confidence, 0) >= %s", connection.cursor_instance.query)
        self.assertIn(0.75, connection.cursor_instance.parameters)

        raw_comparison = {
            "document_id": 42,
            "archive_key": "archive-key",
            "bank_name": "Bank",
            "page_title": "Title",
            "source_url": "https://example.test",
            "archive_url": "https://archive.example.test",
            "snapshot_date": None,
            "product_type_code": "KONUT_FINANSMANI",
            "classification_confidence": 0.9,
            "verified": False,
            "verification_warning": "document warning",
            "attributes": {
                "VADE_SURESI": [
                    {
                        "text": "120 ay",
                        "normalized_value": {"value": 120, "unit": "month"},
                        "evidence_text": "120 ay vade",
                        "source": "ner_v4",
                        "confidence": 0.9,
                        "verified": False,
                        "verification_warning": "fact warning",
                    }
                ]
            },
        }
        with patch(
            "api.fetch_historical_comparison",
            return_value=[raw_comparison],
        ):
            response = history_comparison(
                HistoricalComparisonRequest(
                    product_type_code="KONUT_FINANSMANI",
                )
            )
        self.assertEqual(len(response.warnings), 2)
        self.assertFalse(response.items[0].verified)
        self.assertFalse(
            response.items[0].attributes["VADE_SURESI"][0].verified
        )


if __name__ == "__main__":
    unittest.main()
