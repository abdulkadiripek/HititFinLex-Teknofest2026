from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
import sys
import unittest

import httpx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.chunking import attach_facts, chunk_document
from rag_v2.identity import stable_chunk_id, stable_offer_id
from rag_v2.indexer import (
    EvrenEmbeddingProvider,
    SourceDocument,
    campaign_bounds,
    extract_high_confidence_facts,
    has_ambiguous_campaign_period,
    has_classification_conflict,
    has_multiple_campaign_periods,
    prepare_document,
)
from rag_v2.settings import RagV2Settings


def settings(**overrides) -> RagV2Settings:
    values = {
        "db_host": "127.0.0.1",
        "db_port": 5432,
        "db_name": "test",
        "db_user": "test",
        "db_password": "test",
    }
    values.update(overrides)
    return RagV2Settings(**values)


class RagV2MigrationTest(unittest.TestCase):
    def test_manifest_contains_checksum_verified_rag_v2_migration(self):
        migration_dir = ROOT / "db" / "migrations"
        manifest = json.loads((migration_dir / "manifest.json").read_text("utf-8"))
        entry = next(
            item for item in manifest["migrations"] if item["version"] == "0003"
        )
        payload = (migration_dir / entry["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])

    def test_migration_declares_idempotent_rag_surfaces(self):
        sql = (ROOT / "db" / "migrations" / "0003_rag_v2.sql").read_text(
            "utf-8"
        ).lower()
        for declaration in (
            "create table if not exists rag_chunks",
            "create table if not exists rag_sessions",
            "create table if not exists rag_messages",
            "create table if not exists rag_session_state",
            "create table if not exists rag_turn_evidence",
            "create index if not exists rag_chunks_search_vector_idx",
            "create or replace function rag_v2_update_search_vector",
            "drop trigger if exists rag_chunks_search_vector_trigger",
        ):
            self.assertIn(declaration, sql)
        self.assertIn("owner_hash char(64)", sql)
        self.assertIn("token_hash char(64) not null unique", sql)
        self.assertIn("campaign_end >= campaign_start", sql)


class DeterministicChunkingTest(unittest.TestCase):
    def test_chunk_content_and_ids_are_stable(self):
        text = "# Terms\n" + " ".join(f"word-{index}" for index in range(90))
        first = chunk_document(text, page_title="Offer", max_words=32, overlap_words=8)
        second = chunk_document(text, page_title="Offer", max_words=32, overlap_words=8)
        self.assertEqual(first, second)
        first_ids = [
            stable_chunk_id("current", 42, chunk.chunk_index, chunk.content)
            for chunk in first
        ]
        second_ids = [
            stable_chunk_id("current", 42, chunk.chunk_index, chunk.content)
            for chunk in second
        ]
        self.assertEqual(first_ids, second_ids)
        self.assertGreater(len(first), 1)

    def test_fact_is_attached_only_to_chunk_containing_full_evidence(self):
        text = (
            "Terms\nMaximum term is 120 months for this offer.\n\n"
            "Fees\nThe allocation fee is listed separately."
        )
        chunks = chunk_document(text, page_title="Offer", max_words=18, overlap_words=2)
        attached = attach_facts(
            chunks,
            (
                {
                    "fact_type": "VADE_SURESI",
                    "fact_text": "120 months",
                    "evidence_text": "Maximum term is 120 months for this offer.",
                    "confidence": 0.9,
                },
                {
                    "fact_type": "FINANSMAN_TUTARI",
                    "fact_text": "missing",
                    "evidence_text": "This sentence is absent from the source.",
                    "confidence": 0.9,
                },
            ),
        )
        all_facts = [fact for chunk in attached for fact in chunk.facts]
        self.assertEqual([fact["fact_type"] for fact in all_facts], ["VADE_SURESI"])
        evidence_chunk = next(chunk for chunk in attached if chunk.facts)
        self.assertIn("Maximum term is 120 months", evidence_chunk.content)

    def test_repeated_navigation_blocks_do_not_flood_chunks(self):
        menu = "\n".join(["Altin Gunleri", "Currency Converter", "&nbsp"] * 30)
        text = menu + "\nOffer Terms\nThe maximum term is 36 months."
        chunks = chunk_document(text, page_title="Offer", max_words=32, overlap_words=4)
        self.assertLessEqual(len(chunks), 2)


class IdentityAndPolicyTest(unittest.TestCase):
    def test_multiple_title_product_signals_require_review(self):
        self.assertTrue(
            has_classification_conflict(
                page_title="Konut ve Tasit Finansmani",
                source_url="https://example.test/finansman",
                primary_product="KONUT_FINANSMANI",
            )
        )

    def test_land_title_conflicts_with_housing_label(self):
        self.assertTrue(
            has_classification_conflict(
                page_title="Arsa Finansmani",
                source_url="https://example.test/arsa-finansmani",
                primary_product="KONUT_FINANSMANI",
            )
        )

    def test_reward_and_short_campaign_range_are_extracted_from_source(self):
        facts = extract_high_confidence_facts(
            (
                "Kampanya 1-31 Temmuz 2026 tarihlerinde gecerlidir. "
                "Toplamda 2.000 TL Worldpuan kazanin!"
            ),
            effective_date=date(2026, 7, 10),
        )
        self.assertEqual(
            campaign_bounds(facts, effective_date=date(2026, 7, 10)),
            (date(2026, 7, 1), date(2026, 7, 31)),
        )
        reward = next(
            item for item in facts if item["fact_type"] == "ALISVERIS_PUANI"
        )
        self.assertEqual(reward["normalized_value"]["value"], 2000)
        self.assertEqual(reward["normalized_value"]["currency"], "TRY")

    def test_campaign_end_only_is_preserved_as_open_start_interval(self):
        facts = extract_high_confidence_facts(
            (
                "Kampanya 31 Aralik 2026 tarihine kadar gecerlidir. "
                "200 TL Worldpuan kazanin!"
            ),
            effective_date=date(2026, 8, 28),
        )
        self.assertEqual(
            campaign_bounds(facts, effective_date=date(2026, 8, 28)),
            (None, date(2026, 12, 31)),
        )

    def test_same_url_with_distinct_campaign_periods_has_distinct_offer_ids(self):
        common = {
            "bank": "bank-a",
            "product": "KART_KAMPANYASI",
            "source_url": "https://example.test/campaign?utm_source=test",
            "title": "Campaign",
            "content_boundary": "revision-a",
        }
        first = stable_offer_id(
            **common,
            campaign_start=date(2025, 1, 1),
            campaign_end=date(2025, 1, 31),
        )
        second = stable_offer_id(
            **common,
            campaign_start=date(2025, 2, 1),
            campaign_end=date(2025, 2, 28),
        )
        self.assertNotEqual(first, second)

    def test_prepare_document_preserves_single_label_and_low_confidence_policy(self):
        document = SourceDocument(
            scope="historical",
            source_id=7,
            bank_id=1,
            bank_key="bank-a",
            bank_name="Bank A",
            source_url="https://example.test/housing",
            canonical_url="https://example.test/housing",
            page_title="Housing Finance",
            raw_text="Terms\nMaximum term is 120 months for this housing offer.",
            content_hash="a" * 64,
            primary_product="KONUT_FINANSMANI",
            classification_confidence=0.60,
            classification_decision="ACCEPTED",
            classification_payload={
                "product_top3": [
                    {"label": "KONUT_FINANSMANI", "score": 0.60},
                    {"label": "TASIT_FINANSMANI", "score": 0.20},
                ]
            },
            verified=False,
            effective_date=date(2025, 1, 1),
            facts=(),
            metadata={"quality_status": "accepted"},
        )
        chunks = prepare_document(document, settings())
        self.assertTrue(chunks)
        self.assertTrue(
            all(chunk.classification_status == "required" for chunk in chunks)
        )
        self.assertTrue(
            all(
                chunk.product_types == ("KONUT_FINANSMANI",)
                for chunk in chunks
            )
        )
        self.assertIn("TASIT_FINANSMANI", chunks[0].product_scores)

    def test_campaign_identity_deduplicates_snapshots_but_not_changed_amounts(self):
        campaign_facts = (
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "1 Ocak 2025 - 31 Ocak 2025",
                "evidence_text": "Campaign: 1 Ocak 2025 - 31 Ocak 2025",
                "confidence": 0.9,
            },
            {
                "fact_type": "ODUL_TUTARI",
                "fact_text": "100 TL",
                "evidence_text": "Reward: 100 TL",
                "confidence": 0.9,
            },
        )
        base = SourceDocument(
            scope="historical",
            source_id=10,
            bank_id=1,
            bank_key="bank-a",
            bank_name="Bank A",
            source_url="https://example.test/campaign",
            canonical_url="https://example.test/campaign",
            page_title="Card Campaign",
            raw_text=(
                "Campaign\nCampaign: 1 Ocak 2025 - 31 Ocak 2025\n"
                "Reward: 100 TL"
            ),
            content_hash="a" * 64,
            primary_product="KART_KAMPANYASI",
            classification_confidence=0.9,
            classification_decision="ACCEPTED",
            classification_payload={},
            verified=False,
            effective_date=date(2025, 1, 15),
            facts=campaign_facts,
            metadata={"quality_status": "accepted"},
        )
        same_offer = replace(
            base,
            source_id=11,
            content_hash="b" * 64,
            raw_text=base.raw_text + "\nUpdated navigation",
        )
        changed_amount = replace(
            same_offer,
            source_id=12,
            facts=(
                campaign_facts[0],
                {
                    **campaign_facts[1],
                    "fact_text": "200 TL",
                    "evidence_text": "Reward: 200 TL",
                },
            ),
            raw_text=base.raw_text.replace("100 TL", "200 TL"),
        )
        first_id = prepare_document(base, settings())[0].offer_id
        same_id = prepare_document(same_offer, settings())[0].offer_id
        changed_id = prepare_document(changed_amount, settings())[0].offer_id
        self.assertEqual(first_id, same_id)
        self.assertNotEqual(first_id, changed_id)

    def test_campaign_bounds_parse_turkish_month_names(self):
        facts = (
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "15 Kasim 2025 - 15 Aralik 2025",
                "evidence_text": "Campaign runs 15 Kasim 2025 - 15 Aralik 2025.",
            },
        )
        self.assertEqual(
            campaign_bounds(facts, effective_date=None),
            (date(2025, 11, 15), date(2025, 12, 15)),
        )

    def test_yearless_campaign_range_crosses_new_year_from_snapshot(self):
        facts = (
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "12 Aralik-12 Ocak",
                "normalized_value": {"raw": "12 Aralik-12 Ocak"},
                "evidence_text": (
                    "12 Aralik-12 Ocak tarihleri arasinda kampanya gecerlidir."
                ),
            },
        )
        self.assertEqual(
            campaign_bounds(facts, effective_date=date(2022, 12, 13)),
            (date(2022, 12, 12), date(2023, 1, 12)),
        )
        self.assertFalse(
            has_ambiguous_campaign_period(
                facts,
                effective_date=date(2022, 12, 13),
            )
        )

    def test_yearless_campaign_range_uses_previous_year_for_january_snapshot(self):
        facts = (
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "12 Aralik-12 Ocak",
                "evidence_text": "12 Aralik-12 Ocak tarihleri arasinda.",
            },
        )
        self.assertEqual(
            campaign_bounds(facts, effective_date=date(2023, 1, 5)),
            (date(2022, 12, 12), date(2023, 1, 12)),
        )

    def test_single_explicit_year_anchors_the_cross_year_range(self):
        start_anchored = (
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "12 Aralik 2022 - 12 Ocak",
            },
        )
        end_anchored = (
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "12 Aralik - 12 Ocak 2023",
            },
        )
        expected = (date(2022, 12, 12), date(2023, 1, 12))
        self.assertEqual(
            campaign_bounds(start_anchored, effective_date=None),
            expected,
        )
        self.assertEqual(
            campaign_bounds(end_anchored, effective_date=None),
            expected,
        )

    def test_yearless_range_without_snapshot_is_ambiguous(self):
        facts = (
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "12 Aralik-12 Ocak",
            },
        )
        self.assertEqual(
            campaign_bounds(facts, effective_date=None),
            (None, None),
        )
        self.assertTrue(
            has_ambiguous_campaign_period(facts, effective_date=None)
        )

    def test_explicit_inverted_campaign_years_are_withheld_for_review(self):
        facts = (
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "12 Aralik 2022 - 12 Ocak 2022",
                "evidence_text": "12 Aralik 2022 - 12 Ocak 2022 arasinda.",
                "confidence": 0.99,
            },
        )
        self.assertEqual(
            campaign_bounds(facts, effective_date=date(2022, 12, 13)),
            (None, None),
        )
        self.assertTrue(
            has_ambiguous_campaign_period(
                facts,
                effective_date=date(2022, 12, 13),
            )
        )
        document = SourceDocument(
            scope="historical",
            source_id=99,
            bank_id=1,
            bank_key="bank-a",
            bank_name="Bank A",
            source_url="https://example.test/campaign",
            canonical_url="https://example.test/campaign",
            page_title="Card Campaign",
            raw_text="Campaign 12 Aralik 2022 - 12 Ocak 2022 arasinda.",
            content_hash="d" * 64,
            primary_product="KART_KAMPANYASI",
            classification_confidence=0.95,
            classification_decision="ACCEPTED",
            classification_payload={},
            verified=False,
            effective_date=date(2022, 12, 13),
            facts=facts,
            metadata={"quality_status": "accepted"},
        )
        chunks = prepare_document(document, settings())
        self.assertTrue(chunks)
        self.assertTrue(all(item.classification_status == "review" for item in chunks))
        self.assertTrue(all(item.classification_conflict for item in chunks))
        self.assertTrue(
            all(item.metadata["ambiguous_campaign_period"] for item in chunks)
        )
        self.assertTrue(all(item.campaign_start is None for item in chunks))
        self.assertTrue(all(item.campaign_end is None for item in chunks))

    def test_multiple_campaign_periods_are_withheld_for_review(self):
        facts = (
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "1 Ocak 2025 - 31 Ocak 2025",
                "evidence_text": "Campaign: 1 Ocak 2025 - 31 Ocak 2025",
                "confidence": 0.9,
            },
            {
                "fact_type": "KAMPANYA_TARIH_ARALIGI",
                "fact_text": "1 Subat 2025 - 28 Subat 2025",
                "evidence_text": "Campaign: 1 Subat 2025 - 28 Subat 2025",
                "confidence": 0.9,
            },
        )
        self.assertTrue(
            has_multiple_campaign_periods(facts, effective_date=date(2025, 1, 1))
        )
        document = SourceDocument(
            scope="historical",
            source_id=17,
            bank_id=1,
            bank_key="bank-a",
            bank_name="Bank A",
            source_url="https://example.test/campaigns",
            canonical_url="https://example.test/campaigns",
            page_title="Card Campaigns",
            raw_text=(
                "Campaign A\nCampaign: 1 Ocak 2025 - 31 Ocak 2025\n\n"
                "Campaign B\nCampaign: 1 Subat 2025 - 28 Subat 2025"
            ),
            content_hash="c" * 64,
            primary_product="KART_KAMPANYASI",
            classification_confidence=0.95,
            classification_decision="ACCEPTED",
            classification_payload={},
            verified=False,
            effective_date=date(2025, 1, 1),
            facts=facts,
            metadata={"quality_status": "accepted"},
        )
        chunks = prepare_document(document, settings())
        self.assertTrue(chunks)
        self.assertTrue(all(item.classification_conflict for item in chunks))
        self.assertTrue(all(item.classification_status == "review" for item in chunks))
        self.assertTrue(
            all(item.metadata["multiple_campaign_periods"] for item in chunks)
        )


class ProviderSafetyTest(unittest.TestCase):
    def test_evren_error_does_not_disclose_api_key(self):
        secret = "unit-test-secret-value"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": secret}, request=request)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = EvrenEmbeddingProvider(
            settings(evren_api_key=secret),
            client=client,
        )
        with self.assertRaises(RuntimeError) as raised:
            provider.embed(["test"])
        self.assertNotIn(secret, str(raised.exception))
        client.close()


if __name__ == "__main__":
    unittest.main()
