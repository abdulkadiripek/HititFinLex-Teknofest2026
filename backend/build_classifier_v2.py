from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


BASE_DIR = Path(__file__).resolve().parent
LOCAL_SOURCE_ROOT = BASE_DIR / "data_extracted" / "data" / "curated_v3"
WORKSPACE_SOURCE_ROOT = BASE_DIR.parent / "data_extracted" / "data" / "curated_v3"
SOURCE_ROOT = (
    LOCAL_SOURCE_ROOT if LOCAL_SOURCE_ROOT.exists() else WORKSPACE_SOURCE_ROOT
)
CLASSIFICATION_PATH = (
    SOURCE_ROOT / "classification" / "document_classification_v3_resolved.jsonl"
)
TRAINING_DIR = SOURCE_ROOT / "training"
OUTPUT_DIR = BASE_DIR / "classifier_v2_data"

SPLIT_FILES = {
    "train": "classification_train.jsonl",
    "validation": "classification_val.jsonl",
    "test": "classification_test.jsonl",
}

FAMILY_BY_TYPE = {
    "DIGER": "DIGER",
    "DIGER_FINANSMAN": "FINANSMAN",
    "DIGER_KAMPANYA": "DIGER",
    "IHTIYAC_FINANSMANI": "FINANSMAN",
    "KART_KAMPANYASI": "KART",
    "KART_URUNU": "KART",
    "KATILMA_HESABI": "HESAP_YATIRIM",
    "KONUT_FINANSMANI": "FINANSMAN",
    "ODEME_TRANSFER_HIZMETI": "ODEME_TRANSFER",
    "SIGORTA_TEKAFUL_URUNU": "SIGORTA_TEKAFUL",
    "TASIT_FINANSMANI": "FINANSMAN",
    "TICARI_FINANSMAN": "FINANSMAN",
    "YATIRIM_URUNU": "HESAP_YATIRIM",
}

MANUAL_OVERRIDES = {
    "adil_katilim-790dd5f0a379": ("DIGER", None),
    "adil_katilim-539564082f74": ("DIGER", None),
    "adil_katilim-baf4292d0c82": ("DIGER", None),
    "adil_katilim-d14d4002927a": ("DIGER", None),
    "adil_katilim-b45752aafa88": ("DIGER", None),
    "albaraka-66ebad55959d": ("YATIRIM_URUNU", None),
    "albaraka-974c724d43d4": ("YATIRIM_URUNU", None),
    "albaraka-3a4d4c0aba2c": ("DIGER", None),
    "dunya_katilim-5ee3d820d5e7": ("DIGER", None),
    "dunya_katilim-6e3c993fa0ea": ("KART_URUNU", None),
    "dunya_katilim-b9a7361f9468": ("DIGER", None),
    "dunya_katilim-0eb52136d812": ("YATIRIM_URUNU", None),
    "hayat_finans-f99e5ff0e127": ("KART_KAMPANYASI", None),
    "hayat_finans-8b671a148aff": ("KART_KAMPANYASI", None),
    "hayat_finans-ca029f91ebcb": ("YATIRIM_URUNU", None),
    "hayat_finans-8f1420a1d1fb": ("YATIRIM_URUNU", None),
    "hayat_finans-d388d39e4ebd": ("KART_URUNU", None),
    "hayat_finans-7d08c8bac7af": ("DIGER", None),
    "hayat_finans-a479a20c1f6a": ("KART_URUNU", None),
    "hayat_finans-41f99cedd419": ("KATILMA_HESABI", None),
    "hayat_finans-9144a0c5ec74": ("DIGER", None),
    "hayat_finans-8c86030e0ba1": ("DIGER", None),
    "hayat_finans-073c05108f76": ("DIGER", None),
    "hayat_finans-ddc9363972e9": ("DIGER", None),
    "hayat_finans-3cd5ae2913cf": ("KART_URUNU", None),
    "hayat_finans-c9855e3772ef": ("DIGER", None),
    "hayat_finans-691147b6d76a": ("DIGER", None),
    "hayat_finans-41225b134b62": ("DIGER", None),
    "hayat_finans-b8d3f657cdb3": ("DIGER", None),
    "hayat_finans-ea1acfd3959f": ("YATIRIM_URUNU", None),
    "hayat_finans-09b090a5019d": ("DIGER", None),
    "hayat_finans-32044ced25eb": ("DIGER", None),
    "hayat_finans-64790d305f33": ("DIGER", None),
    "hayat_finans-51c6a4fe122d": ("DIGER", None),
    "hayat_finans-338aa37f868e": ("DIGER", None),
    "kuveyt_turk-1208d74801d9": ("KART_KAMPANYASI", "EVET"),
    "tom_bank-1168e89f91c9": ("YATIRIM_URUNU", None),
    "tom_bank-d711b543637a": ("SIGORTA_TEKAFUL_URUNU", None),
    "tom_bank-b59fb03904a7": ("DIGER", None),
    "tom_bank-9cfc57db28cf": ("KATILMA_HESABI", None),
    "tom_bank-63c06a2dc9b0": ("KATILMA_HESABI", None),
    "tom_bank-1780eb35db0c": ("YATIRIM_URUNU", None),
    "emlak_katilim-1dc4d6626254": ("KART_KAMPANYASI", None),
    "turkiye_finans-aaa57aeb4a4c": ("DIGER", None),
    "turkiye_finans-2bbe5ca21c31": ("DIGER", None),
    "turkiye_finans-608212f2f8d5": ("YATIRIM_URUNU", None),
    "turkiye_finans-3da422fd128e": ("YATIRIM_URUNU", None),
    "turkiye_finans-f6f8f0513887": ("IHTIYAC_FINANSMANI", None),
    "turkiye_finans-9fa9a7346b3a": ("KATILMA_HESABI", None),
    "turkiye_finans-13e0e8164032": ("YATIRIM_URUNU", None),
    "turkiye_finans-cfcb62ac051c": ("DIGER", None),
    "turkiye_finans-30e8242bb052": ("DIGER", None),
    "vakif_katilim-7e8348756084": ("YATIRIM_URUNU", None),
    "vakif_katilim-0b5e42a15440": ("KART_URUNU", None),
    "vakif_katilim-68cf011cdc4a": ("DIGER", None),
    "vakif_katilim-dce3ca5e5124": ("DIGER", None),
    "vakif_katilim-72cde7b7614d": ("KATILMA_HESABI", None),
    "ziraat_katilim-6f59b2c0f78f": ("KART_URUNU", None),
    "adil_katilim-c3a6ec8b8228": ("DIGER", None),
    "albaraka-4e820c2ea15e": ("YATIRIM_URUNU", None),
    "hayat_finans-20fc9be543e2": ("YATIRIM_URUNU", None),
    "hayat_finans-50a235114726": ("DIGER", None),
    "hayat_finans-babea35e70f2": ("DIGER", None),
    "turkiye_finans-dbb3c341964c": ("DIGER", None),
    "turkiye_finans-d707c62e5387": ("DIGER", None),
    "dunya_katilim-1292045905a9": ("DIGER", None),
    "hayat_finans-6dc088772ba0": ("DIGER", None),
    "kuveyt_turk-7f5f30beda0b": ("KART_KAMPANYASI", None),
    "tom_bank-d9c1ee1b14f1": ("YATIRIM_URUNU", None),
    "turkiye_finans-41467e24a90f": ("YATIRIM_URUNU", None),
    "vakif_katilim-25f15f2dfa12": ("DIGER", None),
    "ziraat_katilim-760f7d95fa99": ("DIGER", None),
    "ziraat_katilim-b728bc3c482f": ("DIGER", None),
    "ziraat_katilim-3469ceea8720": ("DIGER", None),
    "ziraat_katilim-2204690813fe": ("DIGER", None),
    "ziraat_katilim-7894ae738d31": ("DIGER", None),
    "ziraat_katilim-41fbd269d354": ("DIGER", None),
    "ziraat_katilim-af4b06e7cd79": ("DIGER", None),
    "ziraat_katilim-9dc4c8b5798e": ("DIGER", None),
    "adil_katilim-b9702a2ed763": ("DIGER", None),
    "adil_katilim-100195cd5222": ("DIGER", None),
    "hayat_finans-6a5e186a3a42": ("KATILMA_HESABI", None),
    "ziraat_katilim-3cc4281f0de5": ("DIGER", None),
    "ziraat_katilim-ab4e9200aa67": ("TICARI_FINANSMAN", None),
    "ziraat_katilim-be6ce2c8e01f": ("TICARI_FINANSMAN", None),
    "ziraat_katilim-fa5158389ad3": ("TICARI_FINANSMAN", None),
    "ziraat_katilim-a1b3ecd24ac1": ("TICARI_FINANSMAN", None),
    "vakif_katilim-e8345dfbdeb7": ("DIGER", None),
    "vakif_katilim-f2beea7f7bfb": ("ODEME_TRANSFER_HIZMETI", None),
    "vakif_katilim-da0d3581f507": ("DIGER", None),
    "vakif_katilim-4da5c16825ad": ("DIGER", None),
    "vakif_katilim-8fd5c098bbaa": ("DIGER", None),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(value: str) -> str:
    value = unquote(value)
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.replace("ı", "i")
    return re.sub(r"[^a-z0-9%]+", " ", value).strip()


def contains_any(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)


def suggest_type(meta: dict, training_row: dict) -> tuple[str | None, str, float]:
    title = normalize(meta["sayfa_basligi"])
    url = normalize(meta["kaynak_url"])
    surface = f"{title} {url}"
    text_head = normalize(str(training_row["text"])[:1800])
    campaign_surface = f"{surface} {text_head}"
    campaign = training_row["is_campaign"] == "EVET"
    old_type = training_row["product_type"]
    product_tags = set(training_row.get("product_tags", []))

    # Non-product pages must not teach the model a product class merely because
    # the body mentions several products.
    general_page = contains_any(
        surface,
        (
            "blog",
            "sikca sorulan sorular",
            "urun ve hizmet ucretleri",
            "ucret ve komisyonlar",
            "belgeler",
        ),
    )
    if general_page and not campaign:
        return "DIGER", "general_or_editorial_page", 0.98

    card_terms = (
        " kredi kart",
        " banka kart",
        " kart kampanya",
        " worldpuan",
        " bankkart",
        " paraf",
        " bonus",
        " maximum",
        " mastercard",
        " troy kart",
    )
    if campaign:
        card_campaign_terms = card_terms + (
            "kartiniz ile",
            "kartinizla",
            "kartlariniz ile",
            "kartlarinizla",
            "vade farksiz",
            "pesin fiyatina",
            "taksit firsati",
            "taksit kampanya",
        )
        if contains_any(campaign_surface, card_campaign_terms):
            return "KART_KAMPANYASI", "campaign_with_card_signal", 0.98
        if old_type == "KART_KAMPANYASI" or "KART_KAMPANYASI" in product_tags:
            return None, "card_campaign_requires_manual_confirmation", 0.0
        return "DIGER_KAMPANYA", "campaign_without_card_signal", 0.92

    if contains_any(
        surface,
        (
            " sigorta",
            " dask",
            " kasko",
            " tekaf",
            " bireysel emeklilik",
            " hayat sigortalari",
            " ferdi kaza",
        ),
    ):
        return "SIGORTA_TEKAFUL_URUNU", "insurance_signal", 0.98

    if contains_any(surface, card_terms + (" kartlar ", " sanal kart", " ek kart")):
        return "KART_URUNU", "card_product_signal", 0.97

    if contains_any(
        surface,
        (
            " konut finans",
            " konut gayrimenkul finans",
            " mortgage",
            " kentsel donusum finans",
            " arsa finans",
            " is yeri finans",
        ),
    ):
        return "KONUT_FINANSMANI", "housing_finance_signal", 0.97

    if contains_any(
        surface,
        (
            " tasit finans",
            " arac finans",
            " otomobil finans",
            " motosiklet finans",
            " togg finans",
            " ticari plaka finans",
        ),
    ):
        return "TASIT_FINANSMANI", "vehicle_finance_signal", 0.97

    if contains_any(
        surface,
        (
            " ihtiyac finans",
            " egitim finans",
            " umre finans",
            " hac finans",
            " tuketim finans",
            " alisveris kredi",
            " alisveris finans",
            " hazir limit",
            " jet finansman",
            " fatura odeyen hesap",
            " kira odeyen hesap",
            " aidat odeyen hesap",
        ),
    ):
        return "IHTIYAC_FINANSMANI", "personal_finance_signal", 0.96

    if "cari hesap" in surface:
        return "DIGER", "current_account_without_specific_class", 0.92

    if contains_any(
        surface,
        (
            " odemeler ",
            " odeme hizmet",
            " para transfer",
            " moneygram",
            " western union",
            " swift",
            " eft ",
            " havale",
            " fast ",
            " fatura odeme",
            " para yukleme",
            " para gonderme",
            " pos urun",
            " sanal pos",
            " mobilde pos",
            " mail order",
            " yazar kasa pos",
            " gprs pos",
            " adsl pos",
        ),
    ):
        return "ODEME_TRANSFER_HIZMETI", "payment_transfer_signal", 0.96

    if contains_any(
        surface,
        (
            " katilma hesap",
            " kar payi odemeli hesap",
            " paylasim oranli hesap",
            " donusumlu hesap",
        ),
    ):
        return "KATILMA_HESABI", "participation_account_signal", 0.97

    if contains_any(
        surface,
        (
            " yatirim urun",
            " yatirim hesab",
            " yatirim fon",
            " kira sertifika",
            " sukuk",
            " hisse sened",
            " pay senet",
            " kiymetli maden",
            " altin hesab",
            " doviz hesab",
            " doviz alim",
            " hizli doviz",
            " hazine urun",
            " sermaye piyasa",
            " fonlar ",
            " forward",
        ),
    ):
        return "YATIRIM_URUNU", "investment_signal", 0.96

    commercial_path = contains_any(
        surface,
        (
            " kobi ",
            " ticari ",
            " kurumsal ",
            " isim icin ",
            " tarim ",
            " isletme finans",
            " tedarikci finans",
            " dis ticaret finans",
            " nakdi finans",
            " gayri nakdi finans",
            " leasing",
            " finansal kiralama",
            " teminat mekt",
            " akreditif",
        ),
    )
    if commercial_path:
        return "TICARI_FINANSMAN", "commercial_finance_signal", 0.95

    if contains_any(surface, (" finansman", " kredi", " finansmanlar")):
        return "DIGER_FINANSMAN", "generic_finance_signal", 0.86

    generic_title = contains_any(
        surface,
        (
            " ana sayfa",
            " urun ve hizmetler",
            " katilim bankaciligi",
            " hakkimizda",
            " iletisim",
            " hesaplar ",
        ),
    )
    if generic_title:
        return "DIGER", "generic_page_signal", 0.92

    return None, "ambiguous_metadata", 0.0


def suggest_type_v2(meta: dict, row: dict) -> tuple[str | None, str, float]:
    title = normalize(meta["sayfa_basligi"])
    url = normalize(meta["kaynak_url"])
    campaign = row["is_campaign"] == "EVET"
    old_type = row["product_type"]

    editorial_url = contains_any(url, (" blog ", " finansal kilavuz ", " politikalarimiz "))
    non_product_title = contains_any(
        title,
        (
            "sikca sorulan sorular",
            "urun ve hizmet ucretleri",
            "ucret ve komisyonlar",
            "site haritasi",
        ),
    )
    if (editorial_url or non_product_title) and not campaign:
        return "DIGER", "editorial_or_reference_page", 0.98

    if campaign:
        if old_type == "KART_KAMPANYASI":
            return "KART_KAMPANYASI", "confirmed_existing_card_campaign", 0.96
        explicit_card = contains_any(
            f"{title} {url}",
            (
                "kredi kart",
                "banka kart",
                "mastercard",
                "worldpuan",
                "bankkart",
                "paraf",
                "business kart",
                "debit kart",
            ),
        )
        if explicit_card:
            return "KART_KAMPANYASI", "explicit_card_campaign", 0.98
        return "DIGER_KAMPANYA", "non_card_campaign", 0.94

    surface = f"{title} {url}"
    if contains_any(
        surface,
        ("sigorta", "dask", "kasko", "tekaf", "ferdi kaza", "hayat sigort"),
    ):
        return "SIGORTA_TEKAFUL_URUNU", "insurance_product", 0.98

    if contains_any(
        surface,
        (
            "kredi kart",
            "banka kart",
            "kartlar",
            "sanal kart",
            "ek kart",
            "business kart",
            "ferah kart",
            "hadi kart",
            "biz kart",
        ),
    ):
        return "KART_URUNU", "card_product", 0.97

    if contains_any(
        surface,
        ("konut finans", "konut gayrimenkul", "mortgage", "arsa finans", "is yeri finans", "kentsel donusum finans"),
    ):
        return "KONUT_FINANSMANI", "housing_finance", 0.97

    if contains_any(
        surface,
        ("tasit finans", "arac finans", "motosiklet finans", "togg finans", "ticari plaka finans"),
    ):
        return "TASIT_FINANSMANI", "vehicle_finance", 0.97

    if contains_any(
        surface,
        (
            "ihtiyac finans",
            "egitim finans",
            "umre finans",
            "hac finans",
            "tuketim finans",
            "alisveris kredi",
            "alisveris finans",
            "hazir limit",
            "jet finans",
            "fatura odeyen hesap",
            "kira odeyen hesap",
            "aidat odeyen hesap",
        ),
    ):
        return "IHTIYAC_FINANSMANI", "personal_finance", 0.97
    if " ihtiyac " in f" {url} ":
        return "IHTIYAC_FINANSMANI", "personal_finance_url", 0.96

    if contains_any(
        surface,
        (
            "para transfer",
            "moneygram",
            "western union",
            "swift",
            "odemeler",
            "fatura odeme",
            "para yukleme",
            "para gonderme",
            "fast fon",
            "pos urun",
            "sanal pos",
            "mobilde pos",
            "mail order",
            "yazar kasa pos",
            "gprs pos",
            "adsl pos",
        ),
    ):
        return "ODEME_TRANSFER_HIZMETI", "payment_or_transfer", 0.97

    if contains_any(
        surface,
        (
            "katilma hesap",
            "kar payi odemeli hesap",
            "kar paylasim oran",
            "gunluk hesap",
            "vadeli hesap",
            "avantajli hesap",
        ),
    ):
        return "KATILMA_HESABI", "participation_account", 0.96

    investment_path = contains_any(url, (" yatirim ", " altin ", " gumus ", " platin ", " paladyum "))
    investment_title = contains_any(
        title,
        (
            "yatirim",
            "hazine urun",
            "sermaye piyasa",
            "yatirim fon",
            "kira sertifika",
            "sukuk",
            "hisse sened",
            "pay senet",
            "kiymetli maden",
            "altin hesap",
            "gumus hesap",
            "platin hesap",
            "paladyum hesap",
            "doviz alim",
            "forward",
            "tradingview",
        ),
    )
    if investment_path or investment_title:
        return "YATIRIM_URUNU", "investment_product", 0.96

    if contains_any(
        surface,
        (
            " kobi ",
            " ticari ",
            " kurumsal ",
            " isim icin ",
            " tarim ",
            "isletme finans",
            "tedarikci finans",
            "dis ticaret finans",
            "nakdi finans",
            "gayri nakdi finans",
            "leasing",
            "finansal kiralama",
            "teminat mekt",
            "akreditif",
        ),
    ):
        return "TICARI_FINANSMAN", "commercial_finance", 0.95

    if "cari hesap" in surface:
        return "DIGER", "current_account", 0.92
    if contains_any(surface, ("finansman", "kredi")):
        return "DIGER_FINANSMAN", "other_finance", 0.88

    exact_general_titles = {
        "anasayfa adil katilim",
        "dunya katilim",
        "urunlerimiz",
        "hesaplar dunya katilim",
        "hesaplar hayat finans",
        "hesaplar bireysel",
        "bize ulasin",
    }
    if title in exact_general_titles:
        return "DIGER", "general_page", 0.94
    return None, "manual_semantic_review_required", 0.0


def build_augmented_train(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    seen_texts: set[str] = set()

    def append_variant(source: dict, text: str, suffix: str, source_name: str) -> None:
        cleaned = re.sub(r"\s+", " ", text).strip()
        normalized = normalize(cleaned)
        if not cleaned or normalized in seen_texts:
            return
        seen_texts.add(normalized)
        variant = dict(source)
        variant["source_document_id"] = source["id"]
        variant["id"] = f"{source['id']}::{suffix}"
        variant["text"] = cleaned
        variant["augmentation_source"] = source_name
        output.append(variant)

    for row in rows:
        append_variant(row, row["text"], "full", "original_full_text")
        words = re.sub(r"\s+", " ", row["text"]).strip().split()
        title = row.get("title", "").strip()
        intro = " ".join(words[:140])
        append_variant(
            row,
            f"{title}. {intro}" if title else intro,
            "intro",
            "derived_intro_window",
        )

    target_per_rare_class = 50
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["product_type"], []).append(row)

    counts = Counter(row["product_type"] for row in output)
    for label, label_rows in sorted(by_class.items()):
        if counts[label] >= target_per_rare_class:
            continue
        round_index = 0
        while counts[label] < target_per_rare_class and round_index < 8:
            added_in_round = 0
            for row in label_rows:
                words = re.sub(r"\s+", " ", row["text"]).strip().split()
                if not words:
                    continue
                start = (round_index + 1) * 70
                if start >= len(words):
                    start = max(0, len(words) - 140)
                snippet = " ".join(words[start : start + 140])
                title = row.get("title", "").strip()
                before = len(output)
                append_variant(
                    row,
                    f"{title}. {snippet}" if title else snippet,
                    f"window{round_index + 1}",
                    "derived_train_window",
                )
                if len(output) > before:
                    counts[label] += 1
                    added_in_round += 1
                if counts[label] >= target_per_rare_class:
                    break
            if added_in_round == 0:
                break
            round_index += 1
    return output


def main() -> None:
    metadata = {row["kayit_id"]: row for row in read_jsonl(CLASSIFICATION_PATH)}
    records: list[dict] = []
    accepted_by_split: dict[str, list[dict]] = {split: [] for split in SPLIT_FILES}

    for split, filename in SPLIT_FILES.items():
        for row in read_jsonl(TRAINING_DIR / filename):
            meta = metadata[row["id"]]
            manual = MANUAL_OVERRIDES.get(row["id"])
            manual_campaign = None
            if manual is not None:
                suggested, manual_campaign = manual
                reason = "manual_semantic_resolution"
                confidence = 0.99
            else:
                suggested, reason, confidence = suggest_type_v2(meta, row)
            old_type = row["product_type"]
            if suggested is None:
                status = "REVIEW"
                final_type = old_type
            elif manual is not None:
                status = "MANUAL"
                final_type = suggested
            else:
                status = "ACCEPT" if suggested == old_type else "CORRECT"
                final_type = suggested

            audit = {
                "id": row["id"],
                "split": split,
                "bank": meta["banka_adi"],
                "title": meta["sayfa_basligi"],
                "url": meta["kaynak_url"],
                "old_is_campaign": row["is_campaign"],
                "is_campaign": manual_campaign or row["is_campaign"],
                "old_product_type": old_type,
                "suggested_product_type": suggested or "",
                "final_product_type": final_type,
                "status": status,
                "confidence": confidence,
                "reason": reason,
            }
            records.append(audit)

            if status != "REVIEW":
                clean_row = dict(row)
                if manual_campaign is not None:
                    clean_row["is_campaign"] = manual_campaign
                clean_row["product_type"] = final_type
                clean_row["product_family"] = FAMILY_BY_TYPE[final_type]
                clean_row["title"] = meta["sayfa_basligi"]
                clean_row["source_url"] = meta["kaynak_url"]
                clean_row["label_source"] = "classifier_v2_metadata_rules"
                clean_row["label_confidence"] = confidence
                accepted_by_split[split].append(clean_row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with (OUTPUT_DIR / "classification_label_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    for split, rows in accepted_by_split.items():
        with (OUTPUT_DIR / f"classification_{split}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    augmented_train = build_augmented_train(accepted_by_split["train"])
    with (OUTPUT_DIR / "classification_train_augmented.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in augmented_train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total": len(records),
        "status": dict(Counter(row["status"] for row in records)),
        "changes": dict(
            Counter(
                f"{row['old_product_type']} -> {row['final_product_type']}"
                for row in records
                if row["status"] == "CORRECT"
            )
        ),
        "split_sizes": {
            split: len(rows) for split, rows in accepted_by_split.items()
        },
        "augmented_train_size": len(augmented_train),
        "augmented_train_distribution": dict(
            Counter(row["product_type"] for row in augmented_train)
        ),
        "changed_product_type_total": sum(
            row["old_product_type"] != row["final_product_type"]
            for row in records
        ),
        "changed_campaign_total": sum(
            row["old_is_campaign"] != row["is_campaign"]
            for row in records
        ),
        "class_distribution": {
            split: dict(Counter(row["product_type"] for row in rows))
            for split, rows in accepted_by_split.items()
        },
    }
    with (OUTPUT_DIR / "classification_v2_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
