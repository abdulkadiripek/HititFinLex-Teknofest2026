from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

from fact_context_rules import campaign_amount_role_at

PIPELINE_VERSION = "coverage_rules_v3_0"

ELIGIBLE_PRODUCT_CODES = {
    "ALISVERIS_PUANI",
    "DIGER_FINANSMAN",
    "DIGER_KAMPANYA",
    "FINANSMAN",
    "HESAP_YATIRIM",
    "IHTIYAC_FINANSMANI",
    "KART",
    "KART_KAMPANYASI",
    "KART_URUNU",
    "KATILMA_HESABI",
    "KONUT_FINANSMANI",
    "ODEME_TRANSFER",
    "ODEME_TRANSFER_HIZMETI",
    "SIGORTA_TEKAFUL",
    "SIGORTA_TEKAFUL_URUNU",
    "TASIT_FINANSMANI",
    "TICARI_FINANSMAN",
    "YATIRIM_URUNU",
    "YENI_MUSTERI",
}

CAMPAIGN_CODES = {
    "ALISVERIS_PUANI",
    "DIGER_KAMPANYA",
    "KART",
    "KART_KAMPANYASI",
    "YENI_MUSTERI",
}

FINANCE_CODES = {
    "DIGER_FINANSMAN",
    "FINANSMAN",
    "IHTIYAC_FINANSMANI",
    "KONUT_FINANSMANI",
    "TASIT_FINANSMANI",
    "TICARI_FINANSMAN",
}

INVESTMENT_CODES = {
    "HESAP_YATIRIM",
    "KATILMA_HESABI",
    "YATIRIM_URUNU",
}

INSURANCE_CODES = {
    "SIGORTA_TEKAFUL",
    "SIGORTA_TEKAFUL_URUNU",
}

PAYMENT_CODES = {
    "ODEME_TRANSFER",
    "ODEME_TRANSFER_HIZMETI",
}

MONTHS = (
    "ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|"
    "eylul|ekim|kasim|aralik"
)

DATE_RANGE_PATTERNS = (
    re.compile(
        r"\b\d{1,2}[./-]\d{1,2}[./-]20\d{2}\s*"
        r"(?:-|\u2013|\u2014|ile|ve)\s*"
        r"\d{1,2}[./-]\d{1,2}[./-]20\d{2}\b"
    ),
    re.compile(
        rf"\b\d{{1,2}}\s+(?:{MONTHS})(?:\s+20\d{{2}})?\s*"
        rf"(?:-|\u2013|\u2014|ile|ve)\s*"
        rf"\d{{1,2}}\s+(?:{MONTHS})(?:\s+20\d{{2}})?\b"
    ),
)

SINGLE_DATE_PATTERN = re.compile(
    rf"(?:\b\d{{1,2}}[./-]\d{{1,2}}[./-]20\d{{2}}\b|"
    rf"\b\d{{1,2}}\s+(?:{MONTHS})\s+20\d{{2}}\b)"
)

AMOUNT_PATTERN = re.compile(
    r"\b\d[\d. ]*(?:,\d+)?\s*(?:tl|try|turk lirasi)\b"
)
PERCENT_PATTERN = re.compile(
    r"(?:%\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*%)"
)
INSTALLMENT_PATTERN = re.compile(r"\b\d{1,3}\s*taksit\b")
DURATION_PATTERN = re.compile(
    r"\b\d{1,4}\s*(?:ay|yil|gun)(?:a|e|lik|luk)?\b"
)
SHARING_PATTERN = re.compile(r"\b\d{1,3}\s*/\s*\d{1,3}\b")


@dataclass(frozen=True)
class RuleFact:
    fact_type: str
    fact_text: str
    normalized_value: dict[str, Any] | None
    evidence_text: str
    confidence: float
    rule_name: str


def fold_text(value: str) -> str:
    translated = value.translate(str.maketrans({"\u0131": "i", "\u0130": "I"}))
    decomposed = unicodedata.normalize("NFKD", translated)
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_marks).strip().casefold()


def normalized_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def split_segments(text: str) -> list[str]:
    segments = []
    seen = set()
    for part in re.split(r"(?<=[.!?;])\s+|[\r\n]+", text):
        clean = normalized_space(part).strip(" -\t")
        folded = fold_text(clean)
        if len(clean) < 12 or len(clean) > 420 or folded in seen:
            continue
        seen.add(folded)
        segments.append(clean)
    return segments


def parse_number(value: str) -> float | int | None:
    clean = re.sub(r"[^0-9.,-]", "", value)
    if not clean or clean == "-":
        return None
    if "," in clean and "." in clean:
        clean = clean.replace(".", "").replace(",", ".")
    elif "," in clean:
        decimals = len(clean) - clean.rfind(",") - 1
        clean = clean.replace(",", "" if decimals == 3 else ".")
    elif "." in clean:
        decimals = len(clean) - clean.rfind(".") - 1
        if decimals == 3:
            clean = clean.replace(".", "")
    try:
        number = float(clean)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def exact_matches(segment: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    folded = fold_text(segment)
    matches = []
    for pattern in patterns:
        for match in pattern.finditer(folded):
            matches.append(segment[match.start() : match.end()])
    return matches


def nearby_cue(
    segment: str,
    value: str,
    cues: tuple[str, ...],
    *,
    radius: int = 64,
) -> bool:
    folded_segment = fold_text(segment)
    folded_value = fold_text(value)
    start = folded_segment.find(folded_value)
    if start < 0:
        return False
    context = folded_segment[
        max(0, start - radius) : min(
            len(folded_segment),
            start + len(folded_value) + radius,
        )
    ]
    return any(cue in context for cue in cues)


def reference_keywords(page_title: str, source_url: str) -> set[str]:
    stop_words = {
        "albaraka",
        "adil",
        "banka",
        "bankasi",
        "bireysel",
        "dunya",
        "emlak",
        "finans",
        "finansman",
        "finansmani",
        "finansmanlar",
        "hayat",
        "hesabi",
        "hesap",
        "hizmetleri",
        "kart",
        "katilim",
        "kredi",
        "kurumsal",
        "kuveyt",
        "sayfasi",
        "ticari",
        "turk",
        "turkiye",
        "urun",
        "urunleri",
        "vakif",
        "yatirim",
        "ziraat",
    }
    path_text = unquote(urlsplit(source_url or "").path).replace("-", " ")
    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            fold_text(f"{page_title} {path_text}"),
        )
        if len(token) >= 4 and token not in stop_words
    }


def extract_coverage_facts(
    text: str,
    product_code: str,
    *,
    page_title: str = "",
    source_url: str = "",
    max_facts: int = 18,
) -> list[RuleFact]:
    code = (product_code or "").strip().upper()
    if code not in ELIGIBLE_PRODUCT_CODES:
        return []

    facts: list[RuleFact] = []
    seen = set()
    per_type: dict[str, int] = {}
    keywords = reference_keywords(page_title, source_url)

    def relevant_segment(segment: str) -> bool:
        folded_segment = fold_text(segment)
        return not keywords or any(
            keyword in folded_segment for keyword in keywords
        )

    def add(
        fact_type: str,
        fact_text: str,
        evidence: str,
        confidence: float,
        rule_name: str,
        normalized_value: dict[str, Any] | None = None,
    ) -> None:
        clean_fact = normalized_space(fact_text)
        clean_evidence = normalized_space(evidence)
        if not clean_fact or clean_fact not in clean_evidence:
            return
        signature = (fact_type, fold_text(clean_fact))
        if signature in seen or per_type.get(fact_type, 0) >= 3:
            return
        seen.add(signature)
        per_type[fact_type] = per_type.get(fact_type, 0) + 1
        facts.append(
            RuleFact(
                fact_type=fact_type,
                fact_text=clean_fact,
                normalized_value=normalized_value,
                evidence_text=clean_evidence,
                confidence=confidence,
                rule_name=rule_name,
            )
        )

    title_rules: list[tuple[str, tuple[re.Pattern[str], ...], str]] = []
    if code in FINANCE_CODES and "sigorta" not in fold_text(page_title):
        title_rules.extend(
            [
                (
                    "FINANSMAN_AMACI",
                    (
                        re.compile(r"\begitim finansmani\b"),
                        re.compile(r"\bhac ve umre finansmani\b"),
                        re.compile(r"\bdayanikli tuketim finansmani\b"),
                        re.compile(r"\bdogal ?gaz donusum finansmani\b"),
                        re.compile(r"\balisveris finansmani\b"),
                        re.compile(r"\btatil(?:iniz)?(?: icin)?\b"),
                        re.compile(r"\bkira odeyen hesap\b"),
                        re.compile(r"\baidat odeyen hesap\b"),
                        re.compile(r"\bkonut finansman(?:i|lari)\b"),
                        re.compile(r"\barsa finansmani\b"),
                        re.compile(r"\bis yeri finansmani\b"),
                        re.compile(r"\b(?:arac|tasit) finansman(?:i|lari)\b"),
                        re.compile(r"\btarim(?: ve hayvancilik)? finansman(?:i|lari)\b"),
                    ),
                    "finance_purpose_from_title",
                ),
                (
                    "FINANSMAN_TURU",
                    (
                        re.compile(r"\bmikro finansman\b"),
                        re.compile(r"\bisletme finansmani\b"),
                        re.compile(r"\btedarikci finansman(?:i| sistemi)?\b"),
                        re.compile(r"\bithalat finansman(?:i|lari)\b"),
                        re.compile(r"\bihracat finansman(?:i|lari)\b"),
                        re.compile(r"\bdis ticaret finansman(?:i|lari)\b"),
                        re.compile(r"(?<!gayri )\bnakdi finansman(?: urunleri)?\b"),
                        re.compile(r"\bgayri ?nakdi finansman(?: urunleri|lar)?\b"),
                        re.compile(r"\bfinansal kiralama(?:lar)?\b"),
                        re.compile(r"\bleasing\b"),
                        re.compile(r"\bteminat mektup(?:lari)?\b"),
                    ),
                    "finance_type_from_title",
                ),
            ]
        )
    if code in INVESTMENT_CODES:
        title_rules.append(
            (
                "YATIRIM_ARACI",
                (
                    re.compile(r"\bdoviz(?: ve)?(?: kiymetli maden)? alim[ -]?satim(?:i)?\b"),
                    re.compile(r"\bkiymetli maden alim[ -]?satim(?:i)?\b"),
                    re.compile(r"\byatirim fon(?:u|lari)\b"),
                    re.compile(r"\bhisse senedi(?: islemleri)?\b"),
                    re.compile(r"\bkira sertifika(?:si|lari)\b"),
                    re.compile(r"\bsukuk\b"),
                    re.compile(r"\bforward(?: islemleri)?\b"),
                    re.compile(r"\belektronik urun senedi\b"),
                    re.compile(r"\belus\b"),
                    re.compile(r"\bkatilma hesap(?:i|lari)\b"),
                    re.compile(r"\baltin hesabi\b"),
                    re.compile(r"\bgumus hesabi\b"),
                    re.compile(r"\byatirim hesabi\b"),
                ),
                "investment_instrument_from_title",
            )
        )
    if code in {"KART", "KART_URUNU"}:
        title_rules.append(
            (
                "KART_TURU",
                (
                    re.compile(r"\bkredi karti\b"),
                    re.compile(r"\bbanka karti\b"),
                    re.compile(r"\bsanal kart\b"),
                    re.compile(r"\bticari kart(?:lar)?\b"),
                    re.compile(r"\bdebit\b"),
                    re.compile(r"\bhicaz kart\b"),
                    re.compile(r"\beflatun kart(?:lar)?\b"),
                    re.compile(r"\bbiz kart\b"),
                ),
                "card_type_from_title",
            )
        )
    if code in INSURANCE_CODES:
        title_rules.append(
            (
                "SIGORTA_TURU",
                (
                    re.compile(r"\bhayat sigorta(?:si|lari)\b"),
                    re.compile(r"\bferdi kaza(?: katilim)? sigortasi\b"),
                    re.compile(r"\bdask\b"),
                    re.compile(r"\bkonut sigortasi\b"),
                    re.compile(r"\barac sigortasi\b"),
                    re.compile(r"\byesil kart sigortasi\b"),
                    re.compile(r"\btekaful\b"),
                ),
                "insurance_type_from_title",
            )
        )
    if code in PAYMENT_CODES:
        title_rules.append(
            (
                "ODEME_HIZMETI",
                (
                    re.compile(r"\bfast\b"),
                    re.compile(r"\beft\b"),
                    re.compile(r"\bhavale\b"),
                    re.compile(r"\bswift\b"),
                    re.compile(r"\bpos(?: hizmetleri)?\b"),
                    re.compile(r"\bbkm express\b"),
                    re.compile(r"\bmail order\b"),
                    re.compile(r"\bdbs\b"),
                    re.compile(r"\bnakit yonetimi\b"),
                ),
                "payment_service_from_title",
            )
        )

    for fact_type, patterns, rule_name in title_rules:
        for value in exact_matches(page_title, patterns):
            add(
                fact_type,
                value,
                page_title,
                0.99,
                rule_name,
                {"source": "page_title"},
            )

    for segment in split_segments(text):
        folded = fold_text(segment)

        if code in CAMPAIGN_CODES:
            for value in exact_matches(segment, DATE_RANGE_PATTERNS):
                add(
                    "KAMPANYA_TARIH_ARALIGI",
                    value,
                    segment,
                    0.99,
                    "campaign_date_range",
                    {"raw": value},
                )

            if "son basvuru" in folded:
                for value in exact_matches(segment, (SINGLE_DATE_PATTERN,)):
                    add(
                        "BASVURU_SON_TARIHI",
                        value,
                        segment,
                        0.98,
                        "application_deadline",
                        {"raw": value},
                    )

            folded_segment = fold_text(segment)
            amount_matches = list(AMOUNT_PATTERN.finditer(folded_segment))
            percentages = exact_matches(segment, (PERCENT_PATTERN,))
            installments = exact_matches(segment, (INSTALLMENT_PATTERN,))
            if "indirim" in folded or "iade" in folded:
                for value in percentages:
                    add(
                        "INDIRIM_ORANI",
                        value,
                        segment,
                        0.97,
                        "discount_percent",
                        {"value": parse_number(value), "unit": "percent"},
                    )

            for amount_match in amount_matches:
                value = segment[amount_match.start() : amount_match.end()]
                role = campaign_amount_role_at(
                    folded_segment,
                    amount_match.start(),
                    amount_match.end(),
                )
                if role == "discount_amount":
                    add(
                        "INDIRIM_TUTARI",
                        value,
                        segment,
                        0.96,
                        "discount_amount",
                        {"value": parse_number(value), "currency": "TRY"},
                    )
                elif role == "reward_amount":
                    add(
                        "ODUL_TUTARI",
                        value,
                        segment,
                        0.96,
                        "reward_amount",
                        {"value": parse_number(value), "currency": "TRY"},
                    )
                elif role == "spend_threshold":
                    add(
                        "HARCAMA_ESIGI",
                        value,
                        segment,
                        0.97,
                        "spend_threshold",
                        {"value": parse_number(value), "currency": "TRY"},
                    )
            for value in installments:
                add(
                    "TAKSIT_SAYISI",
                    value,
                    segment,
                    0.97,
                    "installment_count",
                    {"value": parse_number(value), "unit": "count"},
                )

            eligibility_cue = any(
                cue in folded
                for cue in (
                    "kampanyadan yararlan",
                    "yararlanabilmek icin",
                    "faydalanabilmek icin",
                    "kampanya kosulu",
                    "kampanya sarti",
                )
            )
            eligibility_predicate = any(
                cue in folded
                for cue in (
                    "gerekm",
                    "yeter",
                    "olmal",
                    "yararlanamay",
                    "zorunlu",
                    "verilmesi",
                    "yapilmasi",
                    "kullanmaniz",
                    "gerceklestiril",
                )
            )
            if (
                eligibility_cue
                and eligibility_predicate
                and len(segment) <= 240
            ):
                add(
                    "UYGUNLUK_KOSULU",
                    segment,
                    segment,
                    0.91,
                    "campaign_eligibility",
                )

        channel_patterns = (
            re.compile(r"\bmobil sube\b"),
            re.compile(r"\bmobil uygulama(?:si|dan|mizdan)?\b"),
            re.compile(r"\binternet sube(?:si|den|mizden)?\b"),
            re.compile(r"\bweb sitesi(?:nden)?\b"),
            re.compile(r"\bonline basvuru\b"),
            re.compile(r"\bsube(?:lerimiz|miz|den|ye)?\b"),
            re.compile(r"\bcagri merkezi(?:nden)?\b"),
            re.compile(r"\bmusteri iletisim merkezi(?:nden)?\b"),
            re.compile(r"\bgoruntulu gorusme\b"),
        )
        explicit_application = "basvur" in folded
        investment_operation = (
            code in INVESTMENT_CODES | PAYMENT_CODES
            and any(
                cue in folded
                for cue in ("islem", "alim satim", "hesap acilis")
            )
            and any(
                cue in folded
                for cue in ("yapabilir", "gerceklestirebilir", "acabilirsiniz")
            )
        )
        if explicit_application or investment_operation:
            for value in exact_matches(segment, channel_patterns):
                value_end = folded.find(fold_text(value)) + len(fold_text(value))
                if "gitmeden" in folded[value_end : value_end + 20]:
                    continue
                add(
                    "BASVURU_KANALI",
                    value,
                    segment,
                    0.96,
                    "application_channel",
                    {"channel": fold_text(value)},
                )

        fee_free_cue = any(
            cue in folded
            for cue in (
                "masrafsiz",
                "ucret alinmaz",
                "masraf alinmaz",
                "herhangi bir ucret alinmamaktadir",
            )
        ) or (
            "ucretsiz" in folded
            and any(
                cue in folded
                for cue in (
                    "ucretsiz olarak kullan",
                    "islemler ucretsiz",
                    "islem ucretsiz",
                    "basvuru ucretsiz",
                    "hesap acilis ucretsiz",
                )
            )
        )
        if fee_free_cue and any(
            cue in folded
            for cue in (
                "urun",
                "finansman",
                "hesap",
                "basvuru",
                "islem",
                "kart",
                "kampanya",
                "hizmet",
            )
        ):
            cue_patterns = (
                re.compile(r"\bmasrafsiz\b"),
                re.compile(r"\bucretsiz\b"),
                re.compile(r"\bucret alinmaz\b"),
                re.compile(r"\bmasraf alinmaz\b"),
                re.compile(r"\bherhangi bir ucret alinmamaktadir\b"),
            )
            for value in exact_matches(segment, cue_patterns):
                add(
                    "MASRAF_DURUMU",
                    value,
                    segment,
                    0.95,
                    "fee_free_status",
                    {"status": "fee_free"},
                )

        if (
            ("kkdf" in folded or "bsmv" in folded)
            and any(cue in folded for cue in ("muaf", "alinmaz", "tabi degil"))
        ):
            value_patterns = (
                re.compile(r"\bkkdf\b"),
                re.compile(r"\bbsmv\b"),
            )
            for value in exact_matches(segment, value_patterns):
                add(
                    "VERGI_MUAFIYETI",
                    value,
                    segment,
                    0.98,
                    "tax_exemption",
                    {"tax": fold_text(value)},
                )

        amounts = exact_matches(segment, (AMOUNT_PATTERN,))
        percentages = exact_matches(segment, (PERCENT_PATTERN,))
        durations = exact_matches(segment, (DURATION_PATTERN,))

        if code in FINANCE_CODES:
            finance_relevant = relevant_segment(segment) or any(
                cue in folded
                for cue in (
                    "finansman tutari",
                    "finansman limiti",
                    "kar payi",
                    "vade secenegi",
                    "vade olanagi",
                    "vade imkani",
                )
            )
            if any(
                cue in folded
                for cue in ("finansman tutari", "azami finansman", "finansman limiti")
            ) and finance_relevant:
                for value in amounts:
                    add(
                        "FINANSMAN_TUTARI",
                        value,
                        segment,
                        0.97,
                        "finance_amount",
                        {"value": parse_number(value), "currency": "TRY"},
                    )
            if finance_relevant and (
                "finansman orani" in folded or "finanse edilebil" in folded
            ):
                for value in percentages:
                    add(
                        "FINANSMAN_ORANI",
                        value,
                        segment,
                        0.97,
                        "finance_ratio",
                        {"value": parse_number(value), "unit": "percent"},
                    )
            if finance_relevant and (
                "kar payi" in folded or "kar orani" in folded
            ):
                for value in percentages:
                    add(
                        "KAR_PAYI_ORANI",
                        value,
                        segment,
                        0.97,
                        "profit_rate",
                        {"value": parse_number(value), "unit": "percent"},
                    )
            if finance_relevant and re.search(
                r"\bvade(?:si|de|ye|ler|li|yle)?\b",
                folded,
            ):
                for value in durations:
                    if nearby_cue(
                        segment,
                        value,
                        ("odemesiz donem", "erteleme suresi"),
                        radius=30,
                    ):
                        continue
                    if not nearby_cue(segment, value, ("vade",), radius=36):
                        continue
                    unit = (
                        "month"
                        if "ay" in fold_text(value)
                        else "year"
                        if "yil" in fold_text(value)
                        else "day"
                    )
                    add(
                        "VADE_SURESI",
                        value,
                        segment,
                        0.96,
                        "finance_maturity",
                        {"value": parse_number(value), "unit": unit},
                    )

        if code in INVESTMENT_CODES:
            if (
                ("bakiye" in folded or "hesap acilis" in folded)
                and any(cue in folded for cue in ("minimum", "asgari", "en az"))
            ):
                for value in amounts:
                    add(
                        "MINIMUM_BAKIYE",
                        value,
                        segment,
                        0.97,
                        "minimum_balance",
                        {"value": parse_number(value), "currency": "TRY"},
                    )
            if "kar paylasim" in folded:
                for value in exact_matches(segment, (SHARING_PATTERN,)):
                    parts = [parse_number(part) for part in value.split("/")]
                    add(
                        "KAR_PAYLASIM_ORANI",
                        value,
                        segment,
                        0.98,
                        "profit_sharing_ratio",
                        {"customer": parts[0], "bank": parts[1], "unit": "percent"},
                    )
            investment_maturity = (
                relevant_segment(segment)
                and any(
                    cue in folded
                    for cue in (
                        "hesap ac",
                        "hesabi ac",
                        "katilma hesabi",
                        "vadeli hesap",
                    )
                )
            )
            if investment_maturity and re.search(
                r"\bvade(?:si|de|ye|ler|li|yle)?\b",
                folded,
            ):
                for value in durations:
                    if not nearby_cue(segment, value, ("vade",), radius=36):
                        continue
                    unit = (
                        "month"
                        if "ay" in fold_text(value)
                        else "year"
                        if "yil" in fold_text(value)
                        else "day"
                    )
                    add(
                        "VADE_SURESI",
                        value,
                        segment,
                        0.96,
                        "account_maturity",
                        {"value": parse_number(value), "unit": unit},
                    )

        if code in INVESTMENT_CODES | PAYMENT_CODES:
            operation_context = any(
                cue in folded
                for cue in ("islem", "eft", "fast", "havale", "transfer", "alim satim")
            )
            if operation_context and any(
                cue in folded
                for cue in (
                    "minimum islem",
                    "asgari islem",
                    "en az islem tutari",
                    "islem alt limiti",
                    "alt limit",
                )
            ):
                for value in amounts:
                    add(
                        "ISLEM_ALT_LIMITI",
                        value,
                        segment,
                        0.96,
                        "transaction_minimum",
                        {"value": parse_number(value), "currency": "TRY"},
                    )
            if operation_context and any(
                cue in folded
                for cue in (
                    "maksimum islem",
                    "azami islem",
                    "en fazla islem tutari",
                    "islem ust limiti",
                    "ust limit",
                )
            ):
                for value in amounts:
                    add(
                        "ISLEM_UST_LIMITI",
                        value,
                        segment,
                        0.96,
                        "transaction_maximum",
                        {"value": parse_number(value), "currency": "TRY"},
                    )

        if code in INSURANCE_CODES and (
            "sigorta primi" in folded
            or "tekaful primi" in folded
            or "sigorta ucreti" in folded
        ):
            for value in amounts:
                add(
                    "SIGORTA_UCRETI",
                    value,
                    segment,
                    0.97,
                    "insurance_fee",
                    {"value": parse_number(value), "currency": "TRY"},
                )

        if code in FINANCE_CODES:
            payment_patterns = (
                re.compile(r"\besnek odeme plani\b"),
                re.compile(r"\bodeme plani\b"),
                re.compile(r"\bartan taksit(?:li)?\b"),
                re.compile(r"\bazalan taksit(?:li)?\b"),
                re.compile(r"\bara odeme(?:li)?\b"),
                re.compile(r"\besit taksit(?:li)?\b"),
            )
            for value in exact_matches(segment, payment_patterns):
                if not relevant_segment(segment) and not any(
                    cue in folded
                    for cue in ("nakit akis", "finansman", "taksit")
                ):
                    continue
                add(
                    "ODEME_PLANI",
                    value,
                    segment,
                    0.95,
                    "payment_plan",
                    {"plan": fold_text(value)},
                )

            if "odemesiz donem" in folded:
                for value in durations:
                    if not nearby_cue(
                        segment,
                        value,
                        ("odemesiz donem",),
                        radius=30,
                    ):
                        continue
                    unit = (
                        "month"
                        if "ay" in fold_text(value)
                        else "year"
                        if "yil" in fold_text(value)
                        else "day"
                    )
                    add(
                        "ODEMESIZ_DONEM",
                        value,
                        segment,
                        0.96,
                        "grace_period",
                        {"value": parse_number(value), "unit": unit},
                    )

            if (
                relevant_segment(segment)
                and
                ("gerekli belge" in folded or "belgeler" in folded)
                and any(
                    cue in folded
                    for cue in (
                        "nufus",
                        "kimlik",
                        "gelir belgesi",
                        "bordro",
                        "tapu",
                        "ruhsat",
                        "fatura",
                    )
                )
                and len(segment) <= 240
            ):
                add(
                    "GEREKLI_BELGELER",
                    segment,
                    segment,
                    0.93,
                    "required_documents",
                )

        if code in INVESTMENT_CODES and (
            "tmsf" in folded or "tasarruf mevduati sigorta fonu" in folded
        ) and any(cue in folded for cue in ("guvence", "sigorta", "kapsam")):
            value_patterns = (
                re.compile(r"\btmsf\b"),
                re.compile(r"\btasarruf mevduati sigorta fonu\b"),
            )
            for value in exact_matches(segment, value_patterns):
                add(
                    "MEVDUAT_GUVENCESI",
                    value,
                    segment,
                    0.98,
                    "deposit_guarantee",
                )

        if code in INSURANCE_CODES and "teminat" in folded and any(
            cue in folded
            for cue in ("kapsar", "kapsam", "guvence", "vefat", "maluliyet")
        ) and len(segment) <= 240:
            add(
                "TEMINAT",
                segment,
                segment,
                0.93,
                "insurance_coverage",
            )

        if code in FINANCE_CODES | INSURANCE_CODES and any(
            cue in folded for cue in ("sigorta", "tekaful", "dask")
        ) and any(
            cue in folded
            for cue in ("zorunlu", "istege bagli", "dahil", "yaptiril")
        ) and len(segment) <= 240:
            add(
                "SIGORTA_KOSULU",
                segment,
                segment,
                0.93,
                "insurance_condition",
            )

        target_patterns = (
            re.compile(r"\bbireysel musteriler(?:imiz|e|in)?\b"),
            re.compile(r"\bticari musteriler(?:imiz|e|in)?\b"),
            re.compile(r"\bkobi(?:ler|lere|lerimiz)?\b"),
            re.compile(r"\besnaf(?:lar|lara)?\b"),
            re.compile(r"\bciftci(?:ler|lere)?\b"),
            re.compile(r"\bogrenci(?:ler|lere)?\b"),
            re.compile(r"\bemekli(?:ler|lere)?\b"),
            re.compile(r"\bkamu calisanlari\b"),
            re.compile(r"\bkadin girisimci(?:ler|lere)?\b"),
            re.compile(r"\bgirisimci(?:ler|lere)?\b"),
            re.compile(r"\bisletme(?:ler|lere)?\b"),
            re.compile(r"\bureticiler\b"),
            re.compile(r"\bihracatci(?:lar|lara)?\b"),
            re.compile(r"\barac sahipleri\b"),
            re.compile(r"\bkonut sahipleri\b"),
            re.compile(r"\bmaas musterileri\b"),
        )
        audience_cues = (
            "yararlan",
            "faydalan",
            "yonelik",
            "musterilere sun",
            "musteriler icin sun",
        )
        if relevant_segment(segment) and any(
            cue in folded for cue in audience_cues
        ):
            for value in exact_matches(segment, target_patterns):
                if not nearby_cue(
                    segment,
                    value,
                    audience_cues,
                    radius=72,
                ):
                    continue
                add(
                    "HEDEF_KITLE",
                    value,
                    segment,
                    0.92,
                    "target_audience",
                    {"audience": fold_text(value)},
                )

        if len(facts) >= max_facts:
            break

    return sorted(
        facts[:max_facts],
        key=lambda item: (-item.confidence, item.fact_type, item.fact_text),
    )
