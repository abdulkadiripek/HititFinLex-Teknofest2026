from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Any

from .identity import normalize_text
from .models import OfferReference, QueryRoute, SessionState


BANKS: dict[str, tuple[str, ...]] = {
    "Adil Katilim": ("adil katilim", "adil bank"),
    "Albaraka Turk": ("albaraka turk", "albaraka"),
    "Dunya Katilim": ("dunya katilim", "dunya bank"),
    "Hayat Finans": ("hayat finans", "hayat katilim"),
    "Kuveyt Turk": ("kuveyt turk", "kuveyt katilim"),
    "T.O.M. Katilim": ("t.o.m. katilim", "tom katilim", "tom bank"),
    "Emlak Katilim": ("turkiye emlak katilim", "emlak katilim"),
    "Turkiye Finans": ("turkiye finans",),
    "Vakif Katilim": ("vakif katilim",),
    "Ziraat Katilim": ("ziraat katilim",),
}

BANK_KEYS: dict[str, str] = {
    "Adil Katilim": "adil_katilim",
    "Albaraka Turk": "albaraka",
    "Dunya Katilim": "dunya_katilim",
    "Hayat Finans": "hayat_finans",
    "Kuveyt Turk": "kuveyt_turk",
    "T.O.M. Katilim": "tom_bank",
    "Emlak Katilim": "emlak_katilim",
    "Turkiye Finans": "turkiye_finans",
    "Vakif Katilim": "vakif_katilim",
    "Ziraat Katilim": "ziraat_katilim",
}

PRODUCTS: dict[str, tuple[str, ...]] = {
    "KONUT_FINANSMANI": ("konut finansmani", "ev finansmani", "konut"),
    "TASIT_FINANSMANI": ("tasit finansmani", "arac finansmani", "tasit"),
    "IHTIYAC_FINANSMANI": (
        "ihtiyac finansmani",
        "bireysel finansman",
        "ihtiyac",
    ),
    "TICARI_FINANSMAN": ("ticari finansman", "kobi finansmani", "kobi"),
    "KATILMA_HESABI": ("katilma hesabi", "katilim hesabi"),
    "CARI_HESAP": ("cari hesap", "vadesiz hesap"),
    "KART": ("kredi karti", "banka karti", "kart"),
    "KART_KAMPANYASI": ("kart kampanyasi", "kart kampanyalari"),
    "MOBIL_UYGULAMA_KAMPANYASI": (
        "mobil uygulama kampanyasi",
        "mobil kampanya",
    ),
    "YATIRIM_URUNU": ("yatirim urunu", "yatirim", "altin hesabi"),
    "ODEME_TRANSFER": ("odeme transfer", "para transferi", "havale", "eft"),
    "SIGORTA_TEKAFUL": ("tekaful", "sigorta"),
}

FIELDS: dict[str, tuple[str, ...]] = {
    "amount": ("tutar", "miktar", "limit", "azami"),
    "rate": ("oran", "kar payi", "kar orani", "yuzde"),
    "maturity": ("vade", "kac ay", "vade suresi"),
    "fee": ("ucret", "masraf", "tahsis", "aidat"),
    "reward": ("odul", "puan", "indirim", "iade"),
    "spending_threshold": ("harcama esigi", "harcama kosulu", "harcama"),
    "campaign_date": ("kampanya tarihi", "son tarih", "tarih araligi"),
}

ORDINALS = {
    "birincisi": 0,
    "ilki": 0,
    "ilk teklif": 0,
    "ikincisi": 1,
    "ikinci teklif": 1,
    "ucuncusu": 2,
    "ucuncunun": 2,
    "ucuncu teklif": 2,
}

FOLLOW_UP_MARKERS = (
    "peki",
    "onun",
    "bunun",
    "bu teklif",
    "ayni banka",
    "birincisi",
    "ikincisi",
    "ucuncusu",
    "ucuncunun",
)

TOPIC_CHANGE_MARKERS = ("simdi", "gecelim", "konuyu degistir", "artik")

CONTEXT_REFERENCE_MARKERS = (
    "bankanin",
    "dogru mu",
    "hayir",
    "icin de",
    "olanlari",
    "onceki cevap",
    "tum donem",
)

OUT_OF_DOMAIN_TERMS = (
    "bitcoin",
    "borsa tahmini",
    "doviz tahmini",
    "hava durumu",
    "kripto",
    "mac sonucu",
)

CASUAL_QUERY_TERMS = (
    "merhaba",
    "selam",
    "gunaydin",
    "iyi aksamlar",
    "iyi geceler",
    "nasilsin",
    "tesekkur",
    "sag ol",
    "adin ne",
    "sen kimsin",
    "ne yapabilirsin",
    "fikra",
    "sohbet edelim",
    "samimi",
)

FINANCIAL_DATA_TERMS = (
    "banka",
    "bankalar",
    "kampanya",
    "teklif",
    "urunleri",
    "finansman tutar",
    "finansman oran",
    "kar payi",
    "vade",
    "puan",
    "worldpuan",
    "altin puan",
    "harcama esigi",
    "masraf",
    "ucret",
    "aidat",
)

SOFT_CONTEXT_MARKERS = (
    "biraz daha",
    "daha detayli",
    "daha ayrintili",
    "devam et",
    "detaylandir",
    "ayrintilandir",
    "aciklar misin",
    "neden",
    "hangisi",
    "ozetle",
)

SEMANTIC_STOP_WORDS = {
    "acaba",
    "bana",
    "bunun",
    "icin",
    "kadar",
    "mi",
    "midir",
    "misiniz",
    "mu",
    "mudur",
    "ne",
    "nedir",
    "nelerdir",
    "onun",
    "peki",
}

SEMANTIC_GENERIC_WORDS = {
    "anlat",
    "arasinda",
    "ayni",
    "banka",
    "bankanin",
    "belirt",
    "bilgi",
    "bugun",
    "bu",
    "cevap",
    "cevabinda",
    "dedin",
    "demistin",
    "dogru",
    "dort",
    "dusuk",
    "en",
    "fazla",
    "gecmis",
    "gore",
    "goster",
    "guncel",
    "hangi",
    "hakkinda",
    "iki",
    "ilk",
    "karsilastir",
    "kisa",
    "listele",
    "onceki",
    "sirala",
    "soyle",
    "teklif",
    "tum",
    "uc",
    "uzun",
    "yuksek",
}

FIELD_LABELS = {
    "amount": "tutari",
    "rate": "orani",
    "maturity": "vade suresi",
    "fee": "ucreti",
    "reward": "odulu",
    "spending_threshold": "harcama esigi",
    "campaign_date": "kampanya tarihi",
}

_TOKEN_SUFFIXES = {
    "a",
    "da",
    "daki",
    "dan",
    "de",
    "deki",
    "den",
    "e",
    "i",
    "in",
    "inda",
    "indan",
    "inde",
    "inden",
    "ini",
    "inin",
    "lar",
    "larda",
    "lardan",
    "lari",
    "larin",
    "le",
    "ler",
    "lerde",
    "lerden",
    "leri",
    "lerin",
    "na",
    "nda",
    "ndan",
    "ne",
    "nde",
    "nden",
    "ni",
    "nin",
    "nu",
    "nun",
    "si",
    "sini",
    "sinin",
    "su",
    "sunu",
    "sunun",
    "ta",
    "taki",
    "tan",
    "te",
    "teki",
    "ten",
    "u",
    "un",
    "unda",
    "undan",
    "unde",
    "unden",
    "unu",
    "unun",
    "ya",
    "ye",
    "yi",
    "yu",
}

_MEASUREMENT_QUALIFIER_PATTERN = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*"
    r"(?:ay|gun|m2|metrekare|tl|try|usd|eur|yil)(?!\w)"
)

_CALCULATION_PATTERNS = (
    re.compile(r"(?<!\w)hesapla[a-z]*(?!\w)"),
    re.compile(r"(?<!\w)toplam(?:i|ini|lari|larini)?(?!\w)"),
    re.compile(
        r"(?<!\w)(?:aradaki\s+)?fark(?:i|ini)?\s+"
        r"(?:kac|ne\s+kadar|hesapla[a-z]*)(?!\w)"
    ),
    re.compile(
        r"(?<!\w)(?:getiri|kazanc)(?:si|sini)?\s+"
        r"(?:kac|ne\s+kadar)(?!\w)"
    ),
)

MONTHS = {
    "ocak": 1,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "agustos": 8,
    "eylul": 9,
    "ekim": 10,
    "kasim": 11,
    "aralik": 12,
}

PRODUCT_LABELS = {
    "KONUT_FINANSMANI": "konut finansmani",
    "TASIT_FINANSMANI": "tasit finansmani",
    "IHTIYAC_FINANSMANI": "ihtiyac finansmani",
    "TICARI_FINANSMAN": "ticari finansman",
    "KATILMA_HESABI": "katilma hesabi",
    "CARI_HESAP": "cari hesap",
    "KART": "kredi karti",
    "KART_KAMPANYASI": "kart kampanyasi",
    "MOBIL_UYGULAMA_KAMPANYASI": "mobil uygulama kampanyasi",
    "YATIRIM_URUNU": "yatirim urunu",
    "ODEME_TRANSFER": "odeme transferi",
    "SIGORTA_TEKAFUL": "sigorta tekaful",
}

_TEXTUAL_DATE_PATTERN = re.compile(
    rf"(?<!\d)(\d{{1,2}})\s+({'|'.join(MONTHS)})(?:\s+((?:19|20)\d{{2}}))?(?!\d)"
)


def bank_keys(names: list[str]) -> list[str]:
    output: list[str] = []
    for name in names:
        if name in BANK_KEYS:
            output.append(BANK_KEYS[name])
            continue
        normalized = normalize_text(name)
        for display, aliases in BANKS.items():
            if any(normalize_text(alias) in normalized for alias in aliases):
                output.append(BANK_KEYS[display])
                break
    return list(dict.fromkeys(output))


def _matches(text: str, aliases: tuple[str, ...]) -> bool:
    suffix = (
        r"(?:lar|ler|lari|leri|larin|lerin|inin|unun|ini|unu|nin|nun|"
        r"si|su|yi|yu|da|de|dan|den|na|ne|ni|nu|i|u|a|e)?"
    )
    return any(
        re.search(
            rf"(?<!\w){re.escape(normalize_text(alias))}{suffix}(?!\w)",
            text,
        )
        for alias in aliases
    )


def _extract_banks(text: str) -> list[str]:
    suffixes = (
        "inin",
        "unun",
        "in",
        "nin",
        "un",
        "nun",
        "dan",
        "den",
        "yi",
        "yu",
        "ya",
        "ye",
        "da",
        "de",
        "la",
        "le",
        "i",
        "u",
        "a",
        "e",
    )
    ending = "|".join(suffixes)
    output: list[str] = []
    for name, aliases in BANKS.items():
        if any(
            re.search(
                rf"(?<!\w){re.escape(normalize_text(alias))}(?:['’]?(?:{ending}))?(?!\w)",
                text,
            )
            for alias in aliases
        ):
            output.append(name)
    return output


def _extract_products(text: str) -> list[str]:
    products: list[str] = []
    for code, aliases in PRODUCTS.items():
        if _matches(text, aliases):
            products.append(code)
    if "MOBIL_UYGULAMA_KAMPANYASI" in products:
        products = [
            item
            for item in products
            if item not in {"KART", "KART_KAMPANYASI"}
        ]
    if "KART_KAMPANYASI" in products and "KART" in products:
        products.remove("KART")
    return products


def _extract_fields(text: str) -> list[str]:
    return [
        name
        for name, aliases in FIELDS.items()
        if any(
            re.search(rf"(?<!\w){re.escape(normalize_text(alias))}", text)
            for alias in aliases
        )
    ]


def _extract_year(text: str) -> int | None:
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    return int(match.group(1)) if match else None


def _extract_date_range(text: str) -> tuple[date | None, date | None]:
    year_range = re.search(
        r"(?<!\d)((?:19|20)\d{2})\s*(?:-|ile)\s*((?:19|20)\d{2})(?!\d)",
        text,
    )
    if year_range:
        first_year = int(year_range.group(1))
        second_year = int(year_range.group(2))
        if first_year <= second_year:
            return date(first_year, 1, 1), date(second_year, 12, 31)

    cross_year_month_range = re.search(
        rf"(?<!\w)({'|'.join(MONTHS)})\s+((?:19|20)\d{{2}})\s*"
        rf"(?:-|ile)\s*({'|'.join(MONTHS)})\s+((?:19|20)\d{{2}})(?!\d)",
        text,
    )
    if cross_year_month_range:
        start_year = int(cross_year_month_range.group(2))
        end_year = int(cross_year_month_range.group(4))
        start_month = MONTHS[cross_year_month_range.group(1)]
        end_month = MONTHS[cross_year_month_range.group(3)]
        start = date(start_year, start_month, 1)
        end = date(
            end_year,
            end_month,
            calendar.monthrange(end_year, end_month)[1],
        )
        if start <= end:
            return start, end

    month_names = "|".join(MONTHS)
    month_range = re.search(
        rf"(?<!\w)({month_names})\s*(?:-|ile)\s*({month_names})\s+((?:19|20)\d{{2}})(?!\d)",
        text,
    )
    if month_range:
        start_month = MONTHS[month_range.group(1)]
        end_month = MONTHS[month_range.group(2)]
        year = int(month_range.group(3))
        if start_month <= end_month:
            return (
                date(year, start_month, 1),
                date(year, end_month, calendar.monthrange(year, end_month)[1]),
            )
    values: list[date] = []
    for year, month, day in re.findall(
        r"(?<!\d)((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", text
    ):
        try:
            values.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    for day, month, year in re.findall(
        r"(?<!\d)(\d{1,2})[./](\d{1,2})[./]((?:19|20)\d{2})(?!\d)", text
    ):
        try:
            candidate = date(int(year), int(month), int(day))
        except ValueError:
            continue
        if candidate not in values:
            values.append(candidate)
    textual_matches = list(_TEXTUAL_DATE_PATTERN.finditer(text))
    textual_years = [
        int(match.group(3)) if match.group(3) else None
        for match in textual_matches
    ]
    for index, parsed_year in enumerate(textual_years):
        if parsed_year is not None:
            continue
        later = next(
            (
                item
                for item in textual_years[index + 1 :]
                if item is not None
            ),
            None,
        )
        earlier = next(
            (
                item
                for item in reversed(textual_years[:index])
                if item is not None
            ),
            None,
        )
        textual_years[index] = later or earlier
    for match, parsed_year in zip(
        textual_matches,
        textual_years,
        strict=True,
    ):
        if parsed_year is None:
            continue
        try:
            candidate = date(
                parsed_year,
                MONTHS[match.group(2)],
                int(match.group(1)),
            )
        except ValueError:
            continue
        if candidate not in values:
            values.append(candidate)
    if not values:
        return None, None
    values.sort()
    return values[0], values[-1]


def _extract_ordinal(text: str) -> int | None:
    for marker, index in ORDINALS.items():
        if marker in text:
            return index
    return None


def _asks_for_calculation(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CALCULATION_PATTERNS)


def _is_chat_query(
    text: str,
    *,
    has_state: bool,
    banks: list[str],
    products: list[str],
    fields: list[str],
    year: int | None,
    has_date_range: bool,
) -> bool:
    if banks or products or fields or year is not None or has_date_range:
        return False
    if any(term in text for term in FINANCIAL_DATA_TERMS):
        return False
    if any(term in text for term in CASUAL_QUERY_TERMS):
        return True
    if any(term in text for term in OUT_OF_DOMAIN_TERMS):
        return True
    if any(
        marker in text
        for marker in (*FOLLOW_UP_MARKERS, *CONTEXT_REFERENCE_MARKERS)
    ) or re.search(r"(?<!\w)bu(?:nu|na|nun)?(?!\w)", text):
        return False
    if has_state and any(
        marker in text
        for marker in SOFT_CONTEXT_MARKERS
    ):
        return False
    return True


def _token_is_represented(token: str, roots: set[str]) -> bool:
    if token in roots:
        return True
    for root in roots:
        if len(root) < 3 or not token.startswith(root):
            continue
        if token[len(root) :] in _TOKEN_SUFFIXES:
            return True
    return False


def _represented_roots(
    banks: list[str],
    products: list[str],
    fields: list[str],
) -> set[str]:
    phrases: list[str] = []
    for bank in banks:
        phrases.extend(BANKS.get(bank, (bank,)))
        phrases.append(bank)
    for product in products:
        phrases.extend(PRODUCTS.get(product, (product.replace("_", " "),)))
        phrases.append(PRODUCT_LABELS.get(product, product.replace("_", " ")))
    for field in fields:
        phrases.extend(FIELDS.get(field, ()))
        label = FIELD_LABELS.get(field)
        if label:
            phrases.append(label)
    return {
        token
        for phrase in phrases
        for token in re.findall(r"[a-z0-9]+", normalize_text(phrase))
    }


def _without_date_literals(text: str) -> str:
    output = _TEXTUAL_DATE_PATTERN.sub(" ", text)
    output = re.sub(
        r"(?<!\d)(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}(?!\d)",
        " ",
        output,
    )
    output = re.sub(
        r"(?<!\d)\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}(?!\d)",
        " ",
        output,
    )
    return output


def _semantic_qualifiers(
    query: str,
    banks: list[str],
    products: list[str],
    fields: list[str],
) -> list[str]:
    text = _without_date_literals(normalize_text(query))
    prior_answer_reference = any(
        marker in text
        for marker in ("onceki cevap", "cevabinda", "dedin", "demistin")
    )
    measurements = (
        []
        if prior_answer_reference
        else list(
            dict.fromkeys(
                re.sub(r"\s+", " ", match.group(0)).strip()
                for match in _MEASUREMENT_QUALIFIER_PATTERN.finditer(text)
            )
        )
    )
    text = _MEASUREMENT_QUALIFIER_PATTERN.sub(" ", text)
    roots = _represented_roots(banks, products, fields)
    roots.update(MONTHS)
    generic_roots = set(SEMANTIC_GENERIC_WORDS)
    qualifiers: list[str] = []
    for token in re.findall(r"[a-z0-9]+", text):
        if len(token) < 2:
            continue
        if token in SEMANTIC_STOP_WORDS:
            continue
        if _token_is_represented(token, generic_roots):
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", token):
            continue
        if _token_is_represented(token, roots):
            continue
        if token not in qualifiers:
            qualifiers.append(token)
    qualifiers.extend(item for item in measurements if item not in qualifiers)
    return qualifiers


def _intent(
    text: str,
    year: int | None,
    *,
    has_date_range: bool = False,
    bank_count: int = 0,
) -> str:
    if _asks_for_calculation(text):
        return "calculate"
    if any(
        word in text
        for word in ("karsilastir", "hangisi", "en avantajli", "gore sirala")
    ):
        return "compare"
    if "hangi banka" in text:
        return "compare"
    if bank_count != 1 and any(
        marker in text
        for marker in (
            "en dusuk",
            "en fazla",
            "en kisa",
            "en uzun",
            "en yuksek",
        )
    ):
        return "compare"
    if year is not None or has_date_range or any(
        word in text for word in ("gecmis", "arsiv", "tarihsel", "onceki yil")
    ):
        return "historical"
    plural_show = any(
        verb in text for verb in ("goster", "getir", "sun", "ver")
    ) and bool(
        re.search(r"\b[a-z]+(?:lar|ler)(?:i|ini|unu)?\b", text)
    )
    if plural_show or any(
        word in text for word in ("listele", "sirala", "nelerdir", "kac tane")
    ):
        return "list"
    return "lookup"


def _scope(text: str, year: int | None) -> str:
    if any(word in text for word in ("tum donem", "hepsi", "guncel ve gecmis")):
        return "all"
    if year is not None or any(
        word in text for word in ("gecmis", "arsiv", "tarihsel", "onceki")
    ):
        return "historical"
    return "current"


def _selected_offer(
    references: list[OfferReference], ordinal: int
) -> OfferReference | None:
    ordered = sorted(references, key=lambda item: item.rank)
    if ordinal < 0 or ordinal >= len(ordered):
        return None
    return ordered[ordinal]


def _clarification_from_state(
    query: str,
    fields: list[str],
    state: SessionState,
    question: str,
    *,
    banks: list[str] | None = None,
    products: list[str] | None = None,
    offer_ids: list[str] | None = None,
    scope: str | None = None,
    year: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    use_state_period: bool = True,
    inherit_scope: bool = True,
) -> QueryRoute:
    resolved_banks = list(state.active_banks if banks is None else banks)
    resolved_products = list(
        state.active_products if products is None else products
    )
    resolved_offers = list(
        state.active_offer_ids if offer_ids is None else offer_ids
    )
    inherited: list[str] = []
    if resolved_banks:
        inherited.append("banks")
    if resolved_products:
        inherited.append("product_types")
    if state.last_standalone_query and inherit_scope:
        inherited.append("scope")
        if use_state_period and state.active_year is not None:
            inherited.append("year")
        if use_state_period and (
            state.active_date_from is not None
            or state.active_date_to is not None
        ):
            inherited.append("date_range")
    if resolved_offers:
        inherited.append("offer_ids")
    return QueryRoute(
        standalone_query=query,
        intent="clarification",
        banks=resolved_banks,
        product_types=resolved_products,
        field_types=fields,
        scope=(state.active_scope if use_state_period else scope or "current"),
        year=state.active_year if use_state_period else year,
        date_from=state.active_date_from if use_state_period else date_from,
        date_to=state.active_date_to if use_state_period else date_to,
        offer_ids=resolved_offers,
        inherited_fields=inherited,
        needs_clarification=True,
        clarification_question=question,
    )


def _standalone_text(
    query: str,
    banks: list[str],
    products: list[str],
    fields: list[str],
    scope: str,
    year: int | None,
) -> str:
    normalized_query = normalize_text(query)
    parts: list[str] = []
    parts.extend(banks)
    if year is not None:
        parts.append(str(year))
    elif scope == "current":
        parts.append("guncel")
    elif scope == "historical":
        parts.append("gecmis")
    elif scope == "all":
        parts.append("tum donem")
    parts.extend(
        PRODUCT_LABELS.get(code, code.lower().replace("_", " "))
        for code in products
    )
    parts.extend(_semantic_qualifiers(query, banks, products, fields))
    ordering_prefix = ""
    if any(
        marker in normalized_query
        for marker in ("en dusuk", "en az", "en kisa")
    ):
        ordering_prefix = "en dusuk "
    elif any(
        marker in normalized_query
        for marker in ("en yuksek", "en fazla", "en uzun")
    ):
        ordering_prefix = "en yuksek "
    field_parts = [FIELD_LABELS[item] for item in fields if item in FIELD_LABELS]
    if ordering_prefix and field_parts:
        field_parts[0] = ordering_prefix + field_parts[0]
    parts.extend(field_parts)
    operation = next(
        (
            marker
            for marker in ("karsilastir", "listele", "sirala")
            if marker in normalized_query
        ),
        None,
    )
    if operation and fields:
        count_match = re.search(
            r"(?<!\w)(bir|iki|uc|dort|bes|[1-5])\s+"
            r"(?:guncel\s+)?(?:[a-z]+\s+){0,3}"
            r"(?:teklif|kampanya|urun)[a-z]*(?!\w)",
            normalized_query,
        )
        if count_match:
            parts.append(count_match.group(1))
        parts.append(operation)
    if not parts:
        return query.strip()
    return " ".join(parts) + " nedir?"


def build_standalone_query(query: str, route: QueryRoute) -> str:
    standalone = _standalone_text(
        query,
        route.banks,
        route.product_types,
        route.field_types,
        route.scope,
        route.year,
    )
    if route.year is None and (route.date_from is not None or route.date_to is not None):
        boundaries = " - ".join(
            value.isoformat()
            for value in (route.date_from, route.date_to)
            if value is not None
        )
        standalone = f"{standalone.rstrip('?')} ({boundaries})?"
    return standalone


def has_explicit_product(query: str) -> bool:
    text = normalize_text(query)
    return bool(_extract_products(text))


def _model_query_covers_context(
    candidate: str,
    banks: list[str],
    products: list[str],
    year: int | None,
    qualifiers: list[str] | None = None,
) -> bool:
    normalized = normalize_text(candidate)
    for bank in banks:
        aliases = BANKS.get(bank, (bank,))
        if not any(normalize_text(alias) in normalized for alias in aliases):
            return False
    for product in products:
        aliases = PRODUCTS.get(product, (product.replace("_", " "),))
        if not any(normalize_text(alias) in normalized for alias in aliases):
            return False
    if year is not None and str(year) not in normalized:
        return False
    return all(normalize_text(item) in normalized for item in qualifiers or [])


class QueryRouter:
    def resolve(
        self,
        query: str,
        state: SessionState | None = None,
        model_route: dict[str, Any] | None = None,
    ) -> QueryRoute:
        state = state or SessionState()
        text = normalize_text(query)
        explicit_banks = _extract_banks(text)
        explicit_products = _extract_products(text)
        explicit_fields = _extract_fields(text)
        extracted_year = _extract_year(text)
        explicit_date_from, explicit_date_to = _extract_date_range(text)
        has_explicit_date_range = bool(explicit_date_from or explicit_date_to)
        if (
            has_explicit_date_range
            and "kampanya" in text
            and "campaign_date" not in explicit_fields
        ):
            explicit_fields.append("campaign_date")
        if (
            explicit_products == ["KART"]
            and set(explicit_fields).intersection(
                {"reward", "spending_threshold", "campaign_date"}
            )
        ):
            explicit_products = ["KART_KAMPANYASI"]
        if (
            not explicit_products
            and "reward" in explicit_fields
            and (
                "kampanya" in text
                or "alisveris puan" in text
                or "worldpuan" in text
                or "altin puan" in text
            )
        ):
            explicit_products = ["KART_KAMPANYASI"]
        active_campaign = (
            extracted_year is None
            and not has_explicit_date_range
            and (
                "kampanya" in text
                or "KART_KAMPANYASI" in explicit_products
                or "MOBIL_UYGULAMA_KAMPANYASI" in explicit_products
            )
            and any(
                marker in text
                for marker in ("aktif", "devam eden", "halen gecerli")
            )
        )
        if active_campaign:
            explicit_date_from = date.today()
            explicit_date_to = explicit_date_from
            has_explicit_date_range = True
        year = extracted_year if not has_explicit_date_range else None
        ordinal = _extract_ordinal(text)
        has_state = bool(state.last_standalone_query)
        model_intent = (
            model_route.get("intent")
            if isinstance(model_route, dict)
            else None
        )
        model_financial_follow_up = bool(
            has_state
            and model_intent
            in {
                "lookup",
                "compare",
                "list",
                "calculate",
            }
            and not explicit_banks
            and not explicit_products
            and not any(term in text for term in CASUAL_QUERY_TERMS)
            and not any(term in text for term in OUT_OF_DOMAIN_TERMS)
        )
        if _is_chat_query(
            text,
            has_state=has_state,
            banks=explicit_banks,
            products=explicit_products,
            fields=explicit_fields,
            year=extracted_year,
            has_date_range=has_explicit_date_range,
        ) and not model_financial_follow_up:
            return QueryRoute(
                standalone_query=query.strip(),
                intent="chat",
                scope="current",
            )
        bare_offer_reference = bool(
            re.search(r"(?<!\w)bu(?:nu|na|nun)?(?!\w)", text)
        )
        explicit_scope = any(
            marker in text
            for marker in (
                "guncel",
                "bugun",
                "gecmis",
                "arsiv",
                "tarihsel",
                "tum donem",
            )
        )
        follow_up = ordinal is not None or any(
            marker in text
            for marker in (
                *FOLLOW_UP_MARKERS,
                *CONTEXT_REFERENCE_MARKERS,
                *SOFT_CONTEXT_MARKERS,
            )
        )
        follow_up = follow_up or bare_offer_reference
        follow_up = follow_up or model_financial_follow_up
        follow_up = follow_up or (
            has_state
            and bool(extracted_year or has_explicit_date_range or explicit_scope)
        )
        field_context_follow_up = (
            bool(explicit_fields)
            and not explicit_products
            and has_state
        )
        out_of_domain = any(marker in text for marker in OUT_OF_DOMAIN_TERMS)
        topic_change = any(marker in text for marker in TOPIC_CHANGE_MARKERS) and bool(
            explicit_banks or explicit_products
        )
        topic_change = topic_change or out_of_domain

        banks = list(explicit_banks)
        products = list(explicit_products)
        fields = list(explicit_fields)
        inherited: list[str] = []
        offer_ids: list[str] = []
        selected: OfferReference | None = None
        referential_offer = any(
            marker in text for marker in ("onun", "bunun", "bu teklif")
        ) or bare_offer_reference

        if (
            follow_up
            and not has_state
            and not explicit_banks
            and not explicit_products
        ):
            return QueryRoute(
                standalone_query=query,
                intent="clarification",
                field_types=fields,
                scope="current",
                needs_clarification=True,
                clarification_question=(
                    "Hangi banka, urun veya donem icin bilgi istediginizi "
                    "belirtir misiniz?"
                ),
            )

        if ordinal is not None:
            selected = _selected_offer(state.ranked_offers, ordinal)
            if selected is None:
                return _clarification_from_state(
                    query,
                    fields,
                    state,
                    (
                        "Hangi teklifi kastettiginizi banka veya urun adiyla "
                        "belirtir misiniz?"
                    ),
                    offer_ids=[],
                )
            offer_ids = [selected.offer_id]
            if not banks:
                banks = [selected.bank]
                inherited.append("banks")
            if not products:
                products = list(selected.product_types)
                inherited.append("product_types")
            inherited.append("offer_ids")

        if ordinal is None and referential_offer and state.ranked_offers:
            candidates = list(state.ranked_offers)
            if state.active_offer_ids:
                active_ids = set(state.active_offer_ids)
                active_candidates = [
                    item for item in candidates if item.offer_id in active_ids
                ]
                if active_candidates:
                    candidates = active_candidates
            if explicit_banks:
                candidates = [
                    item for item in candidates if item.bank in explicit_banks
                ]
            if explicit_products:
                product_set = set(explicit_products)
                candidates = [
                    item
                    for item in candidates
                    if product_set.intersection(item.product_types)
                ]
            if len(candidates) != 1:
                return _clarification_from_state(
                    query,
                    fields,
                    state,
                    (
                        "Hangi teklifi kastettiginizi banka veya urun adiyla "
                        "belirtir misiniz?"
                    ),
                    banks=list(
                        dict.fromkeys(item.bank for item in candidates)
                    ),
                    products=list(
                        dict.fromkeys(
                            product
                            for item in candidates
                            for product in item.product_types
                        )
                    ),
                    offer_ids=[item.offer_id for item in candidates],
                )
            selected = candidates[0]
            offer_ids = [selected.offer_id]
            if not banks:
                banks = [selected.bank]
                inherited.append("banks")
            if not products:
                products = list(selected.product_types)
                inherited.append("product_types")
            inherited.append("offer_ids")

        if (
            ordinal is None
            and referential_offer
            and not state.ranked_offers
            and state.active_offer_ids
        ):
            if len(state.active_offer_ids) != 1:
                return _clarification_from_state(
                    query,
                    fields,
                    state,
                    (
                        "Hangi teklifi kastettiginizi banka veya urun adiyla "
                        "belirtir misiniz?"
                    ),
                )
            offer_ids = list(state.active_offer_ids)
            inherited.append("offer_ids")

        if referential_offer and selected is None and not offer_ids:
            return _clarification_from_state(
                query,
                fields,
                state,
                (
                    "Hangi teklifi kastettiginizi banka veya urun adiyla "
                    "belirtir misiniz?"
                ),
            )

        can_inherit = follow_up or field_context_follow_up
        if can_inherit and (not topic_change or out_of_domain):
            inherit_broad_bank = (
                not state.broad_bank_context
                or "ayni banka" in text
                or referential_offer
                or ordinal is not None
            )
            if not banks and state.active_banks and inherit_broad_bank:
                if "ayni banka" in text and len(state.active_banks) != 1:
                    requested_scope = (
                        _scope(text, extracted_year)
                        if bool(
                            extracted_year
                            or has_explicit_date_range
                            or explicit_scope
                        )
                        else state.active_scope
                    )
                    requested_date_from = explicit_date_from
                    requested_date_to = explicit_date_to
                    return _clarification_from_state(
                        query,
                        fields,
                        state,
                        (
                            "Onceki sonuctaki hangi bankayi kastettiginizi "
                            "belirtir misiniz?"
                        ),
                        scope=requested_scope,
                        year=year,
                        date_from=requested_date_from,
                        date_to=requested_date_to,
                        use_state_period=False,
                        inherit_scope=False,
                    )
                if "onun" in text and len(state.active_banks) > 1 and not selected:
                    return _clarification_from_state(
                        query,
                        fields,
                        state,
                        (
                            "Hangi bankaya ait teklifi kastettiginizi belirtir "
                            "misiniz?"
                        ),
                    )
                banks = list(state.active_banks)
                inherited.append("banks")
            if not products and state.active_products and not out_of_domain:
                products = list(state.active_products)
                inherited.append("product_types")
            if not fields and state.last_field_types and not out_of_domain:
                fields = list(state.last_field_types)
                inherited.append("field_types")

        resolved_scope = _scope(text, extracted_year)
        date_from = explicit_date_from
        date_to = explicit_date_to
        period_changed = bool(extracted_year or has_explicit_date_range or explicit_scope)
        if can_inherit and not topic_change and not period_changed:
            resolved_scope = state.active_scope
            year = state.active_year
            date_from = state.active_date_from
            date_to = state.active_date_to
            if year is not None:
                inherited.append("year")
            if date_from is not None or date_to is not None:
                inherited.append("date_range")
            inherited.append("scope")

        identity_changed = (
            bool(explicit_banks)
            and set(explicit_banks) != set(state.active_banks)
        ) or (
            bool(explicit_products)
            and set(explicit_products) != set(state.active_products)
        )
        if (
            can_inherit
            and not topic_change
            and not period_changed
            and not identity_changed
            and not offer_ids
            and state.active_offer_ids
        ):
            offer_ids = list(state.active_offer_ids)
            inherited.append("offer_ids")

        if topic_change:
            offer_ids = []
            if year is None:
                resolved_scope = "current"
                date_from = None
                date_to = None

        if explicit_banks and set(explicit_banks) != set(state.active_banks):
            offer_ids = []
        if explicit_products and set(explicit_products) != set(state.active_products):
            offer_ids = []

        intent = _intent(
            text,
            extracted_year,
            has_date_range=has_explicit_date_range,
            bank_count=len(banks),
        )
        if resolved_scope == "historical" and intent in {"lookup", "list"}:
            intent = "historical"
        standalone = _standalone_text(
            query, banks, products, fields, resolved_scope, year
        )
        if year is None and (date_from is not None or date_to is not None):
            boundaries = " - ".join(
                value.isoformat()
                for value in (date_from, date_to)
                if value is not None
            )
            standalone = f"{standalone.rstrip('?')} ({boundaries})?"
        if model_route:
            candidate = str(model_route.get("standalone_query") or "").strip()
            if (
                candidate
                and len(candidate) <= 2000
                and not period_changed
                and not explicit_fields
                and resolved_scope == "current"
                and not offer_ids
                and _model_query_covers_context(
                    candidate,
                    banks,
                    products,
                    year,
                    _semantic_qualifiers(query, banks, products, fields),
                )
            ):
                standalone = candidate
            if (
                intent == "lookup"
                and ordinal is None
                and not offer_ids
                and model_intent in {
                "compare",
                "list",
                "historical",
                }
            ):
                intent = model_intent

        return QueryRoute(
            standalone_query=standalone,
            intent=intent,
            banks=banks,
            product_types=products,
            field_types=fields,
            scope=resolved_scope,
            year=year,
            date_from=date_from,
            date_to=date_to,
            offer_ids=offer_ids,
            inherited_fields=inherited,
        )
