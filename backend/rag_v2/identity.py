from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}


def normalize_text(value: str | None) -> str:
    text = (value or "").strip().casefold()
    text = text.translate(str.maketrans({"ı": "i", "ş": "s", "ğ": "g"}))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text)


def canonicalize_url(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return normalize_text(value)
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        return normalize_text(value)
    if port and not (
        (parsed.scheme.lower() == "http" and port == 80)
        or (parsed.scheme.lower() == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in TRACKING_PARAMETERS
    ]
    query.sort()
    return urlunsplit(
        (parsed.scheme.lower() or "https", host, path, urlencode(query), "")
    )


def stable_digest(*parts: object) -> str:
    encoded = "\x1f".join(normalize_text(str(part)) for part in parts)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stable_offer_id(
    *,
    bank: str,
    product: str | None,
    source_url: str | None,
    title: str | None,
    content_boundary: str,
    campaign_start: object = None,
    campaign_end: object = None,
) -> str:
    return stable_digest(
        "offer-v2",
        bank,
        product or "",
        canonicalize_url(source_url),
        title or "",
        content_boundary,
        campaign_start or "",
        campaign_end or "",
    )


def stable_chunk_id(
    scope: str,
    document_id: object,
    chunk_index: int,
    content: str,
) -> str:
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return stable_digest(
        "chunk-v2", scope, document_id, chunk_index, content_hash
    )


def qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hititfinlex:{chunk_id}"))
