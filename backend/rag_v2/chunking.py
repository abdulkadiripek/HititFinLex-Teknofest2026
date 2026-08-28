from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .identity import normalize_text


_WORD_PATTERN = re.compile(r"\S+")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+")
_BULLET_PREFIX = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_TERMINAL_PUNCTUATION = frozenset(".!?;,")
_TITLE_CONNECTORS = {
    "and",
    "ile",
    "icin",
    "ve",
    "veya",
}
_NAVIGATION_TERMS = {
    "ac",
    "ana sayfa",
    "anasayfa",
    "giris",
    "giris yap",
    "menu",
    "mobil bankacilik",
}


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    chunk_index: int
    content: str
    section_heading: str | None
    token_count: int
    content_hash: str
    facts: tuple[dict[str, Any], ...] = ()
    is_navigation: bool = False


@dataclass(frozen=True, slots=True)
class _Section:
    heading: str | None
    content: str
    is_navigation: bool


def clean_source_text(value: str | None) -> str:
    text = html.unescape(value or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[\t \u00a0]+", " ", line).strip()
        for line in text.split("\n")
    ]
    lines = [
        "" if normalize_text(line) in {"&nbsp", "nbsp", "&#160;"} else line
        for line in lines
    ]
    output: list[str] = []
    previous_blank = True
    for line in lines:
        if not line:
            if not previous_blank and output:
                output.append("")
            previous_blank = True
            continue
        output.append(line)
        previous_blank = False
    while output and not output[-1]:
        output.pop()
    return "\n".join(_collapse_repeated_blocks(output))


def _collapse_repeated_blocks(
    lines: Sequence[str],
    max_block_size: int = 8,
) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(lines):
        repeated_size: int | None = None
        repeated_count = 1
        maximum = min(max_block_size, (len(lines) - index) // 2)
        for size in range(1, maximum + 1):
            block = lines[index : index + size]
            next_block = lines[index + size : index + (2 * size)]
            if block != next_block or not any(item for item in block):
                continue
            count = 2
            while (
                index + ((count + 1) * size) <= len(lines)
                and lines[index + (count * size) : index + ((count + 1) * size)]
                == block
            ):
                count += 1
            repeated_size = size
            repeated_count = count
            break
        if repeated_size is None:
            output.append(lines[index])
            index += 1
            continue
        output.extend(lines[index : index + repeated_size])
        index += repeated_size * repeated_count
    return output


def _clean_heading(line: str) -> str:
    cleaned = _MARKDOWN_HEADING.sub("", line).strip()
    return cleaned[:-1].strip() if cleaned.endswith(":") else cleaned


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 140 or _BULLET_PREFIX.match(stripped):
        return False
    if _MARKDOWN_HEADING.match(stripped) or stripped.endswith(":"):
        return True
    words = _WORD_PATTERN.findall(stripped)
    if not words or len(words) > 12:
        return False
    letters = "".join(character for character in stripped if character.isalpha())
    if letters and letters.isupper():
        return True
    if stripped[-1] in _TERMINAL_PUNCTUATION:
        return False
    if len(words) == 1:
        return len(letters) >= 4 and words[0][0].isupper()

    meaningful = [
        word
        for word in words
        if normalize_text(word.strip("()[]{}:;-")) not in _TITLE_CONNECTORS
    ]
    return bool(meaningful) and all(
        any(character.isalpha() for character in word)
        and next(character for character in word if character.isalpha()).isupper()
        for word in meaningful
    )


def _looks_like_navigation(line: str) -> bool:
    normalized = normalize_text(line.strip("-*>| "))
    if normalized in _NAVIGATION_TERMS:
        return True
    separators = sum(line.count(marker) for marker in ("|", "›", ">", " / "))
    return separators >= 3 and len(_WORD_PATTERN.findall(line)) <= 24


def _split_sections(text: str, page_title: str | None) -> list[_Section]:
    cleaned = clean_source_text(text)
    if not cleaned:
        return []

    sections: list[_Section] = []
    heading = clean_source_text(page_title) or None
    heading_chain: list[str] = [heading] if heading else []
    body: list[str] = []
    body_navigation: bool | None = None

    def flush() -> None:
        nonlocal body, body_navigation
        source_lines = [*heading_chain[-3:], *body]
        content = clean_source_text("\n".join(source_lines))
        if content:
            sections.append(
                _Section(
                    heading=" > ".join(heading_chain[-3:]) or None,
                    content=content,
                    is_navigation=bool(body_navigation),
                )
            )
        body = []
        body_navigation = None

    for line in cleaned.splitlines():
        if not line:
            if body and body[-1] != "":
                body.append("")
            continue

        navigation = _looks_like_navigation(line)
        if navigation:
            if body_navigation is False:
                flush()
            body_navigation = True
            body.append(line)
            continue
        if body_navigation is True:
            flush()

        if _looks_like_heading(line):
            flush()
            clean_heading = _clean_heading(line)
            if clean_heading:
                if heading_chain and normalize_text(
                    heading_chain[-1]
                ) == normalize_text(clean_heading):
                    continue
                heading_chain.append(clean_heading)
            continue

        body_navigation = False
        body.append(line)

    flush()
    if not sections and heading_chain:
        content = clean_source_text(text)
        if content:
            sections.append(
                _Section(
                    heading=" > ".join(heading_chain[-3:]),
                    content=content,
                    is_navigation=False,
                )
            )
    return sections


def _merge_small_sections(
    sections: Sequence[_Section],
    *,
    minimum_words: int = 24,
) -> list[_Section]:
    merged: list[_Section] = []
    for section in sections:
        word_count = len(_WORD_PATTERN.findall(section.content))
        if not merged or merged[-1].is_navigation != section.is_navigation:
            merged.append(section)
            continue
        previous = merged[-1]
        previous_count = len(_WORD_PATTERN.findall(previous.content))
        if previous_count >= minimum_words and word_count >= minimum_words:
            merged.append(section)
            continue
        merged[-1] = _Section(
            heading=section.heading or previous.heading,
            content=clean_source_text(f"{previous.content}\n{section.content}"),
            is_navigation=section.is_navigation,
        )
    return merged


def _window_section(
    section: _Section,
    *,
    max_words: int,
    overlap_words: int,
) -> list[tuple[str, str | None, bool]]:
    matches = list(_WORD_PATTERN.finditer(section.content))
    if not matches:
        return []
    output: list[tuple[str, str | None, bool]] = []
    start_word = 0
    while start_word < len(matches):
        end_word = min(start_word + max_words, len(matches))
        if end_word < len(matches):
            minimum_break = start_word + max(1, int(max_words * 0.65))
            for candidate in range(end_word - 1, minimum_break - 1, -1):
                token = matches[candidate].group(0)
                if token and token[-1] in ".!?;":
                    end_word = candidate + 1
                    break

        char_start = matches[start_word].start()
        char_end = matches[end_word - 1].end()
        content = clean_source_text(section.content[char_start:char_end])
        if content:
            output.append((content, section.heading, section.is_navigation))
        if end_word >= len(matches):
            break
        next_start = end_word - overlap_words
        start_word = max(start_word + 1, next_start)
    return output


def chunk_document(
    text: str,
    *,
    page_title: str | None = None,
    max_words: int = 220,
    overlap_words: int = 40,
    include_navigation: bool = False,
) -> list[ChunkDraft]:
    if max_words < 16:
        raise ValueError("max_words must be at least 16")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between zero and max_words")

    chunks: list[ChunkDraft] = []
    for section in _merge_small_sections(_split_sections(text, page_title)):
        if section.is_navigation and not include_navigation:
            continue
        for content, heading, is_navigation in _window_section(
            section,
            max_words=max_words,
            overlap_words=overlap_words,
        ):
            chunks.append(
                ChunkDraft(
                    chunk_index=len(chunks),
                    content=content,
                    section_heading=heading,
                    token_count=len(_WORD_PATTERN.findall(content)),
                    content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    is_navigation=is_navigation,
                )
            )
    return chunks


def _fact_payload(fact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fact_type": str(fact.get("fact_type") or "").strip(),
        "fact_text": str(fact.get("fact_text") or "").strip(),
        "normalized_value": fact.get("normalized_value"),
        "evidence_text": str(fact.get("evidence_text") or "").strip(),
        "confidence": float(fact.get("confidence") or 0.0),
    }


def attach_facts(
    chunks: Sequence[ChunkDraft],
    facts: Iterable[Mapping[str, Any]],
) -> list[ChunkDraft]:
    attached: list[list[dict[str, Any]]] = [list(chunk.facts) for chunk in chunks]
    normalized_chunks = [normalize_text(chunk.content) for chunk in chunks]

    for raw_fact in facts:
        fact = _fact_payload(raw_fact)
        evidence = normalize_text(fact["evidence_text"])
        if not evidence:
            continue
        candidates: list[tuple[int, int]] = []
        for index, content in enumerate(normalized_chunks):
            position = content.find(evidence)
            if position < 0:
                continue
            right_margin = len(content) - position - len(evidence)
            candidates.append((min(position, right_margin), index))
        if not candidates:
            continue
        _, best_index = max(candidates, key=lambda item: (item[0], -item[1]))
        attached[best_index].append(fact)

    return [
        replace(chunk, facts=tuple(attached[index]))
        for index, chunk in enumerate(chunks)
    ]


def build_embedding_context(
    *,
    bank_name: str,
    primary_product: str | None,
    page_title: str | None,
    section_heading: str | None,
    content: str,
) -> str:
    return "\n".join(
        (
            f"Title: {page_title or '-'}",
            f"Bank: {bank_name or '-'}",
            f"Product: {primary_product or '-'}",
            f"Section: {section_heading or '-'}",
            "Content:",
            content,
        )
    )
