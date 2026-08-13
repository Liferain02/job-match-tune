from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any


OCR_SEPARATORS = re.compile(r"[\s-]+")
ASCII_WORD_CHAR = re.compile(r"[A-Za-z0-9]")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def _ocr_key(value: str) -> str:
    """Build a comparison key without mutating arbitrary source text."""

    return OCR_SEPARATORS.sub("", _text(value)).casefold()


def _vocabulary(schema: dict[str, Any]) -> dict[str, list[str]]:
    return {
        str(canonical): [str(canonical), *(str(alias) for alias in aliases or [])]
        for canonical, aliases in (schema.get("skill_alias") or {}).items()
    }


def _lookup_maps(schema: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    compact_candidates: dict[str, set[str]] = defaultdict(set)
    for canonical, candidates in _vocabulary(schema).items():
        for candidate in candidates:
            normalized = _text(candidate)
            if not normalized:
                continue
            exact[normalized.casefold()] = canonical
            compact_candidates[_ocr_key(normalized)].add(canonical)
    compact = {
        key: next(iter(canonicals))
        for key, canonicals in compact_candidates.items()
        if key and len(canonicals) == 1
    }
    return exact, compact


def canonicalize_skill_name(value: Any, schema: dict[str, Any]) -> str | None:
    normalized = _text(value)
    if not normalized:
        return None
    exact, compact = _lookup_maps(schema)
    canonical = exact.get(normalized.casefold())
    if canonical:
        return canonical
    return compact.get(_ocr_key(normalized))


def canonicalize_skill_list(
    values: list[Any],
    schema: dict[str, Any],
    *,
    keep_unknown: bool = False,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _text(value)
        canonical = canonicalize_skill_name(normalized, schema)
        item = canonical or (normalized if keep_unknown else "")
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _exact_candidate_pattern(candidate: str) -> str:
    normalized = _text(candidate)
    escaped = re.escape(normalized)
    first = normalized[:1]
    last = normalized[-1:]
    left = r"(?<![A-Za-z0-9])" if ASCII_WORD_CHAR.search(first) else ""
    right = r"(?![A-Za-z0-9-])" if ASCII_WORD_CHAR.search(last) else ""
    if normalized.casefold() == "c":
        # A standalone C skill must not be double-counted inside C++, C# or C 语言.
        right = r"(?!\s*(?:\+|#|语言))(?![A-Za-z0-9-])"
    return f"{left}{escaped}{right}"


def _allows_ocr_flex(candidate: str) -> bool:
    key = _ocr_key(candidate)
    return len(key) >= 4 or any(char in key for char in "+#.")


def _ocr_candidate_pattern(candidate: str) -> str | None:
    if not _allows_ocr_flex(candidate):
        return None
    compact = OCR_SEPARATORS.sub("", _text(candidate))
    if not compact:
        return None
    body = r"[\s-]{0,3}".join(re.escape(char) for char in compact)
    first = compact[:1]
    last = compact[-1:]
    left = r"(?<![A-Za-z0-9])" if ASCII_WORD_CHAR.search(first) else ""
    # Reject ordinary compounds such as "pytest-like" at the right boundary.
    right = r"(?![A-Za-z0-9-])" if ASCII_WORD_CHAR.search(last) else r"(?![A-Za-z0-9])"
    return f"{left}{body}{right}"


def contains_skill_candidate(text: str, candidate: str) -> bool:
    haystack = unicodedata.normalize("NFKC", str(text))
    normalized_candidate = _text(candidate)
    if not haystack or not normalized_candidate:
        return False
    if re.search(_exact_candidate_pattern(normalized_candidate), haystack, flags=re.I):
        return True
    ocr_pattern = _ocr_candidate_pattern(normalized_candidate)
    return bool(ocr_pattern and re.search(ocr_pattern, haystack, flags=re.I))


def extract_known_skills(text: str, schema: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for canonical, candidates in _vocabulary(schema).items():
        if any(contains_skill_candidate(text, candidate) for candidate in candidates):
            found.append(canonical)
    return found


def merge_skill_aliases(*schemas: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, list[str]] = {}
    for schema in schemas:
        for canonical, aliases in (schema.get("skill_alias") or {}).items():
            bucket = merged.setdefault(str(canonical), [])
            for alias in aliases or []:
                alias_text = str(alias)
                if alias_text not in bucket:
                    bucket.append(alias_text)
    return {"skill_alias": merged}
