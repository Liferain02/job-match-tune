from __future__ import annotations

import re
from html import unescape

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - dependency installed in runtime env
    BeautifulSoup = None


BUTTON_PATTERNS = [
    "立即沟通",
    "立即投递",
    "投递简历",
    "申请职位",
    "收藏职位",
    "查看地图",
    "分享职位",
]


def strip_html(html: str) -> str:
    if not html:
        return ""
    if BeautifulSoup is None:
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return unescape(text)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text("\n")


def mask_private_info(text: str) -> str:
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号]", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[邮箱]", text)
    text = re.sub(r"(微信|VX|WeChat)[:：]?\s*[A-Za-z0-9_-]{5,}", r"\1：[已脱敏]", text, flags=re.I)
    return text


def normalize_space(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def remove_boilerplate(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if any(pattern in line for pattern in BUTTON_PATTERNS):
            continue
        if len(line) <= 2 and line in {"展开", "收起", "更多"}:
            continue
        lines.append(line)
    return "\n".join(lines)


def deduplicate_lines(text: str) -> str:
    seen: set[str] = set()
    lines = []
    for line in text.splitlines():
        key = re.sub(r"\s+", "", line.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines)


def clean_text(raw: str, *, is_html: bool = False) -> str:
    text = strip_html(raw) if is_html else raw
    text = unescape(text)
    text = mask_private_info(text)
    text = normalize_space(text)
    text = remove_boilerplate(text)
    text = deduplicate_lines(text)
    return normalize_space(text)
