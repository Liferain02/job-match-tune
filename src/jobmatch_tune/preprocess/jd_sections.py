from __future__ import annotations

import re


SECTION_ALIASES = {
    "responsibilities": [
        "岗位职责",
        "工作职责",
        "职位描述",
        "岗位描述",
        "工作内容",
        "职位内容",
        "主要职责",
        "核心职责",
        "职责描述",
    ],
    "requirements": [
        "任职要求",
        "岗位要求",
        "职位要求",
        "任职资格要求",
        "任职资格",
        "能力要求",
    ],
    "bonus": ["加分项", "加分条件", "优先条件"],
}

SECTION_BY_ALIAS = {
    alias: section for section, aliases in SECTION_ALIASES.items() for alias in aliases
}
_ALIASES_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(SECTION_BY_ALIAS, key=len, reverse=True)
)
SECTION_MARKER = re.compile(
    rf"(?:<[^>]+>\s*(?P<html>{_ALIASES_PATTERN})\s*</[^>]+>"
    rf"|[一二三四五六七八九十]+[、.．]\s*(?P<numbered>{_ALIASES_PATTERN})(?=\s)"
    rf"|[【\[]\s*(?P<bracket>{_ALIASES_PATTERN})\s*[】\]]"
    rf"|(?P<colon>{_ALIASES_PATTERN})\s*[：:]"
    rf"|(?<!\S)(?P<spaced>{_ALIASES_PATTERN})(?=\s))"
)

_RESPONSIBILITY_SIGNALS = (
    "负责",
    "参与",
    "开发",
    "设计",
    "维护",
    "推动",
    "跟进",
    "搭建",
    "建设",
    "制定",
    "优化",
    "协助",
    "完成",
)
_REQUIREMENT_SIGNALS = (
    "学历",
    "本科",
    "硕士",
    "博士",
    "大专",
    "经验",
    "熟悉",
    "精通",
    "具备",
    "优先",
    "能力",
    "专业",
)


def _signal_count(text: str, signals: tuple[str, ...]) -> int:
    return sum(text.count(signal) for signal in signals)


def _repair_known_source_inversion(sections: dict[str, str], source: str) -> dict[str, str]:
    if source != "hr.xiaomi.com":
        return sections
    responsibilities = sections.get("responsibilities", "")
    requirements = sections.get("requirements", "")
    if not responsibilities or not requirements:
        return sections
    normal_score = _signal_count(responsibilities, _RESPONSIBILITY_SIGNALS) + _signal_count(
        requirements, _REQUIREMENT_SIGNALS
    )
    inverted_score = _signal_count(responsibilities, _REQUIREMENT_SIGNALS) + _signal_count(
        requirements, _RESPONSIBILITY_SIGNALS
    )
    if (
        inverted_score >= normal_score + 3
        and _signal_count(responsibilities, _REQUIREMENT_SIGNALS) >= 2
        and _signal_count(requirements, _RESPONSIBILITY_SIGNALS) >= 2
    ):
        repaired = dict(sections)
        repaired["responsibilities"] = requirements
        repaired["requirements"] = responsibilities
        return repaired
    return sections


def split_sections(text: str, *, source: str = "") -> dict[str, str]:
    sections: dict[str, list[str]] = {key: [] for key in SECTION_ALIASES}
    current: str | None = None
    for line in text.splitlines():
        markers = list(SECTION_MARKER.finditer(line))
        if markers:
            prefix = line[: markers[0].start()].strip()
            if prefix and current:
                sections[current].append(prefix)
            for index, marker in enumerate(markers):
                alias = next(group for group in marker.groups() if group)
                current = SECTION_BY_ALIAS[alias]
                end = markers[index + 1].start() if index + 1 < len(markers) else len(line)
                content = line[marker.end() : end].strip()
                if content:
                    sections[current].append(content)
            continue
        normalized = re.sub(r"<[^>]+>", "", line).strip(" ：:")
        if normalized in SECTION_BY_ALIAS:
            current = SECTION_BY_ALIAS[normalized]
            continue
        if current:
            sections[current].append(line)
    parsed = {key: "\n".join(value).strip() for key, value in sections.items() if value}
    return _repair_known_source_inversion(parsed, source)
