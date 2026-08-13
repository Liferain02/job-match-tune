from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from jobmatch_tune.preprocess.skill_canonicalization import extract_known_skills


EvidenceKind = Literal[
    "requirement_evidence",
    "responsibility_evidence",
    "bonus_evidence",
    "other_context_evidence",
]

REQUIREMENT_HEADINGS = ("任职要求", "任职资格", "岗位要求", "职位要求", "能力要求", "基本要求")
RESPONSIBILITY_HEADINGS = ("岗位职责", "职位职责", "工作职责", "核心职责", "岗位描述", "工作内容")
BONUS_HEADINGS = ("加分项", "优先条件", "优先考虑")
REQUIREMENT_CUES = re.compile(r"(?:熟悉|掌握|具备|要求|精通|经验|能力|能够|擅长|了解)", re.I)
RESPONSIBILITY_CUES = re.compile(r"(?:负责|参与|建设|开发|维护|研究|使用|完成|推进|支持)", re.I)
BONUS_CUES = re.compile(r"(?:优先|加分)", re.I)


@dataclass
class SkillEvidence:
    skill: str
    excerpts: dict[EvidenceKind, list[str]] = field(
        default_factory=lambda: {
            "requirement_evidence": [],
            "responsibility_evidence": [],
            "bonus_evidence": [],
            "other_context_evidence": [],
        }
    )

    def add(self, kind: EvidenceKind, excerpt: str) -> None:
        cleaned = re.sub(r"\s+", " ", excerpt).strip()
        if cleaned and cleaned not in self.excerpts[kind]:
            self.excerpts[kind].append(cleaned)

    @property
    def required(self) -> bool:
        return bool(self.excerpts["requirement_evidence"])

    def as_dict(self) -> dict[str, Any]:
        sources = [key for key, values in self.excerpts.items() if values]
        return {
            "技能": self.skill,
            "是否必备": self.required,
            "证据来源": sources,
            "证据片段": {key: values for key, values in self.excerpts.items() if values},
        }


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [text for nested in value.values() for text in _items(nested)]
    if isinstance(value, (list, tuple, set)):
        return [text for nested in value for text in _items(nested)]
    text = str(value).strip()
    return [text] if text else []


def _heading_kind(text: str) -> EvidenceKind | None:
    compact = re.sub(r"\s+", "", text)
    if any(compact.startswith(heading) for heading in REQUIREMENT_HEADINGS):
        return "requirement_evidence"
    if any(compact.startswith(heading) for heading in RESPONSIBILITY_HEADINGS):
        return "responsibility_evidence"
    if any(compact.startswith(heading) for heading in BONUS_HEADINGS):
        return "bonus_evidence"
    return None


def _line_kind(text: str, section: EvidenceKind | None) -> EvidenceKind:
    if section:
        return section
    if BONUS_CUES.search(text):
        return "bonus_evidence"
    if REQUIREMENT_CUES.search(text):
        return "requirement_evidence"
    if RESPONSIBILITY_CUES.search(text):
        return "responsibility_evidence"
    return "other_context_evidence"


def _context_segments(text: str) -> list[tuple[EvidenceKind, str]]:
    if not text.strip():
        return []
    heading_pattern = "|".join(
        re.escape(item) for item in (*REQUIREMENT_HEADINGS, *RESPONSIBILITY_HEADINGS, *BONUS_HEADINGS)
    )
    expanded = re.sub(rf"(?<!^)(?=({heading_pattern})\s*[：:])", "\n", text)
    current: EvidenceKind | None = None
    segments: list[tuple[EvidenceKind, str]] = []
    for raw_line in expanded.splitlines():
        line = raw_line.strip(" \t-•")
        if not line:
            continue
        heading_kind = _heading_kind(line)
        if heading_kind:
            current = heading_kind
            line = re.sub(rf"^(?:{heading_pattern})\s*[：:]?\s*", "", line).strip()
            if not line:
                continue
        segments.append((_line_kind(line, current), line))
    return segments


def collect_jd_skill_evidence(
    jd_data: dict[str, Any],
    context_text: str,
    schema: dict[str, Any],
) -> list[SkillEvidence]:
    """Collect known-skill provenance and decide which skills are hard requirements.

    A model placing a skill in ``必备技能`` makes it a candidate, not proof. The
    skill becomes required only when the parsed requirement field or original
    requirement-like text supports it.
    """

    evidence: dict[str, SkillEvidence] = {}

    def add_from(kind: EvidenceKind, excerpt: str) -> None:
        for skill in extract_known_skills(excerpt, schema):
            evidence.setdefault(skill, SkillEvidence(skill)).add(kind, excerpt)

    for line in _items(jd_data.get("任职要求") or jd_data.get("岗位要求")):
        add_from("requirement_evidence", line)
    for line in _items(jd_data.get("核心职责")):
        add_from("responsibility_evidence", line)
    for line in _items(jd_data.get("加分项")):
        add_from("bonus_evidence", line)
    for kind, line in _context_segments(context_text):
        add_from(kind, line)

    # Preserve candidates emitted by the model only when the original/parsed
    # text has some evidence. They are deliberately not promoted by this step.
    for skill in _items(jd_data.get("必备技能") or jd_data.get("核心技能")):
        for canonical in extract_known_skills(skill, schema):
            evidence.setdefault(canonical, SkillEvidence(canonical))

    return list(evidence.values())


def required_skills_from_evidence(evidence: list[SkillEvidence]) -> list[str]:
    return [item.skill for item in evidence if item.required]
