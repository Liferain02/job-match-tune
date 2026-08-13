from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class DirectionRelation(StrEnum):
    EXACT = "exact"
    COMPATIBLE = "compatible"
    MISMATCH = "mismatch"


@dataclass(frozen=True)
class DirectionDecision:
    relation: DirectionRelation
    jd_leaf: str
    resume_leaf: str
    shared_skills: tuple[str, ...]
    evidence_type: str
    reason: str

    @property
    def matches(self) -> bool:
        return self.relation in {DirectionRelation.EXACT, DirectionRelation.COMPATIBLE}

    def as_dict(self) -> dict[str, object]:
        return {
            "关系": self.relation.value,
            "JD方向类别": self.jd_leaf,
            "简历方向类别": self.resume_leaf,
            "共享技能": list(self.shared_skills),
            "兼容证据类型": self.evidence_type,
            "判断理由": self.reason,
        }


# Specific labels precede broad labels. This is deliberately a small taxonomy
# covering directions already present in the project, not a general ontology.
DIRECTION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("algorithm_platform", ("算法平台", "ai 平台", "ai平台")),
    ("model_serving", ("模型服务", "模型 serving", "model serving", "推理工程")),
    ("ai_backend", ("ai 后端", "ai后端")),
    ("ai_infra", ("ai infra", "ai基础设施", "ai 基础设施")),
    ("ai_application", ("大模型应用", "ai应用", "ai 应用")),
    ("algorithm", ("算法工程", "算法开发", "算法")),
    ("data_engineering", ("数据开发", "数据工程")),
    ("data_science", ("数据科学", "数据分析")),
    ("test_development", ("测试开发", "测试工程")),
    ("frontend", ("前端开发", "前端")),
    ("backend", ("后端开发", "服务端", "后端")),
    ("client", ("客户端开发", "客户端")),
    ("embedded", ("嵌式开发", "嵌入式")),
    ("infrastructure", ("网络与基础设施", "运维开发", "基础设施")),
    ("security", ("安全工程", "网络安全")),
    ("hpc", ("高性能计算", "hpc")),
    ("hardware", ("硬件研发", "硬件")),
    ("automotive", ("汽车软件", "智驾研发", "自动驾驶")),
    ("product", ("产品经理", "产品")),
)

PLATFORM_LEAVES = {"algorithm_platform", "model_serving", "ai_backend", "ai_infra", "backend"}
PLATFORM_PAIRS = {
    frozenset(pair)
    for pair in (
        ("algorithm_platform", "ai_backend"),
        ("algorithm_platform", "backend"),
        ("algorithm_platform", "ai_infra"),
        ("model_serving", "ai_backend"),
        ("model_serving", "backend"),
        ("model_serving", "ai_infra"),
        ("ai_infra", "ai_backend"),
        ("ai_infra", "backend"),
        ("ai_backend", "backend"),
    )
}
PLATFORM_SIGNALS = re.compile(
    r"(?:serving|在线推理|模型部署|推理服务|平台\s*api|服务平台|模型服务|容器化|kubernetes|docker)",
    re.I,
)
APPLICATION_SIGNALS = re.compile(r"(?:rag|agent|智能体|大模型应用|llm应用|应用落地|工作流)", re.I)
DATA_PLATFORM_SIGNALS = re.compile(r"(?:数据平台|数据管道|etl|flink|spark|数仓|数据服务)", re.I)
AI_SKILLS = {"PyTorch", "TensorFlow", "CUDA", "Transformers", "RAG", "Agent"}
ENGINEERING_SKILLS = {"Python", "FastAPI", "Docker", "Kubernetes", "Linux", "Go", "Java"}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def direction_leaf(direction: str, context: str = "") -> str:
    direction_text = _normalize_text(direction)
    context_text = _normalize_text(context)
    for leaf, patterns in DIRECTION_PATTERNS:
        if any(pattern in direction_text for pattern in patterns):
            # Broad algorithm/backend labels can be refined only by concrete
            # platform evidence in the accompanying responsibilities/projects.
            if leaf in {"algorithm", "backend"} and PLATFORM_SIGNALS.search(context_text):
                if "算法" in direction_text:
                    return "algorithm_platform"
                if re.search(r"(?:模型|ai|推理)", context_text, re.I):
                    return "ai_backend"
            return leaf
    return direction_text


def _has_two_sided_signal(pattern: re.Pattern[str], jd_text: str, resume_text: str) -> bool:
    return bool(pattern.search(jd_text) and pattern.search(resume_text))


def evaluate_direction_compatibility(
    jd_direction: str,
    resume_direction: str,
    *,
    jd_context: str = "",
    resume_context: str = "",
    shared_skills: Iterable[str] = (),
) -> DirectionDecision:
    shared = tuple(dict.fromkeys(str(skill) for skill in shared_skills if str(skill).strip()))
    jd_leaf = direction_leaf(jd_direction, jd_context)
    resume_leaf = direction_leaf(resume_direction, resume_context)
    if jd_leaf and jd_leaf == resume_leaf:
        return DirectionDecision(
            DirectionRelation.EXACT,
            jd_leaf,
            resume_leaf,
            shared,
            "same_direction_leaf",
            "双方归入同一最小方向类别。",
        )
    if not jd_leaf or not resume_leaf:
        return DirectionDecision(
            DirectionRelation.MISMATCH,
            jd_leaf,
            resume_leaf,
            shared,
            "missing_direction",
            "至少一侧缺少可判断的岗位方向。",
        )

    pair = frozenset((jd_leaf, resume_leaf))
    enough_overlap = len(shared) >= 2
    combined_skills = set(shared)
    platform_skill_evidence = bool(combined_skills & AI_SKILLS) and bool(
        combined_skills & ENGINEERING_SKILLS
    )
    if (
        pair in PLATFORM_PAIRS
        and enough_overlap
        and platform_skill_evidence
        and _has_two_sided_signal(PLATFORM_SIGNALS, jd_context, resume_context)
    ):
        return DirectionDecision(
            DirectionRelation.COMPATIBLE,
            jd_leaf,
            resume_leaf,
            shared,
            "ai_platform_responsibility_and_skill_overlap",
            "方向名称不同，但双方都有模型服务/平台职责以及 AI 与工程技能交集。",
        )

    if (
        pair == frozenset(("algorithm", "ai_application"))
        and enough_overlap
        and _has_two_sided_signal(APPLICATION_SIGNALS, jd_context, resume_context)
    ):
        return DirectionDecision(
            DirectionRelation.COMPATIBLE,
            jd_leaf,
            resume_leaf,
            shared,
            "ai_application_responsibility_and_skill_overlap",
            "算法与 AI 应用边界不同，但双方均有应用落地职责和技能交集。",
        )

    if (
        pair == frozenset(("data_engineering", "backend"))
        and enough_overlap
        and _has_two_sided_signal(DATA_PLATFORM_SIGNALS, jd_context, resume_context)
    ):
        return DirectionDecision(
            DirectionRelation.COMPATIBLE,
            jd_leaf,
            resume_leaf,
            shared,
            "data_platform_responsibility_and_skill_overlap",
            "数据工程与后端边界不同，但双方均有数据平台职责和技能交集。",
        )

    return DirectionDecision(
        DirectionRelation.MISMATCH,
        jd_leaf,
        resume_leaf,
        shared,
        "insufficient_cross_role_evidence",
        "方向不同，且职责或共享技能证据不足以证明兼容。",
    )
