from __future__ import annotations

import pytest

from jobmatch_tune.preprocess.jd_skill_evidence import (
    collect_jd_skill_evidence,
    required_skills_from_evidence,
)


SCHEMA = {
    "skill_alias": {
        "Python": ["python"],
        "FastAPI": ["fastapi"],
        "Agent": ["agent", "智能体"],
        "Docker": ["docker"],
        "Kafka": ["kafka"],
        "PyTorch": ["pytorch"],
    }
}


@pytest.mark.parametrize(
    "text",
    [
        "岗位职责：负责 Agent 应用开发",
        "工作内容：参与 FastAPI 平台建设",
        "核心职责：维护 Kafka 集群",
        "岗位描述：研究 PyTorch 模型结构",
        "使用 Docker 完成服务部署",
        "项目背景包含 Python 数据处理",
    ],
)
def test_responsibility_or_context_skill_is_not_promoted_to_required(text: str) -> None:
    items = collect_jd_skill_evidence({"必备技能": ["Agent", "Python", "FastAPI"]}, text, SCHEMA)
    assert required_skills_from_evidence(items) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("任职要求：熟悉 Python", "Python"),
        ("任职资格：掌握 FastAPI", "FastAPI"),
        ("岗位要求：具备 Kafka 使用经验", "Kafka"),
        ("职位要求：精通 PyTorch", "PyTorch"),
        ("需要能够使用 Docker", "Docker"),
        ("了解 Agent 开发模式", "Agent"),
    ],
)
def test_requirement_cues_create_required_evidence(text: str, expected: str) -> None:
    items = collect_jd_skill_evidence({}, text, SCHEMA)
    assert required_skills_from_evidence(items) == [expected]


def test_same_skill_in_responsibility_and_requirement_is_required_with_both_sources() -> None:
    items = collect_jd_skill_evidence(
        {},
        "岗位职责：负责 FastAPI 服务开发\n任职要求：熟悉 FastAPI 和 Python",
        SCHEMA,
    )
    by_skill = {item.skill: item for item in items}
    assert set(required_skills_from_evidence(items)) == {"Python", "FastAPI"}
    assert by_skill["FastAPI"].excerpts["responsibility_evidence"]
    assert by_skill["FastAPI"].excerpts["requirement_evidence"]


def test_bonus_and_preferred_experience_are_not_hard_requirements() -> None:
    items = collect_jd_skill_evidence({}, "加分项：有 Kafka 经验优先\n熟悉 Docker 者优先", SCHEMA)
    assert required_skills_from_evidence(items) == []
    assert all(item.excerpts["bonus_evidence"] for item in items)


def test_parsed_requirement_field_is_authoritative_even_without_raw_heading() -> None:
    items = collect_jd_skill_evidence(
        {"任职要求": ["掌握 Python", "具备 FastAPI 开发能力"]},
        "",
        SCHEMA,
    )
    assert required_skills_from_evidence(items) == ["Python", "FastAPI"]
