from __future__ import annotations

from jobmatch_tune.match.rule_engine import compute_match_rule_result


def test_compute_match_rule_result_for_strong_candidate() -> None:
    jd_data = {
        "岗位方向": "AI应用开发",
        "必备技能": ["Python", "FastAPI", "RAG"],
        "学历要求": "本科及以上",
        "经验要求": "3年以上开发经验",
    }
    resume_data = {
        "目标岗位": "AI应用开发工程师",
        "教育背景": ["本科，计算机科学与技术"],
        "核心技能": ["Python", "FastAPI", "RAG", "MySQL"],
        "项目经历": ["负责企业知识库问答系统开发，完成 RAG 链路、FastAPI 接口和 Python 服务开发"],
        "实习经历": ["3年后端与 AI 应用开发经验"],
    }
    result = compute_match_rule_result(
        jd_data,
        resume_data,
        resume_text="本科，3年开发经验，技能包括 Python / FastAPI / RAG",
    )
    assert result["岗位方向匹配"] is True
    assert result["学历匹配"] is True
    assert result["经验匹配"] is True
    assert result["命中技能"] == ["Python", "FastAPI", "RAG"]
    assert result["缺失技能"] == []
    assert result["匹配分数"] >= 80


def test_compute_match_rule_result_for_gap_candidate() -> None:
    jd_data = {
        "岗位方向": "后端开发",
        "必备技能": ["Java", "MySQL", "Redis"],
        "学历要求": "本科及以上",
        "经验要求": "5年以上开发经验",
    }
    resume_data = {
        "目标岗位": "测试开发工程师",
        "教育背景": ["大专，软件技术"],
        "核心技能": ["Python"],
        "项目经历": ["负责自动化测试脚本开发"],
        "实习经历": ["2年测试经验"],
    }
    result = compute_match_rule_result(
        jd_data,
        resume_data,
        resume_text="大专，2年测试经验，掌握 Python",
    )
    assert result["岗位方向匹配"] is False
    assert result["学历匹配"] is False
    assert result["经验匹配"] is False
    assert result["命中技能"] == []
    assert set(result["缺失技能"]) == {"Java", "MySQL", "Redis"}
    assert result["匹配分数"] < 45
