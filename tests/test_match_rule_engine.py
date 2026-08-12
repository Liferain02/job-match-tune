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


def test_compute_match_rule_result_accepts_structured_resume_items() -> None:
    jd_data = {
        "岗位方向": "后端开发",
        "必备技能": "Python",
        "学历要求": "本科及以上",
        "经验要求": "2年以上开发经验",
    }
    resume_data = {
        "目标岗位": "后端开发工程师",
        "教育背景": {"学历": "本科", "专业": "软件工程"},
        "核心技能": "Python",
        "实习经历": [{"公司": "示例科技", "内容": "2年 Python 服务开发"}],
        "项目经历": [{"项目": "订单平台", "内容": "负责 Python API 开发"}],
    }

    result = compute_match_rule_result(jd_data, resume_data)

    assert result["岗位方向匹配"] is True
    assert result["学历匹配"] is True
    assert result["经验匹配"] is True
    assert result["命中技能"] == ["Python"]
    assert result["命中项目"]


def test_compute_match_rule_result_understands_chinese_experience_years() -> None:
    jd_data = {
        "岗位方向": "后端开发",
        "必备技能": ["Python"],
        "学历要求": "",
        "经验要求": "三年以上开发经验",
    }
    resume_data = {
        "目标岗位": "后端开发",
        "教育背景": [],
        "核心技能": ["Python"],
        "实习经历": ["两年服务端开发经验"],
        "项目经历": [],
    }

    result = compute_match_rule_result(jd_data, resume_data, resume_text="两年后端开发经验")

    assert result["经验匹配"] is False


def test_compute_match_rule_result_counts_skill_with_project_evidence() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "后端开发", "必备技能": ["Python", "FastAPI"]},
        {
            "目标岗位": "后端开发",
            "核心技能": ["Python"],
            "项目经历": ["使用 FastAPI 开发网关服务"],
        },
    )

    assert result["命中技能"] == ["Python", "FastAPI"]
    assert result["缺失技能"] == []


def test_broad_direction_text_is_not_project_evidence() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "算法工程", "必备技能": ["Python"]},
        {
            "目标岗位": "算法工程",
            "核心技能": ["PyTorch"],
            "项目经历": ["在算法工程方向团队参与基础开发和联调工作"],
        },
    )

    assert result["命中项目"] == []


def test_compute_match_rule_result_uses_existing_skill_aliases() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "数据开发", "必备技能": ["PostgreSQL"]},
        {"目标岗位": "数据开发", "核心技能": ["Postgres"]},
    )

    assert result["命中技能"] == ["PostgreSQL"]


def test_calendar_year_is_not_treated_as_experience_duration() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "后端开发", "经验要求": "三年以上工作经验"},
        {"目标岗位": "后端开发", "教育背景": ["2019年入学，2023年本科毕业"]},
        resume_text="教育背景：2019年入学，2023年本科毕业",
    )

    assert result["经验匹配"] is False


def test_two_digit_graduation_year_is_not_experience_duration() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "后端开发", "经验要求": "三年以上工作经验"},
        {"目标岗位": "后端开发", "教育背景": ["要求26年或27年毕业的在校生"]},
        resume_text="教育背景：要求26年或27年毕业的在校生",
    )

    assert result["经验匹配"] is False


def test_two_digit_recruitment_cohort_is_not_experience_duration() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "测试开发", "经验要求": "两年以上测试经验"},
        {"目标岗位": "测试开发", "实习经历": ["参与研发测试工程师（25年校招）项目"]},
        resume_text="实习经历：参与研发测试工程师（25年校招）项目",
    )

    assert result["经验匹配"] is False


def test_short_recruitment_experience_is_still_a_duration() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "人力资源", "经验要求": "两年以上校招经验"},
        {"目标岗位": "人力资源", "实习经历": ["具有2年校招经验"]},
    )

    assert result["经验匹配"] is True


def test_experience_range_uses_lower_bound_as_requirement() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "后端开发", "经验要求": "3-7年相关工作经验"},
        {"目标岗位": "后端开发", "实习经历": ["具有4年后端开发经验"]},
    )

    assert result["经验匹配"] is True


def test_graduate_degree_requirement_is_not_treated_as_empty() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "算法工程", "学历要求": "研究生及以上学历"},
        {"目标岗位": "算法工程", "教育背景": ["本科，计算机科学"]},
    )

    assert result["学历匹配"] is False


def test_preferred_degree_is_not_treated_as_hard_requirement() -> None:
    result = compute_match_rule_result(
        {"岗位方向": "算法工程", "学历要求": "硕士优先"},
        {"目标岗位": "算法工程", "教育背景": ["本科，计算机科学"]},
    )

    assert result["学历匹配"] is True
