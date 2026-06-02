import json

from jobmatch_tune.dataset.build_preference_bootstrap_dataset import build_bootstrap_preference


def test_build_bootstrap_preference_creates_distinct_rejected_answer():
    row = {
        "id": "jd_1",
        "task_type": "jd_parse",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "岗位名称：后端工程师"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "岗位方向": "后端开发",
                        "核心职责": ["负责服务开发"],
                        "必备技能": ["Java"],
                        "学历要求": "本科",
                        "经验要求": "3年",
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    built = build_bootstrap_preference(row)

    assert built["source_id"] == "jd_1"
    assert built["meta"]["provenance"] == "synthetic_structured_hard_negative"
    assert built["prompt"] == row["messages"][:-1]
    assert built["chosen"][0]["role"] == "assistant"
    assert json.loads(built["chosen"][0]["content"]) != json.loads(built["rejected"][0]["content"])


def test_build_bootstrap_preference_supports_resume_parse():
    row = {
        "id": "resume_1",
        "task_type": "resume_parse",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "简历"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "目标岗位": "AI应用开发",
                        "教育背景": ["本科，计算机"],
                        "核心技能": ["Python", "RAG"],
                        "项目经历": ["知识库问答系统。"],
                        "优势标签": ["LLM应用落地"],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    built = build_bootstrap_preference(row)

    assert built["task_type"] == "resume_parse"
    assert built["meta"]["rejection_strategy"] in {
        "unexpected_field",
        "resume_strength_drop",
        "resume_project_drop",
        "resume_education_skill_leak",
    }
    assert json.loads(built["chosen"][0]["content"]) != json.loads(built["rejected"][0]["content"])


def test_build_bootstrap_preference_supports_match():
    row = {
        "id": "match_1",
        "task_type": "match",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "JD + 简历 + 规则评分"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "匹配结论": "候选人与岗位整体较匹配。",
                        "匹配优势": ["方向一致", "已覆盖 Python"],
                        "主要短板": ["缺失 Kubernetes"],
                        "简历优化建议": ["补充 Kubernetes 项目证据"],
                        "推荐投递岗位方向": ["同方向相近岗位"],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    built = build_bootstrap_preference(row)

    assert built["task_type"] == "match"
    assert built["meta"]["rejection_strategy"].startswith("match_")
    assert json.loads(built["chosen"][0]["content"]) != json.loads(built["rejected"][0]["content"])
