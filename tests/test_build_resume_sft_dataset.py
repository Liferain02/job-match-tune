from jobmatch_tune.dataset.build_resume_sft_dataset import (
    VARIANT_BUILDERS,
    build_resume_sample,
    split_grouped_samples,
)


def _row():
    return {
        "id": "resume_eval_001",
        "text": "姓名：张三\n目标岗位：AI 应用开发工程师",
        "label": {
            "目标岗位": "AI应用开发",
            "教育背景": ["本科，计算机科学与技术"],
            "核心技能": ["Python", "RAG"],
            "实习经历": ["在平台团队实习"],
            "项目经历": ["做过知识库问答系统"],
            "优势标签": ["LLM应用落地"],
        },
    }


def test_variants_render_non_empty_text():
    row = _row()
    for _, builder in VARIANT_BUILDERS:
        assert builder(row).strip()


def test_build_resume_sample_contains_resume_prompt():
    row = _row()
    sample = build_resume_sample(row, "original", row["text"])
    assert sample["task_type"] == "resume_parse"
    assert sample["source_group"] == row["id"]
    assert "请解析以下简历" in sample["messages"][1]["content"]
    assert "目标岗位" in sample["messages"][2]["content"]


def test_split_grouped_samples_keeps_groups_together():
    samples = [
        {"id": "a_1", "source_group": "a"},
        {"id": "a_2", "source_group": "a"},
        {"id": "b_1", "source_group": "b"},
        {"id": "b_2", "source_group": "b"},
        {"id": "c_1", "source_group": "c"},
        {"id": "c_2", "source_group": "c"},
    ]
    splits = split_grouped_samples(samples, 0.67, 0.17, 42)
    memberships = {}
    for split_name, rows in splits.items():
        for row in rows:
            memberships.setdefault(row["source_group"], set()).add(split_name)
    assert memberships["a"] in ({"train"}, {"valid"}, {"test"})
    assert memberships["b"] in ({"train"}, {"valid"}, {"test"})
    assert memberships["c"] in ({"train"}, {"valid"}, {"test"})
