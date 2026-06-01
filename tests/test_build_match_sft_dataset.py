from jobmatch_tune.dataset.build_match_sft_dataset import (
    build_analysis_from_label,
    build_match_sample,
    split_grouped_samples,
)


def _row():
    return {
        "id": "match_eval_001",
        "jd_text": "岗位名称：后端开发工程师\n任职要求：熟悉 Java、MySQL、Redis。",
        "resume_text": "目标岗位：后端开发\n核心技能：Java、MySQL",
        "label": {
            "匹配等级": "较匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": False,
            "命中技能": ["Java", "MySQL"],
            "缺失技能": ["Redis"],
        },
    }


def test_build_analysis_from_label_contains_strengths_and_gaps():
    analysis = build_analysis_from_label(_row()["label"])
    assert "匹配结论" in analysis
    assert any("Java" in item for item in analysis["匹配优势"])
    assert any("Redis" in item for item in analysis["主要短板"])


def test_build_match_sample_contains_match_prompt_and_json_assistant():
    sample = build_match_sample(_row())
    assert sample["task_type"] == "match"
    assert sample["source_group"] == "match_eval"
    assert "请根据 JD、简历和规则评分结果生成岗位匹配分析" in sample["messages"][1]["content"]
    assert "固定包含匹配结论、匹配优势、主要短板、简历优化建议、推荐投递岗位方向五个字段" in sample["messages"][1]["content"]
    assert "匹配结论" in sample["messages"][2]["content"]


def test_split_grouped_samples_keeps_group_members_together():
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
