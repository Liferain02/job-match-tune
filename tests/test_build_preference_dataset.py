import json

from jobmatch_tune.dataset.build_preference_dataset import build_preference_row, build_prompt_text


def test_build_prompt_text_for_jd_parse():
    prompt = build_prompt_text("jd_parse", "公司：腾讯\n岗位名称：后端开发工程师")
    assert "招聘文本解析助手" in prompt
    assert "请解析以下招聘 JD" in prompt


def test_build_preference_row_uses_gold_and_prediction():
    row = {
        "id": "sample_1",
        "task": "jd_parse",
        "text": "公司：腾讯\n岗位名称：后端开发工程师",
        "label": {
            "岗位方向": "后端开发",
            "核心职责": [],
            "必备技能": [],
            "加分项": [],
            "经验要求": "",
            "学历要求": "",
        },
        "parsed": {"岗位方向": "AI应用开发"},
        "prediction": "{\"岗位方向\":\"AI应用开发\"}",
    }
    built = build_preference_row(row)
    assert built is not None
    assert built["id"] == "sample_1"
    assert built["task_type"] == "jd_parse"
    assert built["prompt"][0]["role"] == "system"
    assert json.loads(built["chosen"][0]["content"])["岗位方向"] == "后端开发"
    assert json.loads(built["rejected"][0]["content"])["岗位方向"] == "AI应用开发"


def test_build_preference_row_skips_identical_outputs():
    row = {
        "id": "sample_2",
        "task": "jd_parse",
        "text": "公司：腾讯\n岗位名称：后端开发工程师",
        "label": {"岗位方向": "后端开发"},
        "parsed": {"岗位方向": "后端开发"},
        "prediction": "{\"岗位方向\":\"后端开发\"}",
    }
    assert build_preference_row(row) is None


def test_build_preference_row_skips_human_verified_evaluation_row():
    row = {
        "id": "gold_1",
        "task": "jd_parse",
        "text": "岗位名称：后端开发工程师",
        "label": {"岗位方向": "后端开发"},
        "parsed": {"岗位方向": "前端开发"},
        "meta": {"annotation_status": "human_verified"},
    }

    assert build_preference_row(row) is None


def test_build_prompt_text_for_match():
    prompt = build_prompt_text(
        "match",
        "岗位名称：AI应用开发工程师",
        resume_text="目标岗位：AI应用开发",
        rule_result={"匹配等级": "高匹配"},
    )
    assert "岗位匹配分析" in prompt
    assert "规则评分结果" in prompt
    assert "高匹配" in prompt


def test_build_preference_row_for_match_analysis_mismatch():
    row = {
        "id": "match_1",
        "task": "match",
        "jd_text": "岗位名称：AI应用开发工程师",
        "resume_text": "目标岗位：AI应用开发",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python", "RAG"],
            "缺失技能": [],
        },
        "rule_result": {"匹配等级": "较匹配", "命中技能": ["Python"], "缺失技能": ["RAG"]},
        "analysis": {"匹配结论": "候选人与岗位仅部分匹配。"},
    }

    built = build_preference_row(row)

    assert built is not None
    assert built["task_type"] == "match"
    assert built["prompt"][0]["role"] == "system"
    assert "规则评分结果" in built["prompt"][1]["content"]
    chosen = json.loads(built["chosen"][0]["content"])
    rejected = json.loads(built["rejected"][0]["content"])
    assert chosen["匹配结论"].startswith("候选人与岗位整体高度匹配")
    assert rejected["匹配结论"] == "候选人与岗位仅部分匹配。"
