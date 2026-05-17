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
        "label": {"岗位方向": "后端开发", "核心职责": [], "必备技能": [], "加分项": [], "经验要求": "", "学历要求": ""},
        "parsed": {"岗位方向": "AI应用开发"},
        "prediction": "{\"岗位方向\":\"AI应用开发\"}",
    }
    built = build_preference_row(row)
    assert built is not None
    assert built["id"] == "sample_1"
    assert built["task_type"] == "jd_parse"
    assert json.loads(built["chosen"])["岗位方向"] == "后端开发"
    assert json.loads(built["rejected"])["岗位方向"] == "AI应用开发"


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
