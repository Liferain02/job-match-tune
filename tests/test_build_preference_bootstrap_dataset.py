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
    assert json.loads(built["chosen"]) != json.loads(built["rejected"])
