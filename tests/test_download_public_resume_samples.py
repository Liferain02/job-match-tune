from jobmatch_tune.dataset.download_public_resume_samples import convert_faircv_rows, convert_resume_ner_rows


def test_convert_faircv_rows_extracts_resume_sections():
    rows = [
        {
            "metadata": {"position": "后端开发工程师"},
            "content": "### 教育背景\n- 示例大学 本科\n### 专业技能\n- Java\n### 项目经验\n- 服务治理项目",
        }
    ]

    converted = convert_faircv_rows(rows)

    assert converted[0]["label"]["目标岗位"] == "后端开发工程师"
    assert converted[0]["label"]["教育背景"] == ["示例大学 本科"]
    assert converted[0]["label"]["核心技能"] == ["Java"]
    assert converted[0]["label"]["项目经历"] == ["服务治理项目"]


def test_convert_resume_ner_rows_keeps_non_empty_tokens():
    converted = convert_resume_ner_rows(
        [
            {"tokens": ["后", "端"], "ner_tags": [1, 2]},
            {"tokens": [], "ner_tags": []},
        ]
    )

    assert converted == [{"tokens": ["后", "端"], "ner_tags": [1, 2]}]
