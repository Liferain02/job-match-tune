from jobmatch_tune.dataset.build_resume_train_pool_from_sft import build_rows


def test_build_rows_extracts_text_and_label_from_sft_messages():
    rows = [
        {
            "id": "resume_1_variant",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "请解析以下简历\n\n简历：\n目标岗位：后端开发"},
                {"role": "assistant", "content": '{"目标岗位":"后端开发","教育背景":["本科"]}'},
            ],
        }
    ]
    built = build_rows(rows)
    assert len(built) == 1
    assert built[0]["text"] == "目标岗位：后端开发"
    assert built[0]["label"]["目标岗位"] == "后端开发"
