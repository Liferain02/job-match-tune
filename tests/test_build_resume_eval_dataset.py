from jobmatch_tune.eval.build_resume_eval_dataset import BASE_ROWS, build_ocr_like_rows, to_ocr_like


def test_to_ocr_like_changes_text_shape():
    text = "教育背景：本科，计算机科学与技术。\n核心技能：Python、MySQL、AI "
    converted = to_ocr_like(text)
    assert "：" not in converted
    assert "，" not in converted
    assert "My SOL" in converted


def test_build_ocr_like_rows_preserves_labels():
    rows = build_ocr_like_rows(BASE_ROWS[:1])
    assert len(rows) == 1
    assert rows[0]["task"] == "resume_parse"
    assert rows[0]["label"] == BASE_ROWS[0]["label"]
    assert rows[0]["source_type"] == "ocr_like"
