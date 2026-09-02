from jobmatch_tune.eval.build_resume_eval_dataset import (
    BASE_ROWS,
    build_ocr_like_rows,
    build_text_variant_rows,
    mark_frozen_evaluation_rows,
    to_ocr_like,
)
from jobmatch_tune.dataset.curated_resume_training_data import CURATED_RESUME_TRAIN_ROWS


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
    assert rows[0]["source_group"] == BASE_ROWS[0]["id"]


def test_format_variants_share_the_original_resume_source_group():
    rows = build_text_variant_rows(BASE_ROWS[:1])

    assert len(rows) == 3
    assert {row["source_group"] for row in rows} == {BASE_ROWS[0]["id"]}


def test_resume_training_rows_are_disjoint_from_frozen_evaluation_rows():
    evaluation_rows = mark_frozen_evaluation_rows(BASE_ROWS)

    assert {row["id"] for row in evaluation_rows}.isdisjoint(
        {row["id"] for row in CURATED_RESUME_TRAIN_ROWS}
    )
    assert all(row["meta"]["training_eligible"] is False for row in evaluation_rows)
    assert all(
        row["source_type"] == "curated_fictional"
        and row["meta"]["contains_real_person_data"] is False
        for row in CURATED_RESUME_TRAIN_ROWS
    )


def test_default_technical_eval_scope_excludes_product_management():
    technical_rows = [
        row for row in BASE_ROWS if row.get("label", {}).get("目标岗位") != "产品经理"
    ]

    assert len(technical_rows) == 30
    assert all(row["label"]["目标岗位"] != "产品经理" for row in technical_rows)
