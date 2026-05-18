from jobmatch_tune.eval.build_match_eval_dataset import (
    ROWS,
    build_train_pool_rows,
    build_variant_rows,
)


def test_build_variant_rows_expands_each_base_row():
    variants = build_variant_rows(ROWS[:2])
    assert len(variants) == 6
    assert variants[0]["id"].endswith("_alt")
    assert variants[1]["id"].endswith("_ocr")
    assert variants[2]["id"].endswith("_compact")


def test_build_train_pool_rows_contains_reason_variants():
    rows = ROWS[:2]
    train_pool = build_train_pool_rows(rows)
    assert len(train_pool) == 16
    assert any(row["id"].endswith("_reason") for row in train_pool)
    assert any("招聘岗位：" in row["jd_text"] for row in train_pool if row["id"].endswith("_reason"))
