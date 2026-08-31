from jobmatch_tune.eval.build_match_eval_dataset import (
    CHALLENGE_ROWS,
    ROWS,
    build_gold_review_candidates,
    build_legacy_review_rows,
    build_variant_rows,
)
from jobmatch_tune.dataset.curated_match_training_data import CURATED_MATCH_TRAIN_ROWS


def test_build_variant_rows_expands_each_base_row():
    variants = build_variant_rows(ROWS[:2])
    assert len(variants) == 6
    assert variants[0]["id"].endswith("_alt")
    assert variants[1]["id"].endswith("_ocr")
    assert variants[2]["id"].endswith("_compact")


def test_build_legacy_review_rows_are_not_training_eligible():
    rows = build_legacy_review_rows(ROWS[:1])

    assert rows[0]["source_group"] == ROWS[0]["id"]
    assert rows[0]["meta"]["annotation_status"] == "legacy_unverified"
    assert rows[0]["meta"]["training_eligible"] is False
    assert rows[0]["meta"]["difficulty_tags"]


def test_gold_review_candidates_cover_requested_difficulties_without_claiming_human_review():
    rows = build_gold_review_candidates(ROWS)
    challenge_rows = rows[-len(CHALLENGE_ROWS) :]
    tags = {tag for row in rows for tag in row["meta"]["difficulty_tags"]}

    assert len(rows) == 25
    assert all(row["meta"]["annotation_status"] == "needs_human_review" for row in challenge_rows)
    assert all(row["meta"]["training_eligible"] is False for row in rows)
    assert tags == {
        "技能同义词",
        "可迁移技能",
        "相近岗位方向",
        "年限不足但项目强",
        "学历不完全满足",
        "技能仅在项目中",
        "AI算法后端交叉",
        "OCR噪声",
        "单项硬门槛不满足",
    }


def test_curated_match_training_rows_are_independent_and_transparently_fictional():
    eval_pairs = {(row["jd_text"], row["resume_text"]) for row in ROWS + CHALLENGE_ROWS}
    train_pairs = {(row["jd_text"], row["resume_text"]) for row in CURATED_MATCH_TRAIN_ROWS}

    assert len(CURATED_MATCH_TRAIN_ROWS) == 16
    assert len(train_pairs) == 16
    assert train_pairs.isdisjoint(eval_pairs)
    assert {row["meta"]["entity_split"] for row in CURATED_MATCH_TRAIN_ROWS} == {
        "train",
        "valid",
        "test",
    }
    assert all(row["meta"]["training_eligible"] is True for row in CURATED_MATCH_TRAIN_ROWS)
    assert all(
        row["meta"]["annotation_status"] == "repository_curated_unverified"
        for row in CURATED_MATCH_TRAIN_ROWS
    )
    assert all(
        row["meta"]["contains_real_person_data"] is False
        for row in CURATED_MATCH_TRAIN_ROWS
    )
    education_labels_by_split = {
        split: {
            row["label"]["学历匹配"]
            for row in CURATED_MATCH_TRAIN_ROWS
            if row["meta"]["entity_split"] == split
        }
        for split in ("train", "valid", "test")
    }
    assert all(False in labels for labels in education_labels_by_split.values())
