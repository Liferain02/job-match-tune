import json
from pathlib import Path

from jobmatch_tune.eval.report_preference_readiness import audit_preference_files


def _preference(row_id: str, prompt: str) -> dict:
    return {
        "id": row_id,
        "source_id": row_id,
        "prompt": prompt,
        "chosen": json.dumps({"岗位方向": "后端开发"}, ensure_ascii=False),
        "rejected": json.dumps({"岗位方向": "前端开发"}, ensure_ascii=False),
        "meta": {
            "rejection_strategy": "direction_mismatch",
            "provenance": "synthetic_structured_hard_negative",
        },
    }


def test_preference_readiness_detects_holdout_overlap(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    train.write_text(json.dumps(_preference("jd_1", "train prompt"), ensure_ascii=False) + "\n", encoding="utf-8")
    valid.write_text(json.dumps(_preference("jd_2", "valid prompt"), ensure_ascii=False) + "\n", encoding="utf-8")
    holdout.write_text(json.dumps({"id": "jd_1"}, ensure_ascii=False) + "\n", encoding="utf-8")

    report = audit_preference_files(str(train), str(valid), str(holdout))

    assert report["holdout_overlap"] == 1
    assert report["format_ready"] is False


def test_preference_readiness_detects_cross_split_prompt(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    train.write_text(json.dumps(_preference("jd_1", "same prompt"), ensure_ascii=False) + "\n", encoding="utf-8")
    valid.write_text(json.dumps(_preference("jd_2", "same prompt"), ensure_ascii=False) + "\n", encoding="utf-8")
    holdout.write_text("", encoding="utf-8")

    report = audit_preference_files(str(train), str(valid), str(holdout))

    assert report["cross_split_prompt_hashes"] == 1
    assert report["format_ready"] is False


def test_preference_readiness_accepts_conversational_format(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    row = _preference("jd_1", "train prompt")
    row["prompt"] = [{"role": "user", "content": "train prompt"}]
    row["chosen"] = [{"role": "assistant", "content": row["chosen"]}]
    row["rejected"] = [{"role": "assistant", "content": row["rejected"]}]
    train.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    valid.write_text("", encoding="utf-8")
    holdout.write_text("", encoding="utf-8")

    report = audit_preference_files(str(train), str(valid), str(holdout))

    assert report["invalid_rows"] == 0
    assert report["synthetic_preference_rate"] == 1.0
    assert report["preference_origin"] == "synthetic_only"
    assert report["preference_quality_ready"] is False
    assert report["ready_for_dpo"] is False


def test_preference_readiness_accepts_non_synthetic_quality_mix(tmp_path: Path, monkeypatch):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    rows = []
    for index in range(4):
        row = _preference(f"human_{index}", f"prompt {index}")
        row["meta"]["provenance"] = "human_preference"
        rows.append(row)
    train.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows[:3]) + "\n", encoding="utf-8")
    valid.write_text(json.dumps(rows[3], ensure_ascii=False) + "\n", encoding="utf-8")
    holdout.write_text("", encoding="utf-8")
    monkeypatch.setattr("jobmatch_tune.eval.report_preference_readiness.FULL_THRESHOLDS", {"train": 3, "valid": 1})
    monkeypatch.setattr("jobmatch_tune.eval.report_preference_readiness.MIN_NON_SYNTHETIC_PREFERENCES", 4)

    report = audit_preference_files(str(train), str(valid), str(holdout))

    assert report["preference_quality_ready"] is True
    assert report["ready_for_dpo"] is True


def test_preference_readiness_does_not_trust_missing_provenance(
    tmp_path: Path, monkeypatch
):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    rows = [_preference(f"unknown_{index}", f"prompt {index}") for index in range(4)]
    for row in rows:
        row["meta"].pop("provenance")
    train.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows[:3]) + "\n",
        encoding="utf-8",
    )
    valid.write_text(json.dumps(rows[3], ensure_ascii=False) + "\n", encoding="utf-8")
    holdout.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "jobmatch_tune.eval.report_preference_readiness.FULL_THRESHOLDS",
        {"train": 3, "valid": 1},
    )
    monkeypatch.setattr(
        "jobmatch_tune.eval.report_preference_readiness.MIN_NON_SYNTHETIC_PREFERENCES",
        4,
    )

    report = audit_preference_files(str(train), str(valid), str(holdout))

    assert report["non_synthetic_rows"] == 0
    assert report["unknown_origin_rows"] == 4
    assert report["preference_quality_ready"] is False
    assert report["ready_for_dpo"] is False
