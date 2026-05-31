from jobmatch_tune.eval.build_jd_quality_review_set import build_review_rows


def _row(row_id: str, tier: str) -> dict:
    return {
        "id": row_id,
        "task_type": "jd_parse",
        "meta": {"quality_tier": tier, "quality_reason": f"{tier}_reason"},
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"prompt {row_id}"},
            {"role": "assistant", "content": "{}"},
        ],
    }


def test_build_review_rows_samples_per_quality_tier() -> None:
    rows = [_row("s1", "strict"), _row("s2", "strict"), _row("w1", "quality_weak")]

    review_rows = build_review_rows(rows, per_tier=1, seed=42)

    assert len(review_rows) == 2
    assert {row["quality_tier"] for row in review_rows} == {"strict", "quality_weak"}
    assert all("review" in row for row in review_rows)
