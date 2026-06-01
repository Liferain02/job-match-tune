from jobmatch_tune.eval.build_jd_quality_review_set import build_review_rows


def _row(row_id: str, tier: str, *, quality_score: int = 100, risk_score: int = 0) -> dict:
    return {
        "id": row_id,
        "task_type": "jd_parse",
        "meta": {
            "quality_tier": tier,
            "quality_reason": f"{tier}_reason",
            "quality_score": quality_score,
            "quality_risk_score": risk_score,
            "quality_risk_reasons": ["empty_skills"] if risk_score else [],
        },
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
    assert all("quality_score" in row for row in review_rows)


def test_build_review_rows_lowest_score_prioritizes_low_quality_rows() -> None:
    rows = [
        _row("s_high", "strict", quality_score=100, risk_score=0),
        _row("s_low", "strict", quality_score=70, risk_score=3),
        _row("w_high", "quality_weak", quality_score=75, risk_score=0),
        _row("w_low", "quality_weak", quality_score=45, risk_score=3),
    ]

    review_rows = build_review_rows(rows, per_tier=1, seed=42, strategy="lowest-score")

    assert [row["id"] for row in review_rows] == ["w_low", "s_low"]
    assert review_rows[0]["quality_risk_reasons"] == ["empty_skills"]
