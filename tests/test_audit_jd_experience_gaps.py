import json

from jobmatch_tune.eval.audit_jd_experience_gaps import audit_experience_gaps


def _row(row_id: str, experience: str, text: str) -> dict:
    return {
        "id": row_id,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"岗位名称：后端工程师\n{text}"},
            {"role": "assistant", "content": json.dumps({"经验要求": experience}, ensure_ascii=False)},
        ],
        "meta": {"quality_tier": "strict", "source": "example"},
    }


def test_audit_experience_gaps_only_reports_recoverable_empty_labels() -> None:
    rows = [
        _row("gap", "", "任职要求：具备 3 年以上后端开发经验"),
        _row("legitimate_empty", "", "任职要求：熟悉 Python"),
        _row("already_labeled", "3年以上", "具备 3 年以上后端开发经验"),
    ]

    report = audit_experience_gaps(rows)

    assert report["empty_experience_rows"] == 2
    assert report["recoverable_empty_rows"] == 1
    assert report["samples"][0]["id"] == "gap"
