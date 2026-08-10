import json

from jobmatch_tune.eval.audit_jd_direction_conflicts import audit_direction_conflicts


def _row(row_id: str, title: str, direction: str) -> dict:
    return {
        "id": row_id,
        "meta": {"quality_tier": "quality_weak", "source": "example"},
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"岗位名称：{title}\n岗位职责：负责研发"},
            {"role": "assistant", "content": json.dumps({"岗位方向": direction}, ensure_ascii=False)},
        ],
    }


def test_audit_direction_conflicts_reports_review_candidates() -> None:
    rows = [
        _row("correct", "网络安全工程师", "安全工程"),
        _row("conflict", "网络安全研究岗", "网络与基础设施"),
        _row("ambiguous", "嵌入式测试工程师", "测试开发"),
    ]

    report = audit_direction_conflicts(rows)

    assert report["single_strong_title_signal_rows"] == 2
    assert report["conflict_rows"] == 1
    assert report["samples"][0]["id"] == "conflict"
    assert report["interpretation"] == "review_candidates_not_automatic_errors"
