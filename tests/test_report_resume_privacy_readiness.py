from pathlib import Path

from jobmatch_tune.eval.report_resume_privacy_readiness import (
    build_resume_privacy_readiness_report,
)
from jobmatch_tune.utils.io import write_jsonl


def _sft_row(row_id: str, user_text: str) -> dict:
    return {
        "id": row_id,
        "source_group": row_id,
        "messages": [
            {"role": "system", "content": "你是招聘信息结构化助手。"},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "{}"},
        ],
    }


def test_resume_privacy_readiness_accepts_clean_sft_rows(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [_sft_row("r1", "教育背景：本科\n核心技能：Java Spring")])
    report = build_resume_privacy_readiness_report(paths=[str(path)])
    assert report["ready_for_resume_training"] is True
    assert report["row_count"] == 1
    assert report["rows_with_pii"] == 0
    assert report["pii_counts"] == {}


def test_resume_privacy_readiness_blocks_pii_in_messages(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    write_jsonl(
        path,
        [
            _sft_row("r1", "教育背景：本科\n核心技能：Java Spring"),
            _sft_row("r2", "电话：13812345678\n核心技能：Python"),
        ],
    )
    report = build_resume_privacy_readiness_report(paths=[str(path)])
    assert report["ready_for_resume_training"] is False
    assert report["row_count"] == 2
    assert report["rows_with_pii"] == 1
    assert report["pii_counts"]["phone"] == 1
    assert report["examples"][0]["id"] == "r2"
