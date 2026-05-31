import json
from pathlib import Path
from unittest.mock import patch

from jobmatch_tune.eval.report_data_readiness import audit_sft_files, build_report, build_task_report


def test_build_task_report_not_ready_when_pool_missing():
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [5000, 500, 500, 0]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            report = build_task_report("jd", "a", "b", "c", "d")
    assert report["counts"]["combined_pool"] == 0
    assert report["ready_for_sft"] is False


def test_build_report_summarizes_not_ready_tasks():
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [
            4000, 500, 500, 8000,  # jd
            2000, 200, 200, 3000,  # resume
            97, 12, 19, 0,  # match
        ]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            report = build_report()
    assert report["summary"]["all_ready_for_training"] is False
    assert "match" in report["summary"]["not_ready_tasks"]


def _sample(row_id: str) -> dict:
    return {
        "id": row_id,
        "task_type": "jd_parse",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "岗位名称：后端开发工程师"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "岗位方向": "后端开发",
                        "核心职责": ["负责服务开发"],
                        "必备技能": ["Java"],
                        "学历要求": "本科",
                        "经验要求": "3年",
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }


def test_audit_sft_files_detects_cross_split_content_overlap(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    row = _sample("row_1")
    train.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    row["id"] = "row_2"
    valid.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = audit_sft_files("jd", [str(train), str(valid)])

    assert audit["cross_split_duplicate_hashes"] == 1
