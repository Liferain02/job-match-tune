import json
from pathlib import Path
from unittest.mock import patch

from jobmatch_tune.eval.report_data_readiness import (
    audit_sft_files,
    build_multitask_report,
    build_report,
    build_task_report,
    _float_or_default,
)


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
            9700, 1200,  # multitask
        ]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            with patch("jobmatch_tune.eval.report_data_readiness.read_json_file") as read_json_file:
                read_json_file.side_effect = [
                    {"tier_counts": {"strict": 1}},
                    {"high_risk_rate": 0.01},
                ]
                report = build_report()
    assert report["summary"]["all_ready_for_training"] is False
    assert "match" in report["summary"]["not_ready_tasks"]
    assert report["tasks"]["jd"]["quality_profile"]["tier_counts"] == {"strict": 1}
    assert report["tasks"]["jd"]["risk_ready"] is True


def test_zero_high_risk_rate_stays_zero():
    assert _float_or_default(0.0, 1.0) == 0.0
    assert _float_or_default(None, 1.0) == 1.0


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


def test_build_multitask_report_requires_all_tasks(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train_rows = []
    valid_rows = []
    for index, task in enumerate(["jd", "resume", "match"], start=1):
        row = _sample(f"train_{index}")
        row["messages"][1]["content"] = f"岗位名称：{task} train"
        row["meta"] = {"dataset_task": task}
        train_rows.append(row)
        valid_row = _sample(f"valid_{index}")
        valid_row["messages"][1]["content"] = f"岗位名称：{task} valid"
        valid_row["meta"] = {"dataset_task": task}
        valid_rows.append(valid_row)
    train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows), encoding="utf-8")
    valid.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid_rows), encoding="utf-8")

    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [9700, 1200]
        report = build_multitask_report(str(train), str(valid))

    assert report["has_required_mix"] is True
    assert report["ready_for_sft"] is True
