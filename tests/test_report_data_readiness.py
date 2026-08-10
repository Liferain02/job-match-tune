import json
from pathlib import Path
from unittest.mock import patch

import pytest

from jobmatch_tune.eval.report_data_readiness import (
    audit_sft_files,
    build_multitask_report,
    build_report,
    build_task_report,
    count_holdout_overlap,
    _float_or_default,
)


@pytest.fixture(autouse=True)
def _fresh_pipeline_for_readiness_unit_tests():
    with patch(
        "jobmatch_tune.eval.report_data_readiness.build_pipeline_freshness_report",
        return_value={"fresh": True, "normalization": {}, "dependencies": []},
    ):
        yield


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
            9540, 1180,  # multitask
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
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                ]
                report = build_report()
    assert report["summary"]["all_ready_for_training"] is False
    assert "match" in report["summary"]["not_ready_tasks"]
    assert report["tasks"]["jd"]["quality_profile"]["tier_counts"] == {"strict": 1}
    assert report["tasks"]["jd"]["risk_ready"] is True


def test_build_report_requires_resume_privacy_and_product_preference():
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [
            4240, 530, 530, 8000,  # jd
            10000, 1000, 1000, 3000,  # resume
            3000, 400, 400, 4500,  # match
            9540, 1180,  # multitask
        ]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            with patch("jobmatch_tune.eval.report_data_readiness.count_holdout_overlap", return_value=0):
                with patch("jobmatch_tune.eval.report_data_readiness.read_json_file") as read_json_file:
                    read_json_file.side_effect = [
                        {},
                        {},
                        {},
                        {},
                        {"profile_ready": True},
                        {"ready_for_resume_training": True},
                        {"ready_for_dpo": True, "ready_for_dpo_smoke": True},
                        {"ready_for_dpo": True, "ready_for_dpo_smoke": True},
                    ]
                    report = build_report()
    assert report["summary"]["all_ready_for_training"] is True
    assert report["summary"]["ready_for_product_dpo"] is True
    assert report["tasks"]["resume"]["privacy_ready"] is True


def test_build_report_blocks_training_when_resume_privacy_fails():
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [
            4240, 530, 530, 8000,
            10000, 1000, 1000, 3000,
            3000, 400, 400, 4500,
            9540, 1180,
        ]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            with patch("jobmatch_tune.eval.report_data_readiness.count_holdout_overlap", return_value=0):
                with patch("jobmatch_tune.eval.report_data_readiness.read_json_file") as read_json_file:
                    read_json_file.side_effect = [
                        {},
                        {},
                        {},
                        {},
                        {"profile_ready": True},
                        {"ready_for_resume_training": False},
                        {"ready_for_dpo": True, "ready_for_dpo_smoke": True},
                        {"ready_for_dpo": True, "ready_for_dpo_smoke": True},
                    ]
                    report = build_report()
    assert report["summary"]["all_ready_for_training"] is False
    assert report["summary"]["not_ready_tasks"] == ["resume"]
    assert report["tasks"]["resume"]["privacy_ready"] is False


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


def test_audit_sft_files_detects_cross_split_normalized_input_overlap(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train_row = _sample("row_1")
    valid_row = _sample("row_2")
    train_row["messages"][1]["content"] = "岗位名称：后端 开发工程师！"
    valid_row["messages"][1]["content"] = "岗位名称:后端开发工程师"
    valid_row["messages"][-1]["content"] = json.dumps(
        {
            "岗位方向": "后端开发",
            "核心职责": ["负责接口治理"],
            "必备技能": ["Python"],
            "学历要求": "本科",
            "经验要求": "3年",
        },
        ensure_ascii=False,
    )
    train.write_text(json.dumps(train_row, ensure_ascii=False) + "\n", encoding="utf-8")
    valid.write_text(json.dumps(valid_row, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = audit_sft_files("jd", [str(train), str(valid)])

    assert audit["cross_split_duplicate_hashes"] == 0
    assert audit["cross_split_normalized_input_hashes"] == 1


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
        mocked.side_effect = [9540, 1180]
        report = build_multitask_report(str(train), str(valid))

    assert report["has_required_mix"] is True
    assert report["source_diversity_ready"] is True
    assert report["cross_split_source_groups"] == 0
    assert report["ready_for_sft"] is True


def test_build_multitask_report_rejects_template_dominated_selection(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train_rows = []
    valid_rows = []
    for split, target in (("train", train_rows), ("valid", valid_rows)):
        for task in ("jd", "match"):
            row = _sample(f"{split}_{task}")
            row["messages"][1]["content"] = f"{split} {task}"
            row["meta"] = {"dataset_task": task}
            row["source_group"] = f"{split}_{task}"
            target.append(row)
        for index in range(10):
            row = _sample(f"{split}_resume_{index}")
            row["messages"][1]["content"] = f"{split} resume {index}"
            row["meta"] = {"dataset_task": "resume"}
            row["source_group"] = f"{split}_resume_same"
            target.append(row)
    train.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    valid.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid_rows),
        encoding="utf-8",
    )

    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [9540, 1180]
        report = build_multitask_report(str(train), str(valid))

    assert report["source_diversity"]["train"]["resume"]["source_group_ratio"] == 0.1
    assert report["source_diversity_ready"] is False
    assert report["ready_for_sft"] is False


def test_count_holdout_overlap_normalizes_jd_parse_suffix(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    train.write_text(json.dumps({"id": "jd_1_jd_parse"}) + "\n", encoding="utf-8")
    holdout.write_text(json.dumps({"source_id": "jd_1"}) + "\n", encoding="utf-8")

    assert count_holdout_overlap([str(train)], str(holdout)) == 1
