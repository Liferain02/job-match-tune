from unittest.mock import patch

from jobmatch_tune.eval.report_data_readiness import build_report, build_task_report


def test_build_task_report_not_ready_when_pool_missing():
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [5000, 500, 500, 0]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {"invalid_json": 0, "duplicate_ids": 0, "field_quality_ok": True}
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
            audit.return_value = {"invalid_json": 0, "duplicate_ids": 0, "field_quality_ok": True}
            report = build_report()
    assert report["summary"]["all_ready_for_training"] is False
    assert "match" in report["summary"]["not_ready_tasks"]
