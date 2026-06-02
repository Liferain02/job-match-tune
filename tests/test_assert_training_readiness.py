from jobmatch_tune.eval.assert_training_readiness import (
    assert_training_readiness,
    summarize_blockers,
)


def _report(summary_updates=None, resume_updates=None):
    summary = {
        "all_ready_for_training": True,
        "all_ready_for_sft": True,
        "ready_for_dpo": True,
        "ready_for_product_dpo": True,
        "not_ready_tasks": [],
    }
    if summary_updates:
        summary.update(summary_updates)
    resume = {"privacy_ready": True}
    if resume_updates:
        resume.update(resume_updates)
    return {"summary": summary, "tasks": {"resume": resume}}


def test_assert_training_readiness_accepts_ready_report():
    result = assert_training_readiness(_report())
    assert result["ready"] is True
    assert result["blockers"] == []


def test_assert_training_readiness_blocks_product_dpo_failure():
    report = _report({"all_ready_for_training": False, "ready_for_product_dpo": False})
    result = assert_training_readiness(report)
    assert result["ready"] is False
    assert "product preference DPO data is not ready" in result["blockers"]


def test_summarize_blockers_includes_resume_privacy_failure():
    report = _report(
        {"all_ready_for_training": False, "all_ready_for_sft": False, "not_ready_tasks": ["resume"]},
        {"privacy_ready": False, "privacy_report": {"rows_with_pii": 3}},
    )
    blockers = summarize_blockers(report)
    assert "SFT not ready; tasks=['resume']" in blockers
    assert "resume privacy gate failed; rows_with_pii=3" in blockers
