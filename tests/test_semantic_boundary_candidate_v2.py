from __future__ import annotations

from jobmatch_tune.eval.audit_semantic_boundary_candidate_v2 import audit_candidates
from jobmatch_tune.eval.build_semantic_boundary_candidate_v2 import build_candidates


def test_candidate_v2_is_independent_review_material_not_gold() -> None:
    rows = build_candidates()
    report = audit_candidates(rows)
    assert len(rows) == 18
    assert report["audit_ok"] is True
    assert report["gold_ready"] is False
    assert report["evaluation_status"] == "needs_human_review"
    assert report["training_eligible_rows"] == 0


def test_candidate_v2_overlap_is_rejected() -> None:
    rows = build_candidates()
    report = audit_candidates(rows, comparison_rows=[rows[0]])
    assert report["audit_ok"] is False
    assert "candidate_pair_overlaps_comparison_set" in report["problems"]


def test_human_verified_status_cannot_be_declared_by_builder() -> None:
    rows = build_candidates()
    rows[0]["meta"]["annotation_status"] = "human_verified"
    report = audit_candidates(rows)
    assert report["audit_ok"] is False
    assert "all_rows_must_need_human_review" in report["problems"]
