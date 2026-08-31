from jobmatch_tune.eval import run_match_eval
from jobmatch_tune.eval.run_match_eval import (
    _diagnostic,
    align_saved_predictions,
    build_report,
    classify_errors,
    evaluate_rows,
    explanation_contradictions,
)


def test_evaluate_rows_basic():
    rows = [
        {
            "id": "match1",
            "source_type": "text",
            "jd_ok": True,
            "resume_ok": True,
            "analysis_ok": True,
            "rule_result": {
                "匹配等级": "高匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Python", "RAG"],
                "缺失技能": [],
            },
            "label": {
                "匹配等级": "高匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Python", "RAG"],
                "缺失技能": [],
            },
        }
    ]
    report = evaluate_rows(rows)
    assert report["jd_resume_parse_success_rate"] == 1.0
    assert report["analysis_json_valid_rate"] == 1.0
    assert report["field_metrics"]["匹配等级"]["exact_match"] == 1.0
    assert report["decision_metrics"]["macro_f1"] == 1.0
    assert report["decision_confidence_intervals"]["macro_f1"]["value"] == 1.0


def test_build_report_groups_by_source():
    rows = [
        {
            "id": "a",
            "source_type": "text",
            "jd_ok": True,
            "resume_ok": True,
            "analysis_ok": True,
            "rule_result": {
                "匹配等级": "高匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Python"],
                "缺失技能": [],
            },
            "meta": {"difficulty_tags": ["技能同义词"]},
            "label": {
                "匹配等级": "高匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Python"],
                "缺失技能": [],
            },
        },
        {
            "id": "b",
            "source_type": "ocr_like",
            "jd_ok": True,
            "resume_ok": True,
            "analysis_ok": False,
            "rule_result": {
                "匹配等级": "较匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": False,
                "命中技能": ["Python"],
                "缺失技能": ["MySQL"],
            },
            "meta": {"difficulty_tags": ["OCR噪声"]},
            "label": {
                "匹配等级": "较匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": False,
                "命中技能": ["Python"],
                "缺失技能": ["MySQL"],
            },
        },
    ]
    report = build_report(rows)
    assert report["overall"]["num_samples"] == 2
    assert "text" in report["by_source_type"]
    assert "ocr_like" in report["by_source_type"]
    assert report["dataset_profile"]["difficulty_tag_counts"] == {
        "技能同义词": 1,
        "OCR噪声": 1,
    }
    assert report["by_difficulty_tag"]["OCR噪声"]["num_samples"] == 1


def test_run_predictions_reuses_loaded_model(monkeypatch):
    load_calls = []
    predict_calls = []

    def fake_load_model(*args):
        load_calls.append(args)
        return "tokenizer", "model"

    def fake_predict_loaded(tokenizer, model, task, text, **kwargs):
        predict_calls.append((tokenizer, model, task, text, kwargs))
        return {
            "ok": True,
            "data": {"task": task},
            "raw_json_ok": True,
            "raw_data": {"task": task},
            "raw_output": "{}",
        }

    monkeypatch.setattr(run_match_eval, "load_model", fake_load_model)
    monkeypatch.setattr(run_match_eval, "predict_loaded", fake_predict_loaded)
    monkeypatch.setattr(run_match_eval, "compute_match_rule_result", lambda *args, **kwargs: {"匹配等级": "高匹配"})

    rows = [
        {
            "id": "match1",
            "source_type": "text",
            "jd_text": "Python 后端工程师",
            "resume_text": "熟悉 Python",
            "label": {},
        }
    ]
    predictions = run_match_eval.run_predictions(rows, "model", "adapter", True, 64)

    assert len(load_calls) == 1
    assert [call[2] for call in predict_calls] == ["jd_parse", "resume_parse", "match"]
    assert predictions[0]["analysis_ok"] is True
    assert predictions[0]["raw_model"]["structured_result_available"] is True
    assert predictions[0]["raw_model"]["jd_raw_output"] == "{}"
    assert predictions[0]["raw_model"]["resume_raw_output"] == "{}"
    assert predictions[0]["normalized"]["structured_result_available"] is True
    assert predictions[0]["product_final"]["analysis_ok"] is True


def test_diagnostic_keeps_failed_generation_details():
    assert _diagnostic({"ok": True, "raw_output": "{}"}) == {}
    assert _diagnostic({"ok": False, "error": "invalid json", "raw_output": "bad"}) == {
        "error": "invalid json",
        "raw_output": "bad",
    }


def test_align_saved_predictions_uses_current_gold_metadata():
    dataset = [
        {
            "id": "a",
            "jd_text": "JD",
            "resume_text": "RESUME",
            "label": {"匹配等级": "低匹配"},
            "meta": {"annotation_status": "human_verified"},
        }
    ]
    saved = [
        {
            "id": "a",
            "jd_text": "JD",
            "resume_text": "RESUME",
            "label": {"匹配等级": "高匹配"},
            "rule_result": {"匹配等级": "较匹配"},
        }
    ]

    aligned = align_saved_predictions(dataset, saved)

    assert aligned[0]["label"] == {"匹配等级": "低匹配"}
    assert aligned[0]["rule_result"] == {"匹配等级": "较匹配"}


def test_report_separates_three_layers_and_ranks_errors():
    row = {
        "id": "gold-1",
        "source_type": "text",
        "jd_ok": True,
        "resume_ok": True,
        "analysis_ok": True,
        "raw_model": {"structured_result_available": True, "rule_result": {"匹配等级": "低匹配"}},
        "normalized": {"structured_result_available": True, "rule_result": {"匹配等级": "高匹配"}},
        "product_final": {
            "rule_result": {
                "匹配等级": "高匹配",
                "岗位方向匹配": True,
                "学历匹配": True,
                "经验匹配": True,
                "命中技能": ["Python"],
                "缺失技能": [],
            },
            "analysis_ok": True,
            "analysis": {"匹配结论": "候选人与岗位高度匹配", "匹配优势": [], "主要短板": []},
        },
        "rule_result": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python"],
            "缺失技能": [],
        },
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python", "PostgreSQL"],
            "缺失技能": [],
        },
        "meta": {"difficulty_tags": ["技能同义词"]},
    }

    report = build_report([row])

    assert report["evaluation_validity"] == "provisional_candidate_diagnosis"
    assert report["layers"]["raw_model_derived"]["field_metrics"]["匹配等级"]["exact_match"] == 0.0
    assert report["layers"]["normalized"]["field_metrics"]["匹配等级"]["exact_match"] == 1.0
    assert report["error_analysis"]["counts"]["技能漏召回"] == 1
    assert report["error_analysis"]["counts"]["同义表达错误"] == 1
    assert report["layers"]["product_final"]["explanation_structural_consistency_rate"] == 1.0
    assert report["layers"]["product_final"]["explanation_evidence_grounding_rate"] == 1.0
    assert report["layers"]["product_final"]["advice_validity"]["status"] == "not_evaluated"


def test_explanation_contradiction_is_classified():
    row = {
        "product_final": {
            "rule_result": {"岗位方向匹配": False, "缺失技能": ["Java"], "匹配等级": "低匹配"},
            "analysis_ok": True,
            "analysis": {
                "匹配结论": "候选人与岗位高度匹配",
                "匹配优势": ["求职方向与岗位方向一致"],
                "主要短板": ["暂无明显硬性短板"],
            },
        },
        "raw_model": {"structured_result_available": True},
        "label": {},
    }

    assert explanation_contradictions(row)
    assert "模型解释与结构矛盾" in classify_errors(row)


def test_inspected_gold_report_is_labeled_as_regression() -> None:
    row = {
        "id": "gold-v1",
        "jd_ok": True,
        "resume_ok": True,
        "analysis_ok": True,
        "rule_result": {},
        "label": {},
        "meta": {"annotation_status": "human_verified"},
    }
    report = build_report([row], evaluation_context="historical_gold_v1_regression")
    assert report["evaluation_validity"] == "historical_gold_v1_regression"
    assert "REGRESSION AFTER INSPECTION" in report["warning"]


def test_blind_holdout_requires_human_verified_rows() -> None:
    row = {
        "id": "blind-1",
        "jd_ok": True,
        "resume_ok": True,
        "analysis_ok": True,
        "rule_result": {},
        "label": {},
        "meta": {
            "annotation_status": "human_verified",
            "evaluation_role": "blind_holdout",
            "inspection_status": "unseen",
            "difficulty_tags": ["OCR噪声"],
        },
    }

    report = build_report([row], evaluation_context="blind_holdout")

    assert report["evaluation_validity"] == "blind_holdout"


def test_blind_holdout_rejects_inspected_rows() -> None:
    row = {
        "id": "blind-1",
        "jd_ok": True,
        "resume_ok": True,
        "analysis_ok": True,
        "rule_result": {},
        "label": {},
        "meta": {
            "annotation_status": "human_verified",
            "evaluation_role": "blind_holdout",
            "inspection_status": "inspected",
        },
    }

    report = build_report([row], evaluation_context="blind_holdout")

    assert report["evaluation_validity"] == "invalid_blind_holdout"


def test_parse_failure_is_not_skipped_from_end_to_end_metrics() -> None:
    row = {
        "id": "failed",
        "jd_ok": False,
        "resume_ok": False,
        "analysis_ok": False,
        "rule_result": {},
        "label": {
            "匹配等级": "低匹配",
            "岗位方向匹配": False,
            "学历匹配": False,
            "经验匹配": False,
            "命中技能": [],
            "缺失技能": ["Python"],
        },
    }

    report = evaluate_rows([row])

    assert report["field_metrics"]["匹配等级"]["exact_match"] == 0.0
    assert report["field_metrics"]["命中技能"]["f1"] == 0.0
    assert report["decision_metrics"]["accuracy"] == 0.0
    assert report["decision_metrics"]["confusion_matrix"]["低匹配"]["__invalid__"] == 1
