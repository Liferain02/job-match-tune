from jobmatch_tune.eval import run_match_eval
from jobmatch_tune.eval.run_match_eval import (
    _diagnostic,
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
