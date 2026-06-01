from jobmatch_tune.eval import run_match_eval
from jobmatch_tune.eval.run_match_eval import _diagnostic, build_report, evaluate_rows


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
        return {"ok": True, "data": {"task": task}}

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


def test_diagnostic_keeps_failed_generation_details():
    assert _diagnostic({"ok": True, "raw_output": "{}"}) == {}
    assert _diagnostic({"ok": False, "error": "invalid json", "raw_output": "bad"}) == {
        "error": "invalid json",
        "raw_output": "bad",
    }
