from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from jobmatch_tune.eval.metrics import precision_recall_f1, text_exact_match
from jobmatch_tune.eval.explanation_grounding import evaluate_explanation
from jobmatch_tune.inference.predict import load_model, predict_loaded
from jobmatch_tune.match.rule_engine import compute_match_rule_result
from jobmatch_tune.utils.io import read_jsonl, write_text


LIST_FIELDS = ["命中技能", "缺失技能"]
TEXT_FIELDS = ["匹配等级"]
BOOL_FIELDS = ["岗位方向匹配", "学历匹配", "经验匹配"]


def _raw_layer(
    jd_result: dict[str, Any],
    resume_result: dict[str, Any],
    *,
    jd_text: str,
    resume_text: str,
) -> dict[str, Any]:
    jd_data = jd_result.get("raw_data")
    resume_data = resume_result.get("raw_data")
    usable = bool(
        jd_result.get("raw_json_ok")
        and resume_result.get("raw_json_ok")
        and isinstance(jd_data, dict)
        and isinstance(resume_data, dict)
    )
    return {
        "jd_json_ok": bool(jd_result.get("raw_json_ok")),
        "resume_json_ok": bool(resume_result.get("raw_json_ok")),
        "jd_raw_output": jd_result.get("raw_output", ""),
        "resume_raw_output": resume_result.get("raw_output", ""),
        "jd_json_error": jd_result.get("raw_json_error", ""),
        "resume_json_error": resume_result.get("raw_json_error", ""),
        "jd_parse": jd_data,
        "resume_parse": resume_data,
        "rule_result": (
            compute_match_rule_result(
                jd_data,
                resume_data,
                jd_text=jd_text,
                resume_text=resume_text,
            )
            if usable
            else {}
        ),
        "structured_result_available": usable,
    }


def _diagnostic(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("ok"):
        return {}
    return {
        "error": result.get("error"),
        "raw_output": result.get("raw_output"),
    }


def run_predictions(
    rows: list[dict[str, Any]],
    model_name: str,
    adapter: str | None,
    load_4bit: bool,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    tokenizer, model = load_model(model_name, adapter, load_4bit)
    results = []
    for row in rows:
        jd_result = predict_loaded(tokenizer, model, "jd_parse", row["jd_text"], max_new_tokens=max_new_tokens)
        resume_result = predict_loaded(
            tokenizer,
            model,
            "resume_parse",
            row["resume_text"],
            max_new_tokens=max_new_tokens,
        )

        if not jd_result.get("ok") or not resume_result.get("ok"):
            results.append(
                {
                    "id": row["id"],
                    "source_type": row.get("source_type", "unknown"),
                    "source_group": row.get("source_group", row["id"]),
                    "meta": row.get("meta") or {},
                    "jd_text": row["jd_text"],
                    "resume_text": row["resume_text"],
                    "label": row["label"],
                    "jd_ok": jd_result.get("ok", False),
                    "resume_ok": resume_result.get("ok", False),
                    "analysis_ok": False,
                    "jd_diagnostic": _diagnostic(jd_result),
                    "resume_diagnostic": _diagnostic(resume_result),
                    "rule_result": {},
                    "analysis": {},
                    "raw_model": _raw_layer(
                        jd_result,
                        resume_result,
                        jd_text=row["jd_text"],
                        resume_text=row["resume_text"],
                    ),
                    "normalized": {
                        "jd_parse": jd_result.get("data"),
                        "resume_parse": resume_result.get("data"),
                        "rule_result": {},
                        "structured_result_available": False,
                    },
                    "product_final": {
                        "rule_result": {},
                        "analysis_ok": False,
                        "analysis": {},
                        "analysis_raw_output": "",
                    },
                }
            )
            continue

        rule_result = compute_match_rule_result(
            jd_result["data"],
            resume_result["data"],
            jd_text=row["jd_text"],
            resume_text=row["resume_text"],
        )
        analysis_result = predict_loaded(
            tokenizer,
            model,
            "match",
            row["jd_text"],
            resume_text=row["resume_text"],
            rule_result=json.dumps(rule_result, ensure_ascii=False),
            max_new_tokens=max_new_tokens,
        )
        results.append(
            {
                "id": row["id"],
                "source_type": row.get("source_type", "unknown"),
                "source_group": row.get("source_group", row["id"]),
                "meta": row.get("meta") or {},
                "jd_text": row["jd_text"],
                "resume_text": row["resume_text"],
                "label": row["label"],
                "jd_ok": jd_result.get("ok", False),
                "resume_ok": resume_result.get("ok", False),
                "analysis_ok": analysis_result.get("ok", False),
                "jd_diagnostic": _diagnostic(jd_result),
                "resume_diagnostic": _diagnostic(resume_result),
                "analysis_diagnostic": _diagnostic(analysis_result),
                "rule_result": rule_result,
                "analysis": analysis_result.get("data") or {},
                "raw_model": _raw_layer(
                    jd_result,
                    resume_result,
                    jd_text=row["jd_text"],
                    resume_text=row["resume_text"],
                ),
                "normalized": {
                    "jd_parse": jd_result["data"],
                    "resume_parse": resume_result["data"],
                    "rule_result": rule_result,
                    "structured_result_available": True,
                },
                "product_final": {
                    "rule_result": rule_result,
                    "analysis_ok": analysis_result.get("ok", False),
                    "analysis": analysis_result.get("data") or {},
                    "analysis_raw_output": analysis_result.get("raw_output", ""),
                },
            }
        )
    return results


def _average_metric_dicts(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = scores[0].keys()
    return {key: sum(score[key] for score in scores) / len(scores) for key in keys}


def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    result_path: tuple[str, ...] = ("rule_result",),
    availability_path: tuple[str, ...] | None = None,
    include_analysis: bool = True,
) -> dict[str, Any]:
    list_scores = {field: [] for field in LIST_FIELDS}
    text_scores = {field: [] for field in TEXT_FIELDS}
    bool_scores = {field: [] for field in BOOL_FIELDS}
    mismatch_count = 0
    rule_valid_rows = 0

    def nested(row: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                return default
            value = value.get(key)
        return default if value is None else value

    for row in rows:
        available = (
            bool(nested(row, availability_path, False))
            if availability_path
            else bool(row.get("jd_ok") and row.get("resume_ok"))
        )
        if not available:
            mismatch_count += 1
            continue
        rule_valid_rows += 1
        pred = nested(row, result_path, {}) or {}
        gold = row.get("label") or {}
        row_has_mismatch = False
        for field in LIST_FIELDS:
            score = precision_recall_f1(pred.get(field, []), gold.get(field, []))
            list_scores[field].append(score)
            if score["f1"] < 0.999:
                row_has_mismatch = True
        for field in TEXT_FIELDS:
            score = text_exact_match(pred.get(field, ""), gold.get(field, ""))
            text_scores[field].append(score)
            if score < 0.999:
                row_has_mismatch = True
        for field in BOOL_FIELDS:
            score = 1.0 if bool(pred.get(field)) == bool(gold.get(field)) else 0.0
            bool_scores[field].append(score)
            if score < 0.999:
                row_has_mismatch = True
        if row_has_mismatch:
            mismatch_count += 1

    return {
        "num_samples": len(rows),
        "jd_resume_parse_success_rate": rule_valid_rows / len(rows) if rows else 0.0,
        "analysis_json_valid_rate": (
            sum(1 for row in rows if row.get("analysis_ok")) / len(rows)
            if rows and include_analysis
            else None
        ),
        "field_metrics": {
            **{field: _average_metric_dicts(scores) for field, scores in list_scores.items()},
            **{
                field: {"exact_match": sum(scores) / len(scores) if scores else 0.0}
                for field, scores in text_scores.items()
            },
            **{
                field: {"exact_match": sum(scores) / len(scores) if scores else 0.0}
                for field, scores in bool_scores.items()
            },
        },
        "num_mismatch_samples": mismatch_count,
    }


def explanation_contradictions(row: dict[str, Any]) -> list[str]:
    product = row.get("product_final") or {}
    rule = product.get("rule_result") or row.get("rule_result") or {}
    analysis = product.get("analysis") or row.get("analysis") or {}
    return evaluate_explanation(rule, analysis)["structural_contradictions"]


def explanation_evaluation(row: dict[str, Any]) -> dict[str, Any]:
    product = row.get("product_final") or {}
    rule = product.get("rule_result") or row.get("rule_result") or {}
    analysis = product.get("analysis") or row.get("analysis") or {}
    return evaluate_explanation(rule, analysis)


def classify_errors(row: dict[str, Any]) -> list[str]:
    gold = row.get("label") or {}
    product = row.get("product_final") or {}
    pred = product.get("rule_result") or row.get("rule_result") or {}
    meta = row.get("meta") or {}
    tags = set(meta.get("difficulty_tags") or [])
    errors = []
    if not (row.get("raw_model") or {}).get("structured_result_available") or not product.get(
        "analysis_ok", row.get("analysis_ok", False)
    ):
        errors.append("输出格式错误")
    gold_matched = set(gold.get("命中技能") or [])
    pred_matched = set(pred.get("命中技能") or [])
    skill_mismatch = gold_matched != pred_matched or set(gold.get("缺失技能") or []) != set(
        pred.get("缺失技能") or []
    )
    if gold_matched - pred_matched:
        errors.append("技能漏召回")
    if pred_matched - gold_matched:
        errors.append("技能误识别")
    if skill_mismatch and "技能同义词" in tags:
        errors.append("同义表达错误")
    if skill_mismatch and "可迁移技能" in tags:
        errors.append("可迁移技能错误")
    if bool(pred.get("岗位方向匹配")) != bool(gold.get("岗位方向匹配")):
        errors.append("岗位方向错误")
    if pred.get("匹配等级") != gold.get("匹配等级"):
        errors.append("匹配等级错误")
    if bool(pred.get("学历匹配")) != bool(gold.get("学历匹配")):
        errors.append("学历判断错误")
    if bool(pred.get("经验匹配")) != bool(gold.get("经验匹配")):
        errors.append("经验判断错误")
    if explanation_contradictions(row):
        errors.append("模型解释与结构矛盾")
    return errors


def build_error_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    error_rows = []
    for row in rows:
        errors = classify_errors(row)
        counts.update(errors)
        if errors:
            error_rows.append({"id": row.get("id"), "errors": errors})
    ranked = [{"error_type": key, "count": count} for key, count in counts.most_common()]
    return {
        "counts": dict(counts),
        "ranked": ranked,
        "top_error_types": ranked[:3],
        "rows": error_rows,
    }


def build_report(
    rows: list[dict[str, Any]],
    *,
    evaluation_context: str = "auto",
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    annotation_statuses: Counter[str] = Counter()
    for row in rows:
        by_source.setdefault(row.get("source_type", "unknown"), []).append(row)
        annotation_statuses[str((row.get("meta") or {}).get("annotation_status") or "missing")] += 1
    human_verified = bool(rows) and annotation_statuses == {"human_verified": len(rows)}
    if evaluation_context == "historical_gold_v1_regression":
        evaluation_validity = "historical_gold_v1_regression"
        warning = (
            "REGRESSION AFTER INSPECTION：Gold V1 已被查看，本报告只用于冻结回归，"
            "不是新的 blind generalization。"
        )
    else:
        evaluation_validity = "formal_gold" if human_verified else "provisional_candidate_diagnosis"
        warning = (
            "全部标签均为人工复核，可结合独立性审计解释指标。"
            if human_verified
            else "存在未人工复核标签；以下指标仅用于候选集诊断，不是正式产品准确率。"
        )
    return {
        "task": "match",
        "evaluation_validity": evaluation_validity,
        "annotation_status_counts": dict(annotation_statuses),
        "warning": warning,
        "overall": evaluate_rows(rows),
        "layers": {
            "raw_model_derived": evaluate_rows(
                rows,
                result_path=("raw_model", "rule_result"),
                availability_path=("raw_model", "structured_result_available"),
                include_analysis=False,
            ),
            "normalized": evaluate_rows(
                rows,
                result_path=("normalized", "rule_result"),
                availability_path=("normalized", "structured_result_available"),
                include_analysis=False,
            ),
            "product_final": {
                **evaluate_rows(rows),
                "explanation_consistency_rate": (
                    sum(not explanation_contradictions(row) for row in rows) / len(rows) if rows else 0.0
                ),
                "explanation_structural_consistency_rate": (
                    sum(explanation_evaluation(row)["structural_consistent"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
                "explanation_evidence_grounding_rate": (
                    sum(explanation_evaluation(row)["evidence_grounded"] for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
                "advice_validity": {
                    "status": "not_evaluated",
                    "reason": "unsupported_by_current_data",
                    "limitation": "没有真实用户反馈或投递结果，不能评估建议是否提高求职成功率。",
                },
                "explanation_metric_semantics": {
                    "explanation_consistency_rate": "legacy alias of structural consistency",
                    "structural_consistency": "解释是否与结构化匹配结果一致",
                    "evidence_grounding": "可确定解析的技能与硬条件陈述是否有结果证据",
                    "advice_validity": "not evaluated",
                },
            },
        },
        "error_analysis": build_error_analysis(rows),
        "by_source_type": {key: evaluate_rows(value) for key, value in by_source.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/match_manual_eval_seed.jsonl")
    parser.add_argument("--model", default="models/Qwen3-14B")
    parser.add_argument("--adapter", default="outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601")
    parser.add_argument("--out", default="outputs/eval_reports/match_eval_report.json")
    parser.add_argument("--predictions-out", default="outputs/eval_reports/match_eval_predictions.jsonl")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--evaluation-context",
        choices=["auto", "historical_gold_v1_regression"],
        default="auto",
    )
    args = parser.parse_args()

    rows = list(read_jsonl(args.dataset))
    predictions = run_predictions(rows, args.model, args.adapter, args.load_4bit, args.max_new_tokens)
    evaluation_context = args.evaluation_context
    if evaluation_context == "auto" and "match_gold" in args.dataset:
        evaluation_context = "historical_gold_v1_regression"
    report = build_report(predictions, evaluation_context=evaluation_context)
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2))
    write_text(args.predictions_out, "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
