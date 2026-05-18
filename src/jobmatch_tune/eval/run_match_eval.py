from __future__ import annotations

import argparse
import json
from typing import Any

from jobmatch_tune.eval.metrics import precision_recall_f1, text_exact_match
from jobmatch_tune.inference.predict import predict
from jobmatch_tune.match.rule_engine import compute_match_rule_result
from jobmatch_tune.utils.io import read_jsonl, write_text


LIST_FIELDS = ["命中技能", "缺失技能"]
TEXT_FIELDS = ["匹配等级"]
BOOL_FIELDS = ["岗位方向匹配", "学历匹配", "经验匹配"]


def run_predictions(
    rows: list[dict[str, Any]],
    model_name: str,
    adapter: str | None,
    load_4bit: bool,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        jd_result = predict(model_name, "jd_parse", row["jd_text"], adapter=adapter, load_4bit=load_4bit, max_new_tokens=max_new_tokens)
        resume_result = predict(
            model_name,
            "resume_parse",
            row["resume_text"],
            adapter=adapter,
            load_4bit=load_4bit,
            max_new_tokens=max_new_tokens,
        )

        if not jd_result.get("ok") or not resume_result.get("ok"):
            results.append(
                {
                    "id": row["id"],
                    "source_type": row.get("source_type", "unknown"),
                    "label": row["label"],
                    "jd_ok": jd_result.get("ok", False),
                    "resume_ok": resume_result.get("ok", False),
                    "analysis_ok": False,
                    "rule_result": {},
                    "analysis": {},
                }
            )
            continue

        rule_result = compute_match_rule_result(
            jd_result["data"],
            resume_result["data"],
            jd_text=row["jd_text"],
            resume_text=row["resume_text"],
        )
        analysis_result = predict(
            model_name,
            "match",
            row["jd_text"],
            resume_text=row["resume_text"],
            rule_result=json.dumps(rule_result, ensure_ascii=False),
            adapter=adapter,
            load_4bit=load_4bit,
            max_new_tokens=max_new_tokens,
        )
        results.append(
            {
                "id": row["id"],
                "source_type": row.get("source_type", "unknown"),
                "label": row["label"],
                "jd_ok": jd_result.get("ok", False),
                "resume_ok": resume_result.get("ok", False),
                "analysis_ok": analysis_result.get("ok", False),
                "rule_result": rule_result,
                "analysis": analysis_result.get("data") or {},
            }
        )
    return results


def _average_metric_dicts(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = scores[0].keys()
    return {key: sum(score[key] for score in scores) / len(scores) for key in keys}


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    list_scores = {field: [] for field in LIST_FIELDS}
    text_scores = {field: [] for field in TEXT_FIELDS}
    bool_scores = {field: [] for field in BOOL_FIELDS}
    mismatch_count = 0
    rule_valid_rows = 0

    for row in rows:
        if not row.get("jd_ok") or not row.get("resume_ok"):
            mismatch_count += 1
            continue
        rule_valid_rows += 1
        pred = row.get("rule_result") or {}
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
        "analysis_json_valid_rate": sum(1 for row in rows if row.get("analysis_ok")) / len(rows) if rows else 0.0,
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


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row.get("source_type", "unknown"), []).append(row)
    return {
        "task": "match",
        "overall": evaluate_rows(rows),
        "by_source_type": {key: evaluate_rows(value) for key, value in by_source.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/match_manual_eval_seed.jsonl")
    parser.add_argument("--model", default="models/Qwen3-14B")
    parser.add_argument("--adapter", default="outputs/checkpoints/qwen3-14b-jobmatch-qlora")
    parser.add_argument("--out", default="outputs/eval_reports/match_eval_report.json")
    parser.add_argument("--predictions-out", default="outputs/eval_reports/match_eval_predictions.jsonl")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    rows = list(read_jsonl(args.dataset))
    predictions = run_predictions(rows, args.model, args.adapter, args.load_4bit, args.max_new_tokens)
    report = build_report(predictions)
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2))
    write_text(args.predictions_out, "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
