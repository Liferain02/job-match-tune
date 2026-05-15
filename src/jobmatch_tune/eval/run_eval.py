from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jobmatch_tune.eval.metrics import precision_recall_f1, text_exact_match
from jobmatch_tune.inference.postprocess_json import parse_json_output
from jobmatch_tune.utils.io import read_jsonl, write_text


LIST_FIELDS = ["核心职责", "必备技能", "加分项"]
TEXT_FIELDS = ["岗位方向", "经验要求", "学历要求"]


def _average_metric_dicts(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = scores[0].keys()
    return {key: sum(score[key] for score in scores) / len(scores) for key in keys}


def evaluate_json_outputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [parse_json_output(row.get("prediction", ""), context_text=row.get("text", "")) for row in rows]
    json_ok = sum(1 for item in parsed if item["ok"])
    list_scores = {field: [] for field in LIST_FIELDS}
    text_scores = {field: [] for field in TEXT_FIELDS}
    for row, item in zip(rows, parsed, strict=False):
        if not item["ok"]:
            continue
        gold = row.get("label", {})
        for field in LIST_FIELDS:
            pred_items = item["data"].get(field) or []
            gold_items = gold.get(field) or []
            list_scores[field].append(precision_recall_f1(pred_items, gold_items))
        for field in TEXT_FIELDS:
            text_scores[field].append(text_exact_match(item["data"].get(field, ""), gold.get(field, "")))
    return {
        "num_samples": len(rows),
        "json_valid_rate": json_ok / len(rows) if rows else 0.0,
        "field_metrics": {
            **{field: _average_metric_dicts(scores) for field, scores in list_scores.items()},
            **{
                field: {"exact_match": sum(scores) / len(scores) if scores else 0.0}
                for field, scores in text_scores.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="outputs/eval_reports/report.json")
    args = parser.parse_args()
    rows = list(read_jsonl(args.predictions))
    report = evaluate_json_outputs(rows)
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
