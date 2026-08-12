from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Iterable

from jobmatch_tune.match.rule_engine import (
    _extract_education_rank,
    _extract_required_years,
    _extract_years,
)
from jobmatch_tune.utils.io import read_jsonl, write_text


def _structured_prediction(row: dict[str, Any]) -> dict[str, Any]:
    product = row.get("product_final") or {}
    return product.get("rule_result") or row.get("rule_result") or {}


def _review_item(gold: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
    label = gold.get("label") or {}
    pred = _structured_prediction(prediction or {})
    jd_text = str(gold.get("jd_text") or "")
    resume_text = str(gold.get("resume_text") or "")
    jd_years = _extract_required_years(jd_text)
    resume_years = _extract_years(resume_text)
    jd_education = _extract_education_rank(jd_text)
    resume_education = _extract_education_rank(resume_text)
    reasons: list[str] = []
    score = 0

    if jd_years:
        evidence_match = resume_years >= jd_years
        if bool(label.get("经验匹配")) != evidence_match:
            reasons.append("经验标签与原文明示年限矛盾")
            score += 4
    if jd_education:
        evidence_match = resume_education >= jd_education
        if bool(label.get("学历匹配")) != evidence_match:
            reasons.append("学历标签与原文明示学历矛盾")
            score += 4

    gold_matched = set(label.get("命中技能") or [])
    gold_missing = set(label.get("缺失技能") or [])
    pred_matched = set(pred.get("命中技能") or [])
    pred_missing = set(pred.get("缺失技能") or [])
    label_missing_but_evidenced = sorted(gold_missing & pred_matched)
    label_omitted_evidence = sorted(pred_matched - gold_matched - gold_missing)
    product_missed_gold = sorted(gold_matched & pred_missing)
    if label_missing_but_evidenced:
        reasons.append("标签判缺失但评分器找到双侧技能证据")
        score += 4
    if label_omitted_evidence:
        reasons.append("标签未覆盖评分器找到的技能证据")
        score += 3
    if product_missed_gold:
        reasons.append("评分器漏掉标签声明命中的技能")
        score += 3
    if pred and bool(pred.get("岗位方向匹配")) != bool(label.get("岗位方向匹配")):
        reasons.append("岗位方向标签与产品结果不一致")
        score += 2
    if pred and pred.get("匹配等级") != label.get("匹配等级"):
        reasons.append("匹配等级标签与产品结果不一致")
        score += 1
    if prediction is None:
        reasons.append("缺少对应预测结果")
        score += 2

    return {
        "id": gold.get("id"),
        "priority_score": score,
        "review_reasons": reasons,
        "annotation_status": (gold.get("meta") or {}).get("annotation_status", "missing"),
        "difficulty_tags": (gold.get("meta") or {}).get("difficulty_tags") or [],
        "evidence": {
            "jd_required_years": jd_years,
            "resume_explicit_years": resume_years,
            "jd_education_rank": jd_education,
            "resume_education_rank": resume_education,
            "label_missing_but_evidenced": label_missing_but_evidenced,
            "label_omitted_evidence": label_omitted_evidence,
            "product_missed_gold": product_missed_gold,
        },
        "label": label,
        "product_prediction": pred,
        "jd_text": jd_text,
        "resume_text": resume_text,
    }


def build_match_review_queue(
    gold_rows: Iterable[dict[str, Any]],
    prediction_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    predictions = {str(row.get("id") or ""): row for row in prediction_rows}
    items = [
        _review_item(row, predictions.get(str(row.get("id") or "")))
        for row in gold_rows
    ]
    items.sort(key=lambda item: (-item["priority_score"], str(item["id"])))
    reason_counts: Counter[str] = Counter(
        reason for item in items for reason in item["review_reasons"]
    )
    return {
        "total_rows": len(items),
        "rows_with_consistency_warnings": sum(bool(item["review_reasons"]) for item in items),
        "reason_counts": dict(reason_counts),
        "review_order": [item["id"] for item in items],
        "items": items,
        "warning": "本报告只排列人工审核优先级，不会自动修改标签或将候选升级为 Gold。",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/private/match_gold.jsonl")
    parser.add_argument(
        "--predictions",
        default="outputs/eval_reports/match_candidate_three_layer_after_skill_fix_predictions.jsonl",
    )
    parser.add_argument(
        "--out",
        default="outputs/eval_reports/match_gold_review_queue.json",
    )
    args = parser.parse_args()

    report = build_match_review_queue(
        read_jsonl(args.gold),
        read_jsonl(args.predictions),
    )
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
