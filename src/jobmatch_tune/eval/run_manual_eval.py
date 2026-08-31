from __future__ import annotations

import argparse
from collections import Counter
import json
from typing import Any

from jobmatch_tune.eval.metrics import aggregate_set_metrics, text_exact_match
from jobmatch_tune.inference.postprocess_json import parse_json_output
from jobmatch_tune.utils.io import read_jsonl, write_text


TASK_FIELD_SPECS = {
    "jd_parse": {
        "list_fields": ["核心职责", "必备技能", "加分项"],
        "text_fields": ["岗位方向", "经验要求", "学历要求"],
    },
    "resume_parse": {
        "list_fields": ["教育背景", "核心技能", "实习经历", "项目经历", "优势标签"],
        "text_fields": ["目标岗位"],
    },
}


INVALID_JSON_ITEM = "__invalid_json__"


def run_predictions(
    rows: list[dict[str, Any]],
    model_name: str,
    adapter: str | None,
    load_4bit: bool,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    from jobmatch_tune.inference.predict import build_prompt, load_model

    tokenizer, model = load_model(model_name, adapter, load_4bit)
    results = []
    for row in rows:
        messages = build_prompt(row.get("task", "jd_parse"), row["text"])
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        parsed = parse_json_output(raw, context_text=row["text"])
        results.append(
            {
                "id": row["id"],
                "task": row.get("task", "jd_parse"),
                "text": row["text"],
                "label": row["label"],
                "meta": row.get("meta") or {},
                "prediction": raw,
                "parsed": parsed.get("data"),
                "ok": parsed["ok"],
                "error": parsed.get("error"),
            }
        )
    return results


def _evaluation_validity(
    rows: list[dict[str, Any]],
    evaluation_context: str,
) -> tuple[str, str]:
    statuses = Counter(
        str((row.get("meta") or {}).get("annotation_status") or "missing")
        for row in rows
    )
    roles = Counter(
        str((row.get("meta") or {}).get("evaluation_role") or "unspecified")
        for row in rows
    )
    inspections = Counter(
        str((row.get("meta") or {}).get("inspection_status") or "unspecified")
        for row in rows
    )
    human_verified = statuses == {"human_verified": len(rows)}
    blind_eligible = bool(
        rows
        and human_verified
        and roles == {"blind_holdout": len(rows)}
        and inspections == {"unseen": len(rows)}
    )
    if evaluation_context == "blind_holdout":
        return (
            ("blind_holdout", "独立人工冻结集，首次揭盲结果可用于泛化判断。")
            if blind_eligible
            else ("invalid_blind_holdout", "Blind holdout 元数据或人工复核条件不完整。")
        )
    if evaluation_context == "frozen_regression":
        return "frozen_regression", "已查看数据，仅用于工程回归。"
    if blind_eligible:
        return "blind_holdout", "独立人工冻结集，首次揭盲结果可用于泛化判断。"
    if roles == {"frozen_regression": len(rows)}:
        return "frozen_regression", "已查看数据，仅用于工程回归。"
    return "provisional_or_unspecified", "评测上下文或人工复核状态不完整，不可声称泛化。"


def evaluate_predictions(
    rows: list[dict[str, Any]],
    *,
    evaluation_context: str = "auto",
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["ok"]]
    task_names = {row.get("task", "jd_parse") for row in rows}
    if len(task_names) != 1:
        raise ValueError(f"Expected a single task dataset, got: {sorted(task_names)}")
    task_name = task_names.pop()
    if task_name not in TASK_FIELD_SPECS:
        raise ValueError(f"Unsupported task for manual eval: {task_name}")

    field_spec = TASK_FIELD_SPECS[task_name]
    list_pairs = {field: [] for field in field_spec["list_fields"]}
    valid_list_pairs = {field: [] for field in field_spec["list_fields"]}
    text_scores = {field: [] for field in field_spec["text_fields"]}
    valid_text_scores = {field: [] for field in field_spec["text_fields"]}
    mismatches = []
    complete_row_matches = 0
    for row in rows:
        is_valid = bool(row["ok"])
        pred = row.get("parsed") or {}
        gold = row["label"] or {}
        row_mismatches = {}
        row_exact = is_valid
        for field in field_spec["list_fields"]:
            pred_items = pred.get(field, []) if is_valid else [INVALID_JSON_ITEM]
            gold_items = gold.get(field, [])
            list_pairs[field].append((pred_items, gold_items))
            if is_valid:
                valid_list_pairs[field].append((pred_items, gold_items))
            score = aggregate_set_metrics([(pred_items, gold_items)])
            if score["f1"] < 0.999:
                row_exact = False
                row_mismatches[field] = {"pred": pred_items, "gold": gold_items}
        for field in field_spec["text_fields"]:
            pred_text = pred.get(field, "") if is_valid else INVALID_JSON_ITEM
            score = text_exact_match(pred_text, gold.get(field, ""))
            text_scores[field].append(score)
            if is_valid:
                valid_text_scores[field].append(score)
            if score < 0.999:
                row_exact = False
                row_mismatches[field] = {"pred": pred_text, "gold": gold.get(field, "")}
        complete_row_matches += row_exact
        if row_mismatches:
            mismatches.append({"id": row["id"], "fields": row_mismatches})

    def render_metrics(
        pairs: dict[str, list[tuple[list[str], list[str]]]],
        exact_scores: dict[str, list[float]],
    ) -> dict[str, Any]:
        return {
            **{field: aggregate_set_metrics(scores) for field, scores in pairs.items()},
            **{
                field: {"exact_match": sum(scores) / len(scores) if scores else 0.0}
                for field, scores in exact_scores.items()
            },
        }

    validity, warning = _evaluation_validity(rows, evaluation_context)
    return {
        "task": task_name,
        "evaluation_validity": validity,
        "warning": warning,
        "num_samples": len(rows),
        "json_valid_rate": len(valid_rows) / len(rows) if rows else 0.0,
        "field_metrics": render_metrics(list_pairs, text_scores),
        "valid_json_only_field_metrics": render_metrics(valid_list_pairs, valid_text_scores),
        "complete_row_exact_match_rate": (
            complete_row_matches / len(rows) if rows else 0.0
        ),
        "metric_semantics": {
            "field_metrics": "端到端指标；JSON 失败样本按字段错误计入",
            "valid_json_only_field_metrics": "仅用于定位解析后字段质量，不作为产品主指标",
            "list_field_averaging": "同时报告逐样本宏平均与全语料 micro 指标",
        },
        "num_mismatch_samples": len(mismatches),
        "mismatches": mismatches[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/jd_manual_eval_50.jsonl")
    parser.add_argument("--model", default="models/Qwen3-14B")
    parser.add_argument("--adapter", default="outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601")
    parser.add_argument("--out", default="outputs/eval_reports/manual_eval_report.json")
    parser.add_argument("--predictions-out", default="outputs/eval_reports/manual_eval_predictions.jsonl")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--evaluation-context",
        choices=["auto", "blind_holdout", "frozen_regression"],
        default="auto",
    )
    args = parser.parse_args()

    rows = list(read_jsonl(args.dataset))
    predictions = run_predictions(rows, args.model, args.adapter, args.load_4bit, args.max_new_tokens)
    report = evaluate_predictions(predictions, evaluation_context=args.evaluation_context)
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2))
    write_text(args.predictions_out, "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
