from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from jobmatch_tune.eval.metrics import precision_recall_f1, text_exact_match
from jobmatch_tune.inference.predict import build_prompt, load_model
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


def _average_metric_dicts(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = scores[0].keys()
    return {key: sum(score[key] for score in scores) / len(scores) for key in keys}


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
                "prediction": raw,
                "parsed": parsed.get("data"),
                "ok": parsed["ok"],
                "error": parsed.get("error"),
            }
        )
    return results


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["ok"]]
    task_names = {row.get("task", "jd_parse") for row in rows}
    if len(task_names) != 1:
        raise ValueError(f"Expected a single task dataset, got: {sorted(task_names)}")
    task_name = task_names.pop()
    if task_name not in TASK_FIELD_SPECS:
        raise ValueError(f"Unsupported task for manual eval: {task_name}")

    field_spec = TASK_FIELD_SPECS[task_name]
    list_scores = {field: [] for field in field_spec["list_fields"]}
    text_scores = {field: [] for field in field_spec["text_fields"]}
    mismatches = []
    for row in valid_rows:
        pred = row["parsed"] or {}
        gold = row["label"] or {}
        row_mismatches = {}
        for field in field_spec["list_fields"]:
            score = precision_recall_f1(pred.get(field, []), gold.get(field, []))
            list_scores[field].append(score)
            if score["f1"] < 0.999:
                row_mismatches[field] = {"pred": pred.get(field, []), "gold": gold.get(field, [])}
        for field in field_spec["text_fields"]:
            score = text_exact_match(pred.get(field, ""), gold.get(field, ""))
            text_scores[field].append(score)
            if score < 0.999:
                row_mismatches[field] = {"pred": pred.get(field, ""), "gold": gold.get(field, "")}
        if row_mismatches:
            mismatches.append({"id": row["id"], "fields": row_mismatches})

    return {
        "task": task_name,
        "num_samples": len(rows),
        "json_valid_rate": len(valid_rows) / len(rows) if rows else 0.0,
        "field_metrics": {
            **{field: _average_metric_dicts(scores) for field, scores in list_scores.items()},
            **{
                field: {"exact_match": sum(scores) / len(scores) if scores else 0.0}
                for field, scores in text_scores.items()
            },
        },
        "num_mismatch_samples": len(mismatches),
        "mismatches": mismatches[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/jd_manual_eval_50.jsonl")
    parser.add_argument("--model", default="models/Qwen3-14B")
    parser.add_argument("--adapter", default="outputs/checkpoints/qwen3-14b-jobmatch-qlora")
    parser.add_argument("--out", default="outputs/eval_reports/manual_eval_report.json")
    parser.add_argument("--predictions-out", default="outputs/eval_reports/manual_eval_predictions.jsonl")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    rows = list(read_jsonl(args.dataset))
    predictions = run_predictions(rows, args.model, args.adapter, args.load_4bit, args.max_new_tokens)
    report = evaluate_predictions(predictions)
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2))
    write_text(args.predictions_out, "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
