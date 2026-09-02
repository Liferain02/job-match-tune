from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from jobmatch_tune.eval.metrics import (
    aggregate_set_metrics,
    bootstrap_confidence_interval,
    text_exact_match,
)
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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def align_saved_predictions(
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the current frozen labels after checking prediction inputs exactly."""
    predictions_by_id = {str(row.get("id") or ""): row for row in prediction_rows}
    dataset_ids = [str(row.get("id") or "") for row in dataset_rows]
    if (
        "" in predictions_by_id
        or "" in dataset_ids
        or len(predictions_by_id) != len(prediction_rows)
        or len(set(dataset_ids)) != len(dataset_ids)
    ):
        raise ValueError("dataset and predictions must have unique non-empty ids")
    if set(dataset_ids) != set(predictions_by_id):
        raise ValueError("saved prediction ids do not match the evaluation dataset")

    aligned = []
    for gold in dataset_rows:
        row_id = str(gold.get("id") or "")
        prediction = predictions_by_id[row_id]
        if str(prediction.get("text") or "") != str(gold.get("text") or ""):
            raise ValueError(f"saved prediction input differs for {row_id}: text")
        aligned.append(
            {
                **prediction,
                "task": gold.get("task", prediction.get("task", "jd_parse")),
                "source_type": gold.get("source_type", prediction.get("source_type", "unknown")),
                "source_group": gold.get("source_group", prediction.get("source_group", row_id)),
                "label": gold.get("label") or {},
                "meta": gold.get("meta") or {},
            }
        )
    return aligned


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
    row_outcomes = []
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
        row_outcomes.append(
            {
                "id": row["id"],
                "json_valid": is_valid,
                "complete_row_exact": row_exact,
            }
        )
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
    confidence_intervals = {
        "json_valid_rate": bootstrap_confidence_interval(
            row_outcomes,
            lambda samples: (
                sum(bool(item["json_valid"]) for item in samples) / len(samples)
                if samples
                else 0.0
            ),
        ),
        "complete_row_exact_match_rate": bootstrap_confidence_interval(
            row_outcomes,
            lambda samples: (
                sum(bool(item["complete_row_exact"]) for item in samples) / len(samples)
                if samples
                else 0.0
            ),
        ),
    }

    def list_field_statistic(sample_rows: list[dict[str, Any]], field: str) -> float:
        pairs = []
        for row in sample_rows:
            is_valid = bool(row["ok"])
            pred = row.get("parsed") or {}
            gold = row.get("label") or {}
            pairs.append(
                (
                    pred.get(field, []) if is_valid else [INVALID_JSON_ITEM],
                    gold.get(field, []),
                )
            )
        return float(aggregate_set_metrics(pairs)["f1"])

    def text_field_statistic(sample_rows: list[dict[str, Any]], field: str) -> float:
        scores = []
        for row in sample_rows:
            is_valid = bool(row["ok"])
            pred = row.get("parsed") or {}
            gold = row.get("label") or {}
            scores.append(
                text_exact_match(
                    pred.get(field, "") if is_valid else INVALID_JSON_ITEM,
                    gold.get(field, ""),
                )
            )
        return sum(scores) / len(scores) if scores else 0.0

    field_confidence_intervals = {
        **{
            field: {
                "metric": "row_macro_f1",
                **bootstrap_confidence_interval(
                    rows,
                    lambda samples, name=field: list_field_statistic(samples, name),
                ),
            }
            for field in field_spec["list_fields"]
        },
        **{
            field: {
                "metric": "exact_match",
                **bootstrap_confidence_interval(
                    rows,
                    lambda samples, name=field: text_field_statistic(samples, name),
                ),
            }
            for field in field_spec["text_fields"]
        },
    }

    target_field = "岗位方向" if task_name == "jd_parse" else "目标岗位"
    slice_groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "target_role": {},
        "source_type": {},
        "difficulty_tag": {},
    }
    for row in rows:
        label = row.get("label") or {}
        meta = row.get("meta") or {}
        role = str(label.get(target_field) or "").strip()
        if role:
            slice_groups["target_role"].setdefault(role, []).append(row)
        source_type = str(row.get("source_type") or meta.get("source_type") or "").strip()
        if source_type and source_type != "unknown":
            slice_groups["source_type"].setdefault(source_type, []).append(row)
        for tag in meta.get("difficulty_tags") or []:
            normalized_tag = str(tag).strip()
            if normalized_tag:
                slice_groups["difficulty_tag"].setdefault(normalized_tag, []).append(row)

    def summarize_slice(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
        sample_ids = {str(row.get("id") or "") for row in sample_rows}
        sample_outcomes = [item for item in row_outcomes if item["id"] in sample_ids]
        field_metrics: dict[str, Any] = {}
        for field in field_spec["list_fields"]:
            field_metrics[field] = {
                "f1": list_field_statistic(sample_rows, field),
            }
        for field in field_spec["text_fields"]:
            field_metrics[field] = {
                "exact_match": text_field_statistic(sample_rows, field),
            }
        return {
            "num_samples": len(sample_rows),
            "small_slice_warning": len(sample_rows) < 5,
            "json_valid_rate": (
                sum(bool(item["json_valid"]) for item in sample_outcomes) / len(sample_outcomes)
                if sample_outcomes
                else 0.0
            ),
            "complete_row_exact_match_rate": (
                sum(bool(item["complete_row_exact"]) for item in sample_outcomes)
                / len(sample_outcomes)
                if sample_outcomes
                else 0.0
            ),
            "field_metrics": field_metrics,
        }

    slice_metrics = {
        dimension: {
            name: summarize_slice(sample_rows)
            for name, sample_rows in sorted(groups.items())
        }
        for dimension, groups in slice_groups.items()
        if groups
    }
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
        "confidence_intervals": confidence_intervals,
        "field_confidence_intervals": field_confidence_intervals,
        "slice_metrics": slice_metrics,
        "metric_semantics": {
            "field_metrics": "端到端指标；JSON 失败样本按字段错误计入",
            "valid_json_only_field_metrics": "仅用于定位解析后字段质量，不作为产品主指标",
            "list_field_averaging": "同时报告逐样本宏平均与全语料 micro 指标",
            "confidence_intervals": "95% percentile bootstrap，1000 次重采样，固定 seed=42",
            "slice_metrics": "按目标技术岗位、输入载体和难例标签诊断；样本少于 5 的切片不单独下结论",
        },
        "num_mismatch_samples": len(mismatches),
        "mismatches": mismatches[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/jd_manual_eval_50.jsonl")
    parser.add_argument("--model", default="models/Qwen3-14B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default="outputs/eval_reports/manual_eval_report.json")
    parser.add_argument("--predictions-out", default="outputs/eval_reports/manual_eval_predictions.jsonl")
    parser.add_argument(
        "--predictions-in",
        help="Replay a saved prediction JSONL against the current dataset without loading a model",
    )
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--evaluation-context",
        choices=["auto", "blind_holdout", "frozen_regression"],
        default="auto",
    )
    args = parser.parse_args()

    rows = list(read_jsonl(args.dataset))
    predictions = (
        align_saved_predictions(rows, list(read_jsonl(args.predictions_in)))
        if args.predictions_in
        else run_predictions(rows, args.model, args.adapter, args.load_4bit, args.max_new_tokens)
    )
    report = evaluate_predictions(predictions, evaluation_context=args.evaluation_context)
    report["evaluation_provenance"] = {
        "dataset": args.dataset,
        "dataset_sha256": file_sha256(args.dataset),
        "model": args.model,
        "adapter": args.adapter or "",
        "load_4bit": args.load_4bit,
        "max_new_tokens": args.max_new_tokens,
        "replayed_predictions": args.predictions_in or "",
        "predictions_sha256": file_sha256(args.predictions_in) if args.predictions_in else "",
    }
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2))
    write_text(args.predictions_out, "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
