from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import torch

from jobmatch_tune.eval.run_manual_eval import evaluate_predictions
from jobmatch_tune.inference.predict import build_prompt, load_model
from jobmatch_tune.inference.postprocess_json import parse_json_output
from jobmatch_tune.resume.normalize import normalize_resume_eval_row
from jobmatch_tune.utils.io import read_jsonl, write_text


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
        normalized = normalize_resume_eval_row(row)
        messages = build_prompt("resume_parse", normalized["normalized_text"])
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
        parsed = parse_json_output(raw, context_text=normalized["normalized_text"])
        results.append(
            {
                "id": row["id"],
                "task": "resume_parse",
                "source_type": row.get("source_type", "text"),
                "text": normalized["normalized_text"],
                "raw_text": row.get("text", ""),
                "label": row["label"],
                "prediction": raw,
                "parsed": parsed.get("data"),
                "ok": parsed["ok"],
                "error": parsed.get("error"),
            }
        )
    return results


def build_report(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    overall = evaluate_predictions(predictions)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row.get("source_type", "unknown")].append(row)
    by_source_type = {key: evaluate_predictions(rows) for key, rows in grouped.items()}
    return {
        "task": "resume_parse",
        "num_samples": len(predictions),
        "overall": overall,
        "by_source_type": by_source_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval/resume_manual_eval_text_seed.jsonl")
    parser.add_argument("--model", default="models/Qwen3-14B")
    parser.add_argument("--adapter", default="outputs/checkpoints/qwen3-14b-jobmatch-qlora")
    parser.add_argument("--out", default="outputs/eval_reports/resume_pipeline_eval_report.json")
    parser.add_argument(
        "--predictions-out", default="outputs/eval_reports/resume_pipeline_eval_predictions.jsonl"
    )
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
