from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
import json
from typing import Any

from jobmatch_tune.inference.predict import load_model, predict_loaded
from jobmatch_tune.match.rule_engine import compute_match_rule_result
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


ParseText = Callable[[str, str], dict[str, Any]]


def generate_predictions(
    label_rows: Iterable[dict[str, Any]],
    job_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
    parse_text: ParseText,
) -> list[dict[str, Any]]:
    labels = list(label_rows)
    jobs = {str(row.get("query_id") or ""): row for row in job_rows}
    candidates = {str(row.get("candidate_id") or ""): row for row in candidate_rows}
    query_ids = sorted({str(row.get("query_id") or "") for row in labels})
    candidate_ids = sorted({str(row.get("candidate_id") or "") for row in labels})
    if any(not value for value in (*query_ids, *candidate_ids)):
        raise ValueError("ranking labels require query_id and candidate_id")
    missing_jobs = sorted(set(query_ids) - set(jobs))
    missing_candidates = sorted(set(candidate_ids) - set(candidates))
    if missing_jobs or missing_candidates:
        raise ValueError(
            f"ranking records missing: jobs={missing_jobs[:5]} candidates={missing_candidates[:5]}"
        )

    parsed_jobs = {
        query_id: parse_text("jd_parse", str(jobs[query_id].get("text") or ""))
        for query_id in query_ids
    }
    parsed_candidates = {
        candidate_id: parse_text(
            "resume_parse", str(candidates[candidate_id].get("text") or "")
        )
        for candidate_id in candidate_ids
    }
    predictions = []
    for row in labels:
        query_id = str(row["query_id"])
        candidate_id = str(row["candidate_id"])
        jd_result = parsed_jobs[query_id]
        resume_result = parsed_candidates[candidate_id]
        parse_ok = bool(jd_result.get("ok") and resume_result.get("ok"))
        rule_result = (
            compute_match_rule_result(
                jd_result["data"],
                resume_result["data"],
                jd_text=str(jobs[query_id].get("text") or ""),
                resume_text=str(candidates[candidate_id].get("text") or ""),
            )
            if parse_ok
            else {}
        )
        predictions.append(
            {
                "query_id": query_id,
                "candidate_id": candidate_id,
                "predicted_score": float(rule_result.get("匹配分数", -1)),
                "baseline": "jobmatch_parse_plus_rule_score",
                "meta": {
                    "parse_ok": parse_ok,
                    "match_level": str(rule_result.get("匹配等级") or ""),
                },
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ranking scores with the current parser and match rule engine"
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--model", default="models/Qwen3-14B")
    parser.add_argument("--adapter", default="outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601")
    parser.add_argument("--out", required=True)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.adapter or None, args.load_4bit)

    def parse_text(task: str, text: str) -> dict[str, Any]:
        return predict_loaded(
            tokenizer,
            model,
            task,
            text,
            max_new_tokens=args.max_new_tokens,
        )

    predictions = generate_predictions(
        read_jsonl(args.labels),
        read_jsonl(args.jobs),
        read_jsonl(args.candidates),
        parse_text,
    )
    write_jsonl(args.out, predictions)
    summary = {
        "pairs": len(predictions),
        "parse_failures": sum(not row["meta"]["parse_ok"] for row in predictions),
        "output": args.out,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

