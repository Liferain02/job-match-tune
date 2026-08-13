from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from jobmatch_tune.eval.run_match_eval import build_report
from jobmatch_tune.match.rule_engine import compute_match_rule_result
from jobmatch_tune.utils.io import read_jsonl, write_text


def replay_current_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replayed: list[dict[str, Any]] = []
    for row in rows:
        normalized = row.get("normalized") or {}
        jd_data = normalized.get("jd_parse")
        resume_data = normalized.get("resume_parse")
        available = bool(
            normalized.get("structured_result_available")
            and isinstance(jd_data, dict)
            and isinstance(resume_data, dict)
        )
        rule_result = (
            compute_match_rule_result(
                jd_data,
                resume_data,
                jd_text=str(row.get("jd_text") or ""),
                resume_text=str(row.get("resume_text") or ""),
            )
            if available
            else {}
        )
        product = {**(row.get("product_final") or {}), "rule_result": rule_result}
        replayed.append(
            {
                **row,
                "rule_result": rule_result,
                "normalized": {**normalized, "rule_result": rule_result},
                "product_final": product,
                "regression_replay": {
                    "current_rules_recomputed": available,
                    "generation_reused": True,
                    "evaluation_context": "historical_gold_v1_regression",
                },
            }
        )
    return replayed


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay saved match parsing outputs through current rules without generation"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()

    rows = replay_current_rules(list(read_jsonl(args.predictions)))
    report = build_report(rows, evaluation_context="historical_gold_v1_regression")
    report["regression_provenance"] = {
        "source_predictions": args.predictions,
        "source_predictions_sha256": file_sha256(args.predictions),
        "generation_reused": True,
        "current_rules_recomputed": True,
        "warning": "REGRESSION AFTER INSPECTION",
    }
    write_text(args.report_out, json.dumps(report, ensure_ascii=False, indent=2))
    write_text(
        args.predictions_out,
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
