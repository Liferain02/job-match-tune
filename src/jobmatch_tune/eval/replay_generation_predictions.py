from __future__ import annotations

import argparse
import json
from typing import Any

from jobmatch_tune.eval.run_manual_eval import evaluate_predictions
from jobmatch_tune.inference.postprocess_json import parse_json_output
from jobmatch_tune.utils.io import read_jsonl, write_text


def replay_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replayed = []
    for row in rows:
        context_text = row.get("text") or row.get("normalized_text") or row.get("raw_text") or ""
        parsed = parse_json_output(row.get("prediction", ""), context_text=context_text)
        replayed.append(
            {
                **row,
                "parsed": parsed.get("data"),
                "ok": parsed["ok"],
                "error": parsed.get("error"),
            }
        )
    return replayed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay saved JD/resume generation predictions through the latest JSON postprocessor."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replayed-out")
    args = parser.parse_args()

    replayed = replay_predictions(list(read_jsonl(args.predictions)))
    report = evaluate_predictions(replayed)
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2))
    if args.replayed_out:
        replayed_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in replayed)
        write_text(args.replayed_out, replayed_text + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
