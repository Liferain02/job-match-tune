from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jobmatch_tune.utils.io import read_jsonl, write_text


REQUIRED_TAG_GROUPS = {
    "ocr": {"OCR词内断开", "OCR误报控制"},
    "requirement_boundary": {"职责与要求边界", "职责要求重复", "加分项非硬门槛"},
    "direction": {"跨岗位方向"},
    "project": {"项目证据"},
}


def _normalized_hash(*texts: str) -> str:
    normalized = "\n".join(re.sub(r"\s+", "", text).casefold() for text in texts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_candidates(
    rows: list[dict[str, Any]],
    *,
    comparison_rows: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    ids = [str(row.get("id") or "") for row in rows]
    statuses = Counter(str((row.get("meta") or {}).get("annotation_status") or "missing") for row in rows)
    tags = Counter(
        str(tag)
        for row in rows
        for tag in ((row.get("meta") or {}).get("difficulty_tags") or [])
    )
    pair_hashes = [_normalized_hash(str(row.get("jd_text") or ""), str(row.get("resume_text") or "")) for row in rows]
    comparison_hashes = {
        _normalized_hash(str(row.get("jd_text") or ""), str(row.get("resume_text") or ""))
        for row in comparison_rows
    }
    problems: list[str] = []
    if not 15 <= len(rows) <= 20:
        problems.append("candidate_count_must_be_15_to_20")
    if len(set(ids)) != len(ids) or any(not value.startswith("semantic_boundary_v2_") for value in ids):
        problems.append("ids_must_be_unique_v2_ids")
    if statuses != {"needs_human_review": len(rows)}:
        problems.append("all_rows_must_need_human_review")
    if any((row.get("meta") or {}).get("training_eligible") is not False for row in rows):
        problems.append("all_rows_must_be_training_ineligible")
    if len(set(pair_hashes)) != len(pair_hashes):
        problems.append("duplicate_candidate_pairs")
    if set(pair_hashes) & comparison_hashes:
        problems.append("candidate_pair_overlaps_comparison_set")
    missing_groups = [
        name for name, accepted_tags in REQUIRED_TAG_GROUPS.items() if not (set(tags) & accepted_tags)
    ]
    if missing_groups:
        problems.append(f"missing_difficulty_groups:{','.join(missing_groups)}")
    return {
        "dataset": "semantic_boundary_candidate_v2",
        "row_count": len(rows),
        "annotation_status_counts": dict(statuses),
        "difficulty_tag_counts": dict(tags),
        "training_eligible_rows": sum(
            bool((row.get("meta") or {}).get("training_eligible")) for row in rows
        ),
        "comparison_pair_overlap": len(set(pair_hashes) & comparison_hashes),
        "audit_ok": not problems,
        "problems": problems,
        "gold_ready": False,
        "evaluation_status": "needs_human_review",
        "warning": "该集合是未审核候选，不是 human Gold，不得报告产品泛化指标。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Semantic Boundary Candidate V2")
    parser.add_argument("--input", default="data/private/semantic_boundary_candidate_v2.jsonl")
    parser.add_argument("--compare", default="data/private/match_gold.jsonl")
    parser.add_argument("--out", default="outputs/eval_reports/semantic_boundary_candidate_v2_audit.json")
    args = parser.parse_args()
    rows = list(read_jsonl(args.input))
    comparison = list(read_jsonl(args.compare)) if Path(args.compare).exists() else []
    report = audit_candidates(rows, comparison_rows=comparison)
    write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["audit_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
