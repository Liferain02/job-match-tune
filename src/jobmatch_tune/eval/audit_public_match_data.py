from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from statistics import mean
from typing import Any

from jobmatch_tune.utils.io import read_jsonl, write_text


def compute_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counter = Counter()
    language_counter = Counter()
    source_type_counter = Counter()
    raw_label_counter = Counter()
    license_counter = Counter()
    usage_counter = Counter()
    provenance_counter = Counter()
    jd_lengths: list[int] = []
    resume_lengths: list[int] = []
    score_present = 0
    training_eligible_rows = 0
    pair_hashes: Counter[str] = Counter()

    for row in rows:
        meta = row.get("meta") or {}
        source_counter[str(meta.get("source_name") or "")] += 1
        language_counter[str(meta.get("language") or "")] += 1
        source_type_counter[str(row.get("source_type") or "")] += 1
        license_status = str(meta.get("license_status") or "unconfirmed").lower()
        intended_usage = str(meta.get("intended_usage") or "audit_only").lower()
        provenance_status = str(meta.get("provenance_status") or "undocumented").lower()
        license_counter[license_status] += 1
        usage_counter[intended_usage] += 1
        provenance_counter[provenance_status] += 1
        jd_text = str(row.get("jd_text") or "")
        resume_text = str(row.get("resume_text") or "")
        pair_digest = hashlib.sha1(
            f"{jd_text.strip()}\n---\n{resume_text.strip()}".encode("utf-8")
        ).hexdigest()
        pair_hashes[pair_digest] += 1
        if jd_text:
            jd_lengths.append(len(jd_text))
        if resume_text:
            resume_lengths.append(len(resume_text))
        label = row.get("label") or {}
        raw_label = str(label.get("raw_label") or "")
        if raw_label:
            raw_label_counter[raw_label] += 1
        if label.get("raw_score") not in ("", None):
            score_present += 1
        if (
            license_status == "confirmed"
            and intended_usage
            in {"training", "sft_training", "training_and_evaluation"}
            and provenance_status not in {"", "undocumented", "unknown"}
            and bool(raw_label or label.get("raw_score") not in ("", None))
        ):
            training_eligible_rows += 1

    total = len(rows)
    return {
        "total_rows": total,
        "source_distribution": source_counter.most_common(),
        "language_distribution": language_counter.most_common(),
        "source_type_distribution": source_type_counter.most_common(),
        "raw_label_distribution_top20": raw_label_counter.most_common(20),
        "license_status_distribution": license_counter.most_common(),
        "intended_usage_distribution": usage_counter.most_common(),
        "provenance_status_distribution": provenance_counter.most_common(),
        "avg_jd_length": mean(jd_lengths) if jd_lengths else 0.0,
        "avg_resume_length": mean(resume_lengths) if resume_lengths else 0.0,
        "score_coverage": score_present / total if total else 0.0,
        "duplicate_pair_rows": sum(count - 1 for count in pair_hashes.values() if count > 1),
        "training_eligible_rows": training_eligible_rows,
        "training_ready": training_eligible_rows > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    report = compute_report(rows)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
