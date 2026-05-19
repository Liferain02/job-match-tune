from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.build_sft_dataset import (
    HIGH_TRUST_SOURCES,
    get_effective_direction,
    is_tencent_short_tech_row,
    is_high_trust_strong_row,
    title_has_excluded_signal,
    title_has_exclusion_exception,
    title_has_strong_tech_signal,
)
from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
)
from jobmatch_tune.utils.io import read_jsonl, write_jsonl, write_text


def _has_structure_marker(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "岗位职责",
            "工作职责",
            "职位描述",
            "工作内容",
            "职责描述",
            "任职要求",
            "岗位要求",
            "职位要求",
            "任职资格",
            "能力要求",
            "技能要求",
            "加分项",
        )
    )


def classify_rejection(row: dict[str, Any]) -> str:
    language = str(row.get("language") or "").strip().lower()
    source = row.get("source")
    title = str(row.get("job_title") or "").strip()
    lowered_title = title.lower()
    clean_text = str(row.get("clean_text") or "").strip()
    labels = row.get("labels") or {}
    direction = get_effective_direction(row)

    is_zh_like = language in {"zh", "zh-cn"} or (not language and source == "careers.tencent.com")
    if not is_zh_like:
        return "language_not_zh"
    if not row.get("sft_ready", True):
        return "sft_not_ready"
    if source not in HIGH_TRUST_SOURCES:
        return "source_not_high_trust"
    if not title:
        return "missing_title"
    if not clean_text:
        return "missing_clean_text"
    if not direction:
        return "missing_direction"
    if title_has_excluded_signal(lowered_title) and not title_has_exclusion_exception(lowered_title, direction):
        return "excluded_title"
    if not title_has_strong_tech_signal(lowered_title, direction):
        return "missing_title_signal"

    sections = row.get("sections") or {}
    has_responsibilities = bool(str(sections.get("responsibilities") or "").strip())
    has_requirements = bool(str(sections.get("requirements") or "").strip())
    has_bonus = bool(str(sections.get("bonus") or "").strip())
    has_skills = bool(labels.get("必备技能"))
    has_education = bool(labels.get("学历要求") or extract_education_requirement(clean_text))
    has_experience = bool(labels.get("经验要求") or extract_experience_requirement(clean_text))
    has_structure_marker = _has_structure_marker(clean_text)

    base_ok = (
        ((has_responsibilities and has_requirements) or (len(clean_text) >= 180 and (has_responsibilities or has_requirements)))
        and (has_education or has_experience or has_skills)
    )
    if base_ok:
        return "accepted"

    fallback_signals = sum(bool(flag) for flag in [has_skills, has_education, has_experience, has_bonus])
    if has_structure_marker and len(clean_text) >= 180 and fallback_signals >= 2:
        return "accepted"
    if is_tencent_short_tech_row(row, direction):
        return "accepted"

    if not (has_responsibilities or has_requirements):
        return "missing_sections"
    if len(clean_text) < 180:
        return "clean_text_too_short"
    if not (has_education or has_experience or has_skills):
        return "missing_edu_exp_skill"
    return "structure_or_signal_fail"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counter: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_direction: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        reason = classify_rejection(row)
        if reason == "accepted":
            continue
        reason_counter[reason] += 1
        by_source[str(row.get("source") or "")][reason] += 1
        by_direction[str((row.get("labels") or {}).get("岗位方向") or "")][reason] += 1

    return {
        "total_rejected": sum(reason_counter.values()),
        "top_reasons": [{"name": name, "count": count} for name, count in reason_counter.most_common(20)],
        "top_sources": [
            {"source": source, "count": sum(counter.values()), "top_reasons": counter.most_common(5)}
            for source, counter in sorted(by_source.items(), key=lambda item: sum(item[1].values()), reverse=True)[:20]
        ],
        "top_directions": [
            {"direction": direction or "<empty>", "count": sum(counter.values()), "top_reasons": counter.most_common(5)}
            for direction, counter in sorted(by_direction.items(), key=lambda item: sum(item[1].values()), reverse=True)[:20]
        ],
    }


def build_samples(rows: list[dict[str, Any]], per_reason: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        reason = classify_rejection(row)
        if reason == "accepted":
            continue
        grouped[reason].append(row)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for reason, candidates in grouped.items():
        chosen = candidates[:]
        rng.shuffle(chosen)
        for row in chosen[:per_reason]:
            sampled.append(
                {
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "job_title": row.get("job_title"),
                    "reason": reason,
                    "direction": get_effective_direction(row),
                    "experience": (row.get("labels") or {}).get("经验要求", ""),
                    "education": (row.get("labels") or {}).get("学历要求", ""),
                    "skills": (row.get("labels") or {}).get("必备技能", []),
                    "clean_text_preview": str(row.get("clean_text") or "")[:400],
                }
            )
    sampled.sort(key=lambda item: (item["reason"], str(item["source"]), str(item["job_title"])))
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/jd_clean_dedup.jsonl")
    parser.add_argument("--out", default="outputs/eval_reports/jd_strict_rejection_report.json")
    parser.add_argument("--sample-out", default="outputs/eval_reports/jd_strict_rejection_samples.jsonl")
    parser.add_argument("--per-reason", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.input) if row.get("source") in HIGH_TRUST_SOURCES]
    report = summarize_rows(rows)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")
    samples = build_samples(rows, per_reason=args.per_reason, seed=args.seed)
    write_jsonl(args.sample_out, samples)
    print(f"wrote {len(samples)} samples to {args.sample_out}")


if __name__ == "__main__":
    main()
