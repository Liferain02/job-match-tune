from __future__ import annotations

import argparse
import json
from collections import Counter
import random
from pathlib import Path
from typing import Any

from jobmatch_tune.dataset.build_jd_bootstrap_sft_dataset import build_bootstrap_rows
from jobmatch_tune.dataset.build_jd_strict_plus_sft_dataset import (
    LOW_SIGNAL_TEXT_PATTERNS,
    LOW_SIGNAL_TITLE_KEYWORDS,
)
from jobmatch_tune.dataset.build_jd_strict_plus_sft_dataset import build_rows as build_strict_plus_rows
from jobmatch_tune.dataset.build_jd_strict_plus_sft_dataset import load_schema
from jobmatch_tune.dataset.build_sft_dataset import (
    build_jd_parse_sample,
    is_high_trust_strong_row,
    split_samples,
)
from jobmatch_tune.dataset.jd_quality_risk import HIGH_RISK_THRESHOLD, is_high_risk, risk_reasons, risk_score
from jobmatch_tune.dataset.jd_quality_risk import REQUIREMENT_MARKERS, RESPONSIBILITY_MARKERS
from jobmatch_tune.preprocess.normalize_jd import normalize_jd_row
from jobmatch_tune.utils.io import write_text
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


DEGREE_ONLY_EXPERIENCE = {
    "专科",
    "大专",
    "本科",
    "本科及以上",
    "硕士",
    "硕士及以上",
    "博士",
    "博士及以上",
    "统招本科",
    "全日制本科",
}

WEAK_SOURCE_PREFIXES = ("hf_", "github_")

NON_TECH_TITLE_KEYWORDS = [
    "教师",
    "老师",
    "讲师",
    "助教",
    "编导",
    "编剧",
    "摄制",
    "新闻编辑",
    "课程开发",
    "培训方案",
    "招生",
]

QUALITY_TIER_BASE_SCORE = {
    "strict": 100,
    "strict_plus": 90,
    "quality_weak": 75,
    "bootstrap": 60,
}


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "")


def _append_unique(
    target: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    seen_ids: set[str],
    *,
    limit: int,
    max_risk_score: int | None = None,
) -> None:
    for row in rows:
        row_id = _row_id(row)
        if not row_id or row_id in seen_ids:
            continue
        if max_risk_score is not None and is_high_risk(
            build_jd_parse_sample(row),
            threshold=max_risk_score + 1,
        ):
            continue
        target.append(row)
        seen_ids.add(row_id)
        if len(target) >= limit:
            return


def _with_quality_meta(row: dict[str, Any], tier: str, reason: str) -> dict[str, Any]:
    row = dict(row)
    meta = dict(row.get("meta") or {})
    meta["quality_tier"] = tier
    meta["quality_reason"] = reason
    row["meta"] = meta
    return row


def _tag_rows(rows: list[dict[str, Any]], tier: str, reason: str) -> list[dict[str, Any]]:
    return [_with_quality_meta(row, tier, reason) for row in rows]


def _with_quality_score_meta(row: dict[str, Any]) -> dict[str, Any]:
    sample = build_jd_parse_sample(row)
    reasons = risk_reasons(sample)
    score = risk_score(reasons)
    tier = str((row.get("meta") or {}).get("quality_tier") or "unknown")
    quality_score = max(0, QUALITY_TIER_BASE_SCORE.get(tier, 50) - score * 10)
    row = dict(row)
    meta = dict(row.get("meta") or {})
    meta["quality_risk_score"] = score
    meta["quality_risk_reasons"] = reasons
    meta["quality_score"] = quality_score
    row["meta"] = meta
    return row


def _clean_experience(value: Any) -> str:
    text = str(value or "").strip()
    if text in DEGREE_ONLY_EXPERIENCE:
        return ""
    return text


def _repair_section_boundary(sections: dict[str, Any]) -> dict[str, str]:
    repaired = {key: str(value or "").strip() for key, value in sections.items()}
    responsibilities = repaired.get("responsibilities", "")
    requirements = repaired.get("requirements", "")
    for marker in REQUIREMENT_MARKERS:
        if marker in responsibilities:
            before, after = responsibilities.split(marker, 1)
            repaired["responsibilities"] = before.strip(" ：:;；\n")
            leaked_requirement = f"{marker}：{after.strip(' ：:;；')}".strip()
            repaired["requirements"] = "\n".join(
                value for value in [requirements, leaked_requirement] if value
            ).strip()
            break
    requirements = repaired.get("requirements", "")
    responsibilities = repaired.get("responsibilities", "")
    for marker in RESPONSIBILITY_MARKERS:
        if marker in requirements:
            before, after = requirements.split(marker, 1)
            repaired["requirements"] = before.strip(" ：:;；\n")
            leaked_responsibility = f"{marker}：{after.strip(' ：:;；')}".strip()
            repaired["responsibilities"] = "\n".join(
                value for value in [responsibilities, leaked_responsibility] if value
            ).strip()
            break
    return {key: value for key, value in repaired.items() if value}


def _repair_direction_from_title(title: str, current: str) -> str:
    lowered = title.lower()
    if "ai infra" in lowered or "训练平台" in title or "推理平台" in title or "算力平台" in title:
        return "AI Infra"
    if "hpc" in lowered or "高性能计算" in title or "gpu集群" in title or "gpu 集群" in title:
        return "高性能计算"
    if "网络" in title or "基础设施" in title or "基础架构" in title:
        return "网络与基础设施"
    if "sre" in lowered or "devops" in lowered or "运维" in title:
        return "运维开发"
    if "测试" in title or "qa" in lowered:
        return "测试开发"
    if "算法" in title or "机器学习" in title or "深度学习" in title or "大模型" in title:
        return "算法工程"
    if "android" in lowered or "ios" in lowered or "客户端" in title:
        return "客户端开发"
    if "嵌入式" in title or "驱动" in title or "bsp" in lowered or "固件" in title:
        return "嵌入式开发"
    if "前端" in title or "web前端" in lowered or "vue" in lowered or "react" in lowered:
        return "前端开发"
    backend_hints = ("后端", "后台", "服务端", "java", "golang", "python", "php", ".net", "c#")
    if any(hint in lowered or hint in title for hint in backend_hints):
        return "后端开发"
    return current


def _sanitize_normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    labels = dict(row.get("labels") or {})
    experience = _clean_experience(labels.get("经验要求"))
    labels["经验要求"] = experience
    direction = str(labels.get("岗位方向") or "").strip()
    title = str(row.get("job_title") or "").strip()
    source = str(row.get("source") or "")
    if source.startswith(WEAK_SOURCE_PREFIXES):
        labels["岗位方向"] = _repair_direction_from_title(title, direction)
    row = dict(row)
    row["labels"] = labels
    row["sections"] = _repair_section_boundary(row.get("sections") or {})
    return row


def is_quality_weak_row(row: dict[str, Any]) -> bool:
    title = str(row.get("job_title") or "").strip().lower()
    clean_text = str(row.get("clean_text") or "").strip()
    labels = row.get("labels") or {}
    sections = row.get("sections") or {}

    if not title or not clean_text:
        return False
    if any(keyword in title for keyword in LOW_SIGNAL_TITLE_KEYWORDS):
        return False
    if any(keyword in title for keyword in NON_TECH_TITLE_KEYWORDS):
        return False

    direction = str(labels.get("岗位方向") or "").strip()
    education = str(labels.get("学历要求") or "").strip()
    experience = _clean_experience(labels.get("经验要求"))
    skills = labels.get("必备技能") or []
    responsibilities = str(sections.get("responsibilities") or "").strip()
    requirements = str(sections.get("requirements") or "").strip()
    resp_len = len(responsibilities)
    req_len = len(requirements)
    skill_count = len(skills)

    if not direction or skill_count < 2 or not (education or experience):
        return False
    if len(clean_text) < 180 or max(resp_len, req_len) < 30:
        return False
    if any(pattern in clean_text for pattern in LOW_SIGNAL_TEXT_PATTERNS):
        return bool(education and skill_count >= 2 and max(resp_len, req_len) >= 80)
    return bool((education or experience) and max(resp_len, req_len) >= 30)


def build_quality_weak_rows(rows: list[dict[str, Any]], schema: dict[str, Any]) -> list[dict[str, Any]]:
    built = []
    for row in rows:
        normalized = normalize_jd_row(
            {
                "id": row["id"],
                "job_title": row.get("job_title", ""),
                "source": row.get("source", ""),
                "company": row.get("company", ""),
                "location": row.get("location", ""),
                "raw_text": row.get("raw_text", ""),
                "meta": row.get("meta") or {},
            },
            schema,
        )
        normalized = _sanitize_normalized_row(normalized)
        if is_quality_weak_row(normalized):
            built.append(
                _with_quality_meta(
                    normalized,
                    "quality_weak",
                    "weak_source_direction_skill_education_or_experience",
                )
            )
    return built


def build_quality_rows(
    *,
    strict_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    schema: dict[str, Any],
    target_total: int,
    seed: int,
    max_risk_score: int | None = HIGH_RISK_THRESHOLD - 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    strict_selected = [_sanitize_normalized_row(row) for row in strict_rows if is_high_trust_strong_row(row)]
    strict_selected = _tag_rows(
        strict_selected,
        "strict",
        "high_trust_zh_tech_structured",
    )
    _append_unique(selected, strict_selected, seen_ids, limit=target_total, max_risk_score=max_risk_score)

    stats = {"strict": len(selected), "strict_plus": 0, "quality_weak": 0, "bootstrap": 0}
    if len(selected) >= target_total:
        return selected[:target_total], stats

    strict_plus_rows = build_strict_plus_rows(candidate_rows, schema)
    strict_plus_rows = [_sanitize_normalized_row(row) for row in strict_plus_rows]
    strict_plus_rows = _tag_rows(
        strict_plus_rows,
        "strict_plus",
        "candidate_pool_structured_direction_education",
    )
    before = len(selected)
    _append_unique(selected, strict_plus_rows, seen_ids, limit=target_total, max_risk_score=max_risk_score)
    stats["strict_plus"] = len(selected) - before
    if len(selected) >= target_total:
        return selected[:target_total], stats

    quality_weak_rows = build_quality_weak_rows(candidate_rows, schema)
    quality_weak_rows = _tag_rows(
        quality_weak_rows,
        "quality_weak",
        "weak_source_direction_skill_education_or_experience",
    )
    rng = random.Random(seed)
    rng.shuffle(quality_weak_rows)
    before = len(selected)
    _append_unique(selected, quality_weak_rows, seen_ids, limit=target_total, max_risk_score=max_risk_score)
    stats["quality_weak"] = len(selected) - before
    if len(selected) >= target_total:
        return selected[:target_total], stats

    bootstrap_rows = build_bootstrap_rows(candidate_rows, schema)
    bootstrap_rows = _tag_rows(
        bootstrap_rows,
        "bootstrap",
        "bootstrap_repairable_structured",
    )
    rng.shuffle(bootstrap_rows)
    before = len(selected)
    _append_unique(selected, bootstrap_rows, seen_ids, limit=target_total, max_risk_score=max_risk_score)
    stats["bootstrap"] = len(selected) - before
    return selected[:target_total], stats


def build_quality_profile(rows: list[dict[str, Any]], stats: dict[str, int]) -> dict[str, Any]:
    tier_counts = Counter()
    reason_counts = Counter()
    source_counts = Counter()
    direction_counts = Counter()
    empty_counts = Counter()
    risk_score_counts = Counter()
    quality_score_counts = Counter()
    quality_scores = []
    for row in rows:
        meta = row.get("meta") or {}
        labels = row.get("labels") or {}
        tier_counts[str(meta.get("quality_tier") or "unknown")] += 1
        reason_counts[str(meta.get("quality_reason") or "unknown")] += 1
        source_counts[str(row.get("source") or "unknown")] += 1
        direction_counts[str(labels.get("岗位方向") or "unknown")] += 1
        risk_score_counts[int(meta.get("quality_risk_score") or 0)] += 1
        quality_score = int(meta.get("quality_score") or 0)
        quality_scores.append(quality_score)
        quality_score_counts[str((quality_score // 10) * 10)] += 1
        assistant = json.loads(build_jd_parse_sample(row)["messages"][-1]["content"])
        if not assistant.get("核心职责"):
            empty_counts["核心职责"] += 1
        if not assistant.get("必备技能"):
            empty_counts["必备技能"] += 1
        if not assistant.get("学历要求"):
            empty_counts["学历要求"] += 1
        if not assistant.get("经验要求"):
            empty_counts["经验要求"] += 1
    total = len(rows)
    return {
        "total": total,
        "stage_stats": stats,
        "tier_counts": dict(tier_counts),
        "reason_counts": dict(reason_counts),
        "top_sources": source_counts.most_common(20),
        "direction_counts": dict(direction_counts),
        "empty_counts": dict(empty_counts),
        "empty_rates": {key: round(value / total, 4) for key, value in empty_counts.items()},
        "risk_score_counts": {str(key): value for key, value in sorted(risk_score_counts.items())},
        "quality_score_avg": round(sum(quality_scores) / total, 2) if total else 0.0,
        "quality_score_buckets": dict(sorted(quality_score_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-input", default="data/interim/jd_clean_dedup.jsonl")
    parser.add_argument("--candidate-input", default="data/eval/jd_train_pool_combined.jsonl")
    parser.add_argument("--schema", default="configs/label_schema.yaml")
    parser.add_argument("--out-dir", default="data/sft_jd_quality")
    parser.add_argument("--profile-out", default="outputs/eval_reports/jd_quality_profile.json")
    parser.add_argument("--target-total", type=int, default=5000)
    parser.add_argument("--max-risk-score", type=int, default=HIGH_RISK_THRESHOLD - 1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    strict_rows = list(read_jsonl(args.strict_input))
    candidate_rows = list(read_jsonl(args.candidate_input))
    quality_rows, stats = build_quality_rows(
        strict_rows=strict_rows,
        candidate_rows=candidate_rows,
        schema=schema,
        target_total=args.target_total,
        seed=args.seed,
        max_risk_score=args.max_risk_score,
    )
    if len(quality_rows) < args.target_total:
        raise SystemExit(
            f"Only built {len(quality_rows)} rows, below target_total={args.target_total}. "
            f"stage_stats={stats}"
        )
    quality_rows = [_with_quality_score_meta(row) for row in quality_rows]

    samples = [build_jd_parse_sample(row) for row in quality_rows]
    splits = split_samples(samples, args.train_ratio, args.valid_ratio, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        write_jsonl(str(out_dir / f"{split}.jsonl"), split_rows)
        print(f"wrote {len(split_rows)} {split} samples")
    profile = build_quality_profile(quality_rows, stats)
    write_text(args.profile_out, json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote quality profile to {args.profile_out}")
    print(f"stage_stats={stats}")


if __name__ == "__main__":
    main()
