from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobmatch_tune.database import connect
from jobmatch_tune.utils.io import write_text


NORMALIZATION_MANIFEST = "data/interim/jd_clean.manifest.json"
DEFAULT_LABEL_SCHEMA = "configs/label_schema.yaml"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PACKAGE_ROOT / "dataset"
PREPROCESS_ROOT = PACKAGE_ROOT / "preprocess"
RESUME_ROOT = PACKAGE_ROOT / "resume"
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
NORMALIZATION_TRANSFORM_FILES = (
    PACKAGE_ROOT / "preprocess" / "clean_text.py",
    PACKAGE_ROOT / "preprocess" / "jd_field_rules.py",
    PACKAGE_ROOT / "preprocess" / "jd_sections.py",
    PACKAGE_ROOT / "preprocess" / "normalize_jd.py",
    PACKAGE_ROOT / "preprocess" / "skill_canonicalization.py",
)

DERIVED_DEPENDENCIES = (
    (
        "JD规范化到去重",
        ("data/interim/jd_clean.jsonl", str(PREPROCESS_ROOT / "deduplicate.py")),
        ("data/interim/jd_clean_dedup.jsonl",),
    ),
    (
        "公开JD原始层到候选池",
        (
            "data/raw/public_job_datasets_raw.jsonl",
            str(DATASET_ROOT / "build_public_jd_candidate_pool.py"),
        ),
        ("data/eval/public_jd_candidate_pool.jsonl",),
    ),
    (
        "JD候选池到组合池",
        (
            "data/interim/jd_clean_dedup.jsonl",
            "data/eval/public_jd_candidate_pool.jsonl",
            str(DATASET_ROOT / "build_jd_train_pool_combined.py"),
            str(DATASET_ROOT / "build_sft_dataset.py"),
        ),
        ("data/eval/jd_train_pool_combined.jsonl",),
    ),
    (
        "JD组合池到严格SFT",
        (
            "data/eval/jd_train_pool_combined.jsonl",
            str(DATASET_ROOT / "build_jd_strict_plus_sft_dataset.py"),
            str(DATASET_ROOT / "build_sft_dataset.py"),
            str(DATASET_ROOT / "templates.py"),
        ),
        (
            "data/sft_jd_strict_plus/train.jsonl",
            "data/sft_jd_strict_plus/valid.jsonl",
            "data/sft_jd_strict_plus/test.jsonl",
        ),
    ),
    (
        "中文技术简历公开源到训练导入",
        (
            "data/external/faircv/data/resumes_template.json",
            "data/eval/public_jd_candidate_pool.jsonl",
            str(PROJECT_ROOT / "configs" / "public_training_sources.yaml"),
            str(PROJECT_ROOT / "configs" / "public_chinese_resume_sources.yaml"),
            str(PROJECT_ROOT / "configs" / "human_reviewed_public_match_pairs.yaml"),
            str(DATASET_ROOT / "import_public_training_data.py"),
            str(RESUME_ROOT / "privacy.py"),
        ),
        (
            "data/external/public_resume_imports_zh_tech.jsonl",
            "data/external/public_match_imports_zh_tech.jsonl",
        ),
    ),
    (
        "简历原始池到组合池",
        (
            "data/eval/resume_manual_train_pool.jsonl",
            "data/external/public_resume_imports_zh_tech.jsonl",
            str(DATASET_ROOT / "curated_resume_training_data.py"),
            str(DATASET_ROOT / "build_resume_train_pool_combined.py"),
            str(RESUME_ROOT / "privacy.py"),
        ),
        ("data/eval/resume_train_pool_combined.jsonl",),
    ),
    (
        "简历组合池到SFT",
        (
            "data/eval/resume_train_pool_combined.jsonl",
            str(DATASET_ROOT / "build_resume_sft_dataset.py"),
            str(DATASET_ROOT / "grouped_split.py"),
            str(DATASET_ROOT / "templates.py"),
            str(RESUME_ROOT / "privacy.py"),
        ),
        ("data/sft_resume/train.jsonl", "data/sft_resume/valid.jsonl", "data/sft_resume/test.jsonl"),
    ),
    (
        "匹配准入数据到组合池",
        (
            "data/eval/match_curated_train_pool.jsonl",
            "data/external/public_match_imports_zh_tech.jsonl",
            str(PROJECT_ROOT / "configs" / "public_match_sources.yaml"),
            str(PROJECT_ROOT / "configs" / "public_training_sources.yaml"),
            str(DATASET_ROOT / "curated_match_training_data.py"),
            str(DATASET_ROOT / "build_match_train_pool_combined.py"),
        ),
        ("data/eval/match_train_pool_combined.jsonl",),
    ),
    (
        "匹配组合池到SFT",
        (
            "data/eval/match_train_pool_combined.jsonl",
            str(DATASET_ROOT / "build_match_sft_dataset.py"),
            str(DATASET_ROOT / "grouped_split.py"),
            str(DATASET_ROOT / "templates.py"),
        ),
        ("data/sft_match/train.jsonl", "data/sft_match/valid.jsonl", "data/sft_match/test.jsonl"),
    ),
    (
        "单任务SFT到多任务SFT",
        (
            "data/sft_jd_strict_plus/train.jsonl",
            "data/sft_jd_strict_plus/valid.jsonl",
            "data/sft_resume/train.jsonl",
            "data/sft_resume/valid.jsonl",
            "data/sft_match/train.jsonl",
            "data/sft_match/valid.jsonl",
            str(DATASET_ROOT / "build_multitask_sft_dataset.py"),
            str(PROJECT_ROOT / "configs" / "dataset_registry.yaml"),
        ),
        ("data/sft_multitask/train.jsonl", "data/sft_multitask/valid.jsonl"),
    ),
    (
        "JDSFT到偏好数据",
        (
            "data/sft_jd_strict_plus/train.jsonl",
            "data/sft_jd_strict_plus/valid.jsonl",
            str(DATASET_ROOT / "build_preference_dataset.py"),
            str(DATASET_ROOT / "build_match_sft_dataset.py"),
            str(DATASET_ROOT / "templates.py"),
        ),
        ("data/preference/train.jsonl", "data/preference/valid.jsonl"),
    ),
)

DPO_DEPENDENCY_NAMES = {
    "JDSFT到偏好数据",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalization_transform_sha256(schema_path: str | Path = DEFAULT_LABEL_SCHEMA) -> str:
    digest = hashlib.sha256()
    for path in (*NORMALIZATION_TRANSFORM_FILES, Path(schema_path)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def get_jd_raw_state(db_path: str | Path) -> dict[str, Any]:
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS row_count,
                   MAX(crawl_time) AS max_crawl_time,
                   SUM(LENGTH(raw_text)) AS raw_text_chars,
                   SUM(LENGTH(meta_json)) AS meta_chars
            FROM jd_raw
            """
        ).fetchone()
    return {
        "row_count": int(row["row_count"] or 0),
        "max_crawl_time": str(row["max_crawl_time"] or ""),
        "raw_text_chars": int(row["raw_text_chars"] or 0),
        "meta_chars": int(row["meta_chars"] or 0),
    }


def write_normalization_manifest(
    db_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path = NORMALIZATION_MANIFEST,
    schema_path: str | Path = DEFAULT_LABEL_SCHEMA,
) -> dict[str, Any]:
    output = Path(output_path)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database": str(db_path),
        "transform_sha256": normalization_transform_sha256(schema_path),
        "label_schema": str(schema_path),
        "raw_state": get_jd_raw_state(db_path),
        "output": {
            "path": str(output_path),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }
    write_text(manifest_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def verify_normalization_manifest(
    manifest_path: str | Path = NORMALIZATION_MANIFEST,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    reasons: list[str] = []
    if not manifest_file.exists():
        return {"fresh": False, "reasons": [f"缺少规范化清单：{manifest_file}"]}
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    schema_path = str(manifest.get("label_schema") or DEFAULT_LABEL_SCHEMA)
    current_transform_sha256 = normalization_transform_sha256(schema_path)
    if str(manifest.get("transform_sha256") or "") != current_transform_sha256:
        reasons.append("JD 规范化规则或标签 schema 已变化，需重新规范化")
    db_path = str(manifest.get("database") or "")
    output = manifest.get("output") or {}
    output_path = Path(str(output.get("path") or ""))
    if not Path(db_path).exists():
        reasons.append(f"数据库不存在：{db_path}")
        current_raw_state = {}
    else:
        current_raw_state = get_jd_raw_state(db_path)
        if current_raw_state != (manifest.get("raw_state") or {}):
            reasons.append("jd_raw 状态已变化，需重新规范化")
    if not output_path.exists():
        reasons.append(f"规范化输出不存在：{output_path}")
        current_sha256 = ""
    else:
        current_sha256 = sha256_file(output_path)
        if current_sha256 != str(output.get("sha256") or ""):
            reasons.append("规范化输出哈希与清单不一致")
    return {
        "fresh": not reasons,
        "reasons": reasons,
        "manifest": str(manifest_file),
        "recorded_raw_state": manifest.get("raw_state") or {},
        "current_raw_state": current_raw_state,
        "recorded_output_sha256": str(output.get("sha256") or ""),
        "current_output_sha256": current_sha256,
        "recorded_transform_sha256": str(manifest.get("transform_sha256") or ""),
        "current_transform_sha256": current_transform_sha256,
    }


def verify_dependency(
    name: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
) -> dict[str, Any]:
    missing = [path for path in (*inputs, *outputs) if not Path(path).exists()]
    if missing:
        return {"name": name, "fresh": False, "missing": missing}
    newest_input_ns = max(Path(path).stat().st_mtime_ns for path in inputs)
    oldest_output_ns = min(Path(path).stat().st_mtime_ns for path in outputs)
    return {
        "name": name,
        "fresh": oldest_output_ns >= newest_input_ns,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "newest_input_mtime_ns": newest_input_ns,
        "oldest_output_mtime_ns": oldest_output_ns,
    }


def build_pipeline_freshness_report() -> dict[str, Any]:
    normalization = verify_normalization_manifest()
    dependencies = [verify_dependency(name, inputs, outputs) for name, inputs, outputs in DERIVED_DEPENDENCIES]
    sft_dependencies = [item for item in dependencies if item["name"] not in DPO_DEPENDENCY_NAMES]
    dpo_dependencies = [item for item in dependencies if item["name"] in DPO_DEPENDENCY_NAMES]
    sft_fresh = bool(normalization["fresh"] and all(item["fresh"] for item in sft_dependencies))
    dpo_fresh = bool(all(item["fresh"] for item in dpo_dependencies))
    return {
        "fresh": bool(sft_fresh and dpo_fresh),
        "sft_fresh": sft_fresh,
        "dpo_fresh": dpo_fresh,
        "normalization": normalization,
        "dependencies": dependencies,
        "stale_dependencies": [item["name"] for item in dependencies if not item["fresh"]],
        "stale_sft_dependencies": [item["name"] for item in sft_dependencies if not item["fresh"]],
        "stale_dpo_dependencies": [item["name"] for item in dpo_dependencies if not item["fresh"]],
    }
