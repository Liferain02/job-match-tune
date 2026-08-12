from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.dataset.grouped_split import normalized_input_hash
from jobmatch_tune.dataset.import_public_resume_data import read_rows
from jobmatch_tune.resume.privacy import (
    detect_resume_pii,
    sanitize_resume_text_for_training,
)
from jobmatch_tune.utils.io import read_jsonl, write_text


ALLOWED_DECISIONS = {"training_allowed", "audit_only", "evaluation_only", "rejected"}
TRAINING_USAGES = {"training", "sft_training", "training_and_evaluation"}
SENSITIVE_NER_TYPES = {"CONT", "LOC", "NAME", "ORG", "RACE"}
CORE_RESUME_FIELDS = ("目标岗位", "教育背景", "核心技能", "实习经历", "项目经历")


def _load_sources(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    sources = payload.get("sources") or []
    if not isinstance(sources, list):
        raise ValueError(f"Invalid resume source manifest: {path}")
    return sources


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_sensitive_ner(row: dict[str, Any]) -> bool:
    return any(
        str(tag).split("-", 1)[-1] in SENSITIVE_NER_TYPES
        for tag in row.get("ner_tags") or []
    )


def _schema_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resume_parse_rows = [row for row in rows if row.get("task") == "resume_parse"]
    coverage: dict[str, float] = {}
    usable_rows = 0
    for field in CORE_RESUME_FIELDS:
        covered = sum(bool((row.get("label") or {}).get(field)) for row in resume_parse_rows)
        coverage[field] = round(covered / len(resume_parse_rows), 4) if resume_parse_rows else 0.0
    for row in resume_parse_rows:
        label = row.get("label") or {}
        signals = sum(bool(label.get(field)) for field in CORE_RESUME_FIELDS)
        usable_rows += int(signals >= 3)
    return {
        "task_counts": dict(Counter(str(row.get("task") or "missing") for row in rows)),
        "resume_parse_field_coverage": coverage,
        "resume_parse_usable_rows": usable_rows,
        "resume_parse_usable_rate": (
            round(usable_rows / len(resume_parse_rows), 4) if resume_parse_rows else 0.0
        ),
    }


def _audit_source(source: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_path = Path(str(source.get("path") or ""))
    raw_rows = read_rows(source_path) if source_path.exists() else []
    schema = str(source.get("schema") or "")
    rows_with_pii = 0
    pii_counts: Counter[str] = Counter()
    privacy_clean_texts: list[str] = []
    residual_privacy_rows = 0

    for row in rows:
        text = str(row.get("text") or "")
        findings = detect_resume_pii(text)
        ner_sensitive = schema == "resume_ner_rows" and _has_sensitive_ner(row)
        if findings or ner_sensitive:
            rows_with_pii += 1
            pii_counts.update(finding.kind for finding in findings)
            if ner_sensitive:
                pii_counts["annotated_sensitive_entity"] += 1
        # Removing NER tokens would invalidate their BIO labels. Exclude those rows
        # from the privacy-clean candidate count instead of claiming redaction kept
        # a usable annotation.
        if ner_sensitive:
            continue
        sanitized = sanitize_resume_text_for_training(text).strip()
        if not sanitized:
            continue
        if detect_resume_pii(sanitized):
            residual_privacy_rows += 1
            continue
        privacy_clean_texts.append(sanitized)

    exact_hashes = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in privacy_clean_texts}
    normalized_hashes = {normalized_input_hash(text) for text in privacy_clean_texts}
    after_privacy = len(privacy_clean_texts)
    unique_groups = len(normalized_hashes)
    duplicate_rows = after_privacy - unique_groups
    duplicate_ratio = round(duplicate_rows / after_privacy, 4) if after_privacy else 0.0
    schema_profile = _schema_profile(rows)
    resume_parse_usable = bool(
        schema == "resume_parse_rows"
        and schema_profile["resume_parse_usable_rate"] >= 0.9
    )
    provenance_status = str(source.get("provenance_status") or "undocumented").lower()
    origin = str(source.get("origin") or "")
    is_template_generated = "template" in str(source.get("collection_method") or "").lower()
    diverse_content = unique_groups >= int(source.get("min_unique_content_groups") or 20)
    template_ratio_ok = duplicate_ratio <= float(source.get("max_duplicate_template_ratio") or 0.8)
    license_ok = str(source.get("license_status") or "").lower() == "confirmed"
    usage_ok = str(source.get("intended_usage") or "").lower() in TRAINING_USAGES
    provenance_ok = provenance_status not in {
        "",
        "undocumented",
        "unknown",
        "source_collection_and_original_producer_undocumented",
    }
    privacy_ok = residual_privacy_rows == 0 and after_privacy > 0
    usable_for_match = bool(
        resume_parse_usable
        and diverse_content
        and template_ratio_ok
        and not is_template_generated
    )
    computed_training_allowed = bool(
        license_ok
        and usage_ok
        and provenance_ok
        and privacy_ok
        and resume_parse_usable
        and diverse_content
        and template_ratio_ok
        and not is_template_generated
    )
    declared_decision = str(source.get("admission_decision") or "rejected")
    if declared_decision not in ALLOWED_DECISIONS:
        declared_decision = "rejected"
    final_decision = (
        "training_allowed"
        if computed_training_allowed and declared_decision == "training_allowed"
        else declared_decision if declared_decision != "training_allowed" else "rejected"
    )
    return {
        "source_name": str(source.get("name") or ""),
        "source_url": str(source.get("source_url") or ""),
        "origin": origin,
        "upstream_revision": str(source.get("upstream_revision") or ""),
        "local_path": str(source_path),
        "local_sha256": _sha256(source_path) if source_path.exists() else "",
        "license": str(source.get("license") or ""),
        "license_status": str(source.get("license_status") or "unconfirmed"),
        "allowed_usage": str(source.get("allowed_usage") or ""),
        "intended_usage": str(source.get("intended_usage") or "audit_only"),
        "provenance": provenance_status,
        "collection_method": str(source.get("collection_method") or "undocumented"),
        "annotation_method": str(source.get("annotation_method") or "undocumented"),
        "privacy_status": str(source.get("privacy_status") or "unknown"),
        "pii_risk": str(source.get("pii_risk") or "unknown"),
        "schema": schema,
        "schema_completeness": schema_profile,
        "number_of_raw_rows": len(raw_rows),
        "number_of_imported_rows": len(rows),
        "rows_with_pii_or_sensitive_entities": rows_with_pii,
        "privacy_finding_counts": dict(pii_counts),
        "number_after_privacy_clean": after_privacy,
        "rows_removed_by_privacy_gate": len(rows) - after_privacy,
        "residual_privacy_rows": residual_privacy_rows,
        "exact_content_hashes_after_privacy_clean": len(exact_hashes),
        "number_after_content_dedup": unique_groups,
        "unique_content_groups": unique_groups,
        "duplicate_template_rows": duplicate_rows,
        "duplicate_template_ratio": duplicate_ratio,
        "usable_for_resume_parse": resume_parse_usable,
        "usable_for_match": usable_for_match,
        "usable_for_training": final_decision == "training_allowed",
        "gate_checks": {
            "license_ready": license_ok,
            "allowed_usage_ready": usage_ok,
            "provenance_ready": provenance_ok,
            "privacy_ready": privacy_ok,
            "schema_ready": resume_parse_usable,
            "content_diversity_ready": diverse_content and template_ratio_ok,
            "independent_real_content_ready": not is_template_generated,
        },
        "final_admission_decision": final_decision,
        "rejection_reason": str(source.get("rejection_reason") or ""),
    }


def build_resume_source_admission_report(
    manifest_path: str,
    imported_path: str,
) -> dict[str, Any]:
    imported_rows = list(read_jsonl(imported_path)) if Path(imported_path).exists() else []
    rows_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in imported_rows:
        rows_by_source[str((row.get("meta") or {}).get("source_name") or "")].append(row)
    sources = [
        _audit_source(source, rows_by_source.get(str(source.get("name") or ""), []))
        for source in _load_sources(manifest_path)
    ]
    decision_counts = Counter(item["final_admission_decision"] for item in sources)
    return {
        "manifest": manifest_path,
        "imported_data": imported_path,
        "source_count": len(sources),
        "decision_counts": dict(decision_counts),
        "training_allowed_source_count": decision_counts.get("training_allowed", 0),
        "has_qualifying_new_training_source": decision_counts.get("training_allowed", 0) > 0,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/public_resume_sources.yaml")
    parser.add_argument("--input", default="data/external/public_resume_imports.jsonl")
    parser.add_argument(
        "--out",
        default="outputs/eval_reports/resume_source_admission_report.json",
    )
    args = parser.parse_args()
    report = build_resume_source_admission_report(args.manifest, args.input)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    write_text(args.out, rendered + "\n")


if __name__ == "__main__":
    main()
