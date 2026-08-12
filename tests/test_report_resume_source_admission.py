import json
from pathlib import Path

from jobmatch_tune.eval.report_resume_source_admission import (
    build_resume_source_admission_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_template_source_is_rejected_after_content_dedup(tmp_path: Path) -> None:
    raw_path = tmp_path / "template.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    manifest_path = tmp_path / "sources.yaml"
    raw_rows = [
        {"text": "姓名：张三\n核心技能：Python\n项目经历：接口开发"}
        for _ in range(20)
    ]
    _write_jsonl(raw_path, raw_rows)
    imported_rows = [
        {
            "id": f"r{i}",
            "task": "resume_parse",
            "text": row["text"],
            "label": {
                "目标岗位": "后端开发",
                "教育背景": ["本科"],
                "核心技能": ["Python"],
                "项目经历": ["接口开发"],
            },
            "meta": {"source_name": "template_source"},
        }
        for i, row in enumerate(raw_rows)
    ]
    _write_jsonl(imported_path, imported_rows)
    manifest_path.write_text(
        "sources:\n"
        "  - name: template_source\n"
        f"    path: {raw_path}\n"
        "    schema: resume_parse_rows\n"
        "    license_status: unconfirmed\n"
        "    intended_usage: candidate_pool_only\n"
        "    provenance_status: synthetic_template_generation_documented\n"
        "    collection_method: template_generation\n"
        "    admission_decision: rejected\n"
        "    rejection_reason: template_only\n",
        encoding="utf-8",
    )

    report = build_resume_source_admission_report(str(manifest_path), str(imported_path))
    source = report["sources"][0]

    assert report["has_qualifying_new_training_source"] is False
    assert source["final_admission_decision"] == "rejected"
    assert source["usable_for_training"] is False
    assert source["unique_content_groups"] == 1
    assert source["duplicate_template_rows"] == 19


def test_sensitive_ner_rows_are_excluded_instead_of_claiming_safe_annotations(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "ner.jsonl"
    imported_path = tmp_path / "imported.jsonl"
    manifest_path = tmp_path / "sources.yaml"
    _write_jsonl(raw_path, [{"tokens": ["张", "三"]}, {"tokens": ["开", "发"]}])
    _write_jsonl(
        imported_path,
        [
            {
                "id": "sensitive",
                "task": "resume_ner",
                "text": "张三",
                "ner_tags": ["B-NAME", "I-NAME"],
                "meta": {"source_name": "ner"},
            },
            {
                "id": "clean",
                "task": "resume_ner",
                "text": "开发",
                "ner_tags": ["O", "O"],
                "meta": {"source_name": "ner"},
            },
        ],
    )
    manifest_path.write_text(
        "sources:\n"
        "  - name: ner\n"
        f"    path: {raw_path}\n"
        "    schema: resume_ner_rows\n"
        "    license_status: declared_upstream_provenance_incomplete\n"
        "    intended_usage: audit_only\n"
        "    provenance_status: source_collection_and_original_producer_undocumented\n"
        "    admission_decision: audit_only\n",
        encoding="utf-8",
    )

    source = build_resume_source_admission_report(
        str(manifest_path), str(imported_path)
    )["sources"][0]

    assert source["rows_with_pii_or_sensitive_entities"] == 1
    assert source["rows_removed_by_privacy_gate"] == 1
    assert source["number_after_privacy_clean"] == 1
    assert source["usable_for_resume_parse"] is False
    assert source["final_admission_decision"] == "audit_only"
