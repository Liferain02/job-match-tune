import os

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.dataset.pipeline_freshness import (
    DERIVED_DEPENDENCIES,
    PACKAGE_ROOT,
    normalization_transform_sha256,
    verify_dependency,
    verify_normalization_manifest,
    write_normalization_manifest,
)


def _raw_row(crawl_time: str, raw_text: str = "岗位职责：负责服务开发") -> dict:
    return {
        "id": "job-1",
        "source": "example",
        "url": "https://example.test/job-1",
        "crawl_time": crawl_time,
        "job_title": "后端开发工程师",
        "company": "示例公司",
        "location": "北京",
        "salary": "",
        "raw_text": raw_text,
        "html": None,
        "meta": {"language": "zh"},
    }


def test_normalization_manifest_detects_raw_table_change(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    output_path = tmp_path / "clean.jsonl"
    manifest_path = tmp_path / "clean.manifest.json"
    init_db(db_path)
    upsert_jd_raw(db_path, [_raw_row("2026-08-09 10:00:00")])
    output_path.write_text('{"id":"job-1"}\n', encoding="utf-8")
    write_normalization_manifest(db_path, output_path, manifest_path)

    assert verify_normalization_manifest(manifest_path)["fresh"] is True
    assert normalization_transform_sha256()

    upsert_jd_raw(db_path, [_raw_row("2026-08-09 11:00:00", "岗位职责：负责平台服务开发")])
    report = verify_normalization_manifest(manifest_path)
    assert report["fresh"] is False
    assert "jd_raw 状态已变化，需重新规范化" in report["reasons"]


def test_normalization_manifest_detects_label_schema_change(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    output_path = tmp_path / "clean.jsonl"
    manifest_path = tmp_path / "clean.manifest.json"
    schema_path = tmp_path / "label_schema.yaml"
    init_db(db_path)
    upsert_jd_raw(db_path, [_raw_row("2026-08-09 10:00:00")])
    output_path.write_text('{"id":"job-1"}\n', encoding="utf-8")
    schema_path.write_text("skill_alias: {}\n", encoding="utf-8")
    write_normalization_manifest(db_path, output_path, manifest_path, schema_path)

    schema_path.write_text("skill_alias:\n  Vue: [vue]\n", encoding="utf-8")

    report = verify_normalization_manifest(manifest_path)
    assert report["fresh"] is False
    assert "JD 规范化规则或标签 schema 已变化，需重新规范化" in report["reasons"]


def test_verify_dependency_detects_output_older_than_input(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text("input\n", encoding="utf-8")
    output_path.write_text("output\n", encoding="utf-8")
    os.utime(output_path, ns=(1, 1))
    os.utime(input_path, ns=(2, 2))

    report = verify_dependency("测试依赖", (str(input_path),), (str(output_path),))

    assert report["fresh"] is False


def test_core_derived_dependencies_include_builder_code():
    dependencies = {name: inputs for name, inputs, _outputs in DERIVED_DEPENDENCIES}
    outputs = {name: derived for name, _inputs, derived in DERIVED_DEPENDENCIES}

    assert str(PACKAGE_ROOT / "preprocess" / "deduplicate.py") in dependencies["JD规范化到去重"]
    assert (
        str(PACKAGE_ROOT / "dataset" / "build_jd_quality_sft_dataset.py")
        in dependencies["JD组合池到质量集"]
    )
    assert (
        str(PACKAGE_ROOT / "dataset" / "build_multitask_sft_dataset.py")
        in dependencies["单任务SFT到多任务SFT"]
    )
    assert (
        str(PACKAGE_ROOT / "dataset" / "build_preference_bootstrap_dataset.py")
        in dependencies["多任务SFT到产品偏好数据"]
    )
    assert all(not path.endswith(".py") for derived in outputs.values() for path in derived)
