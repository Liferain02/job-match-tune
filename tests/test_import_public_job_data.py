from jobmatch_tune.crawler.import_public_job_data import (
    convert_job_educational_row,
    extract_tagged_text,
    import_sources,
)


def test_extract_tagged_text_returns_inner_value() -> None:
    text = "<岗位名称>算法工程师</岗位名称><岗位描述>负责训练与推理</岗位描述>"
    assert extract_tagged_text(text, "岗位名称") == "算法工程师"
    assert extract_tagged_text(text, "岗位描述") == "负责训练与推理"


def test_convert_job_educational_row_builds_chinese_raw_record() -> None:
    row = {
        "job_id": 123,
        "system": "从岗位中提取学历",
        "user": (
            "<岗位名称>后端开发工程师</岗位名称>"
            "<岗位描述>负责 Java 服务开发，本科及以上学历。</岗位描述>"
            "<学历描述>本科</学历描述>"
        ),
        "assistant": "本科",
        "user_short": "<岗位名称>后端开发工程师</岗位名称>",
        "ai": "本科",
        "diff": True,
    }
    converted = convert_job_educational_row(
        row=row,
        source_name="hf_job_edu",
        source_url="local.parquet",
        crawl_time="2026-05-17 00:00:00",
    )
    assert converted["id"] == "hf_job_edu_123"
    assert converted["job_title"] == "后端开发工程师"
    assert converted["meta"]["language"] == "zh"
    assert converted["meta"]["sft_ready"] is False
    assert converted["meta"]["education_label"] == "本科"
    assert "岗位描述：" in converted["raw_text"]


def test_import_sources_propagates_governance_metadata(tmp_path) -> None:
    csv_path = tmp_path / "jobs.csv"
    csv_path.write_text(
        "job_name,job_area,job_salary,com_name,work_year,education,job_benefits\n"
        "后端开发工程师,北京,20-30K,示例公司,3年,本科,五险一金\n",
        encoding="utf-8",
    )
    rows = import_sources(
        [
            {
                "name": "example",
                "type": "bosszp_csv",
                "local_path": str(csv_path),
                "source_url": "https://example.test/jobs.csv",
                "source_revision": "revision-1",
                "artifact_sha256": "a" * 64,
                "package_license_status": "unconfirmed",
                "content_rights_status": "unconfirmed",
                "intended_usage": "audit_only",
                "training_eligible": False,
                "holdout_eligible": False,
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0]["meta"]["source_revision"] == "revision-1"
    assert rows[0]["meta"]["artifact_sha256"] == "a" * 64
    assert rows[0]["meta"]["training_eligible"] is False
    assert rows[0]["meta"]["intended_usage"] == "audit_only"
