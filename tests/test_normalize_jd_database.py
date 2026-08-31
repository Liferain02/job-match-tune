import pytest

from jobmatch_tune.database import init_db, upsert_jd_raw
from jobmatch_tune.preprocess import normalize_jd
from jobmatch_tune.preprocess.normalize_jd import normalize_database, split_sections
from jobmatch_tune.utils.io import read_jsonl


def test_split_sections_keeps_content_before_inline_requirement_marker():
    sections = split_sections(
        "岗位描述：1、负责服务开发；2、负责性能优化。岗位要求：本科，熟悉 Python。"
    )

    assert sections["responsibilities"] == "1、负责服务开发；2、负责性能优化。"
    assert sections["requirements"] == "本科，熟悉 Python。"


def test_split_sections_supports_core_responsibility_and_does_not_treat_preferred_as_heading():
    sections = split_sections(
        "核心职责：负责模型训练与部署。\n任职要求：熟悉 Python，有 PyTorch 经验者优先。\n沟通能力良好。"
    )

    assert sections["responsibilities"] == "负责模型训练与部署。"
    assert sections["requirements"] == "熟悉 Python，有 PyTorch 经验者优先。\n沟通能力良好。"
    assert "bonus" not in sections


def test_split_sections_supports_bracketed_and_spaced_requirement_headings():
    sections = split_sections(
        "岗位描述：负责模型开发。 【任职资格】 熟悉 Python。\n"
        "职位内容：负责服务部署。 岗位要求 本科及以上学历。"
    )

    assert sections["responsibilities"] == "负责模型开发。\n负责服务部署。"
    assert sections["requirements"] == "熟悉 Python。\n本科及以上学历。"


def test_split_sections_supports_html_and_numbered_requirement_headings():
    sections = split_sections(
        "职位描述：<p>岗位描述</p><p>负责产品设计。</p>"
        "<p>任职资格要求</p><p>本科及以上学历。</p>\n"
        "岗位描述：负责客户交付。 二、任职资格 熟悉国际贸易。"
    )

    assert sections["responsibilities"] == "<p>负责产品设计。</p>\n负责客户交付。"
    assert sections["requirements"] == "<p>本科及以上学历。</p>\n熟悉国际贸易。"


def test_split_sections_repairs_clear_xiaomi_source_inversion():
    sections = split_sections(
        "岗位职责：本科及以上学历，3年以上经验，熟悉 Java，具备沟通能力。\n"
        "任职要求：负责服务设计与开发，参与架构优化，推动项目交付。",
        source="hr.xiaomi.com",
    )

    assert sections["responsibilities"].startswith("负责服务设计与开发")
    assert sections["requirements"].startswith("本科及以上学历")


def test_normalize_database_streams_batches_to_jsonl_and_sqlite(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    out_path = tmp_path / "clean.jsonl"
    init_db(db_path)
    upsert_jd_raw(
        db_path,
        [
            {
                "id": f"job-{index}",
                "source": "example",
                "url": "https://example.test/job",
                "crawl_time": "2026-08-09 00:00:00",
                "job_title": "后端开发工程师",
                "company": "示例公司",
                "location": "北京",
                "salary": "",
                "raw_text": "岗位职责：负责 Java 服务开发。任职要求：本科，3 年经验，熟悉 Java。",
                "html": None,
                "meta": {"language": "zh", "sft_ready": True},
            }
            for index in range(3)
        ],
    )
    schema = {
        "job_directions": ["后端开发"],
        "skill_alias": {"Java": ["java"]},
    }

    count = normalize_database(str(db_path), str(out_path), schema, batch_size=2)
    rows = list(read_jsonl(out_path))

    assert count == 3
    assert len(rows) == 3
    assert rows[0]["meta"]["language"] == "zh"
    assert rows[0]["labels"]["必备技能"] == ["Java"]


def test_normalize_database_preserves_previous_output_on_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.sqlite3"
    out_path = tmp_path / "clean.jsonl"
    init_db(db_path)
    upsert_jd_raw(
        db_path,
        [
            {
                "id": "job-1",
                "source": "example",
                "url": "https://example.test/job",
                "crawl_time": "2026-08-09 00:00:00",
                "job_title": "后端开发工程师",
                "company": "示例公司",
                "location": "北京",
                "salary": "",
                "raw_text": "岗位职责：负责服务开发。",
                "html": None,
                "meta": {"language": "zh"},
            }
        ],
    )
    out_path.write_text("previous output\n", encoding="utf-8")

    def fail_normalization(row, schema):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(normalize_jd, "normalize_jd_row", fail_normalization)

    with pytest.raises(RuntimeError, match="forced failure"):
        normalize_jd.normalize_database(str(db_path), str(out_path), {}, batch_size=1)

    assert out_path.read_text(encoding="utf-8") == "previous output\n"


def test_normalize_database_can_skip_unused_clean_table_sync(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.sqlite3"
    out_path = tmp_path / "clean.jsonl"
    init_db(db_path)
    upsert_jd_raw(
        db_path,
        [
            {
                "id": "job-1",
                "source": "example",
                "url": "https://example.test/job",
                "crawl_time": "2026-08-09 00:00:00",
                "job_title": "后端开发工程师",
                "company": "示例公司",
                "location": "北京",
                "salary": "",
                "raw_text": "岗位职责：负责服务开发。",
                "html": None,
                "meta": {"language": "zh"},
            }
        ],
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("jd_clean sync should be skipped")

    monkeypatch.setattr(normalize_jd, "upsert_jd_clean", fail_if_called)

    count = normalize_jd.normalize_database(
        str(db_path), str(out_path), {}, batch_size=1, sync_clean_table=False
    )

    assert count == 1
    assert len(list(read_jsonl(out_path))) == 1
