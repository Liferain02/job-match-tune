from __future__ import annotations

from jobmatch_tune.preprocess.deduplicate import deduplicate_rows


def test_deduplicate_rows_removes_exact_duplicates() -> None:
    rows = [
        {"id": "1", "source": "jd", "company": "京东", "job_title": "后端开发工程师", "location": "北京", "clean_text": "岗位职责：负责服务开发 任职要求：熟悉 Java"},
        {"id": "2", "source": "jd", "company": "京东", "job_title": "后端开发工程师", "location": "北京", "clean_text": "岗位职责：负责服务开发 任职要求：熟悉 Java"},
    ]
    unique = deduplicate_rows(rows)
    assert [row["id"] for row in unique] == ["1"]


def test_deduplicate_rows_removes_near_duplicates_in_same_bucket() -> None:
    rows = [
        {
            "id": "1",
            "source": "jd",
            "company": "京东",
            "job_title": "后端开发工程师",
            "location": "北京",
            "clean_text": "岗位职责：负责服务开发、接口设计和性能优化。任职要求：熟悉 Java、MySQL、Linux，具备分布式系统经验。",
        },
        {
            "id": "2",
            "source": "jd",
            "company": "京东",
            "job_title": "后端开发工程师",
            "location": "北京",
            "clean_text": "岗位职责：负责服务开发、接口设计、性能优化。任职要求：熟悉 Java / MySQL / Linux，并具备分布式系统经验。",
        },
    ]
    unique = deduplicate_rows(rows, near_threshold=0.85)
    assert [row["id"] for row in unique] == ["1"]


def test_deduplicate_rows_keeps_similar_text_when_title_differs() -> None:
    rows = [
        {
            "id": "1",
            "source": "jd",
            "company": "京东",
            "job_title": "后端开发工程师",
            "location": "北京",
            "clean_text": "岗位职责：负责服务开发、接口设计和性能优化。任职要求：熟悉 Java、MySQL、Linux。",
        },
        {
            "id": "2",
            "source": "jd",
            "company": "京东",
            "job_title": "测试开发工程师",
            "location": "北京",
            "clean_text": "岗位职责：负责服务开发、接口设计和性能优化。任职要求：熟悉 Java、MySQL、Linux。",
        },
    ]
    unique = deduplicate_rows(rows, near_threshold=0.85)
    assert [row["id"] for row in unique] == ["1", "2"]
