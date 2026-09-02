import json

from jobmatch_tune.dataset.import_public_training_data import (
    _evidence_preserving_excerpt,
    _match_level,
    _pdf_to_text,
    _public_source_to_text,
    _source_asset_urls,
    build_faircv_resume_rows,
    build_human_reviewed_public_match_rows,
    build_netsol_match_rows,
    build_public_web_resume_rows,
    extract_public_resume_sections,
    html_to_visible_text,
    refresh_public_resume_snapshots,
)


def test_public_image_source_uses_existing_ocr_pipeline(monkeypatch):
    seen = {}

    def fake_ocr(path):
        seen["bytes"] = path.read_bytes()
        return "教育背景\n某大学 计算机科学与技术 本科"

    monkeypatch.setattr(
        "jobmatch_tune.dataset.import_public_training_data.ocr_image_file",
        fake_ocr,
    )

    text, method = _public_source_to_text(b"image bytes", "image")

    assert text == "教育背景\n某大学 计算机科学与技术 本科"
    assert method == "image_ocr"
    assert seen["bytes"] == b"image bytes"


def test_public_source_supports_multi_image_resume_assets():
    assert _source_asset_urls({"urls": ["https://example.test/1.jpg", "https://example.test/2.jpg"]}) == [
        "https://example.test/1.jpg",
        "https://example.test/2.jpg",
    ]


def test_refresh_public_resume_snapshots_can_skip_existing_cache(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        """
sources:
  - id: cached
    url: https://example.invalid/resume.pdf
    format: pdf
    training_decision: allowed_after_deidentification
""",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    cache.joinpath("cached.json").write_text("{}", encoding="utf-8")

    report = refresh_public_resume_snapshots(manifest, cache, skip_existing=True)

    assert report == {
        "requested": 1,
        "refreshed": 0,
        "failed": [],
        "already_cached": 1,
    }


def test_public_pdf_falls_back_to_ocr_when_font_text_is_unusable(monkeypatch):
    def fake_pdftotext(command, **kwargs):
        del kwargs
        from pathlib import Path

        Path(command[-1]).write_text("13021008001 Word2Vec LR GBDT", encoding="utf-8")

    monkeypatch.setattr(
        "jobmatch_tune.dataset.import_public_training_data.subprocess.run",
        fake_pdftotext,
    )
    monkeypatch.setattr(
        "jobmatch_tune.dataset.import_public_training_data.ocr_pdf_file",
        lambda path: "教育经历\n某大学 计算机硕士\n项目经历\n推荐系统项目",
    )

    assert _pdf_to_text(b"pdf bytes").startswith("教育经历")


def test_evidence_preserving_excerpt_keeps_late_label_evidence():
    text = "intro " * 300 + "Bachelor degree project using Python and PostgreSQL."

    excerpt = _evidence_preserving_excerpt(
        text,
        evidence=["Bachelor degree project using Python and PostgreSQL."],
        head_chars=100,
        max_chars=220,
    )

    assert len(excerpt) <= 220
    assert "Bachelor degree project using Python and PostgreSQL." in excerpt


def test_match_level_uses_four_stable_bands():
    assert [_match_level(score) for score in (2.9, 3.0, 5.5, 7.5)] == [
        "低匹配",
        "基本匹配",
        "较匹配",
        "高匹配",
    ]


def test_build_netsol_match_rows_redacts_and_deduplicates(tmp_path):
    payload = {
        "input": {
            "job_description": "Backend role requiring Python, PostgreSQL, APIs and monitoring. " * 4,
            "resume": (
                "Name: Jane Doe\nEmail: jane@example.com\nPhone: +1 415-555-0101\n"
                + "Backend engineer using Python, PostgreSQL, APIs and monitoring. " * 4
            ),
        },
        "output": {
            "valid_resume_and_jd": True,
            "personal_info": {"current_position": "Backend Engineer"},
            "scores": {
                "aggregated_scores": {"macro_scores": 8.0, "micro_scores": 7.0},
                "macro_scores": [{"criteria": "Python", "score": 9}],
                "micro_scores": [{"criteria": "Kubernetes", "score": 3}],
                "requirements": [
                    {"criteria": "Bachelor degree", "meets": True},
                    {"criteria": "3 years experience", "meets": False},
                ],
            },
            "justification": ["Strong Python evidence; Kubernetes evidence is missing."],
        },
        "details": {"projects": [{"title": "API", "description": "Built a backend API"}]},
    }
    for name in ("match_0.json", "match_1.json"):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    rows, report = build_netsol_match_rows(tmp_path)

    assert len(rows) == 1
    assert report["selected_pairs"] == 1
    assert report["duplicate_pairs"] == 1
    row = rows[0]
    assert "jane@example.com" not in row["resume_text"]
    assert "415-555-0101" not in row["resume_text"]
    assert row["label"]["匹配等级"] == "高匹配"
    assert row["label"]["命中技能"] == ["Python"]
    assert row["label"]["缺失技能"] == ["Kubernetes"]
    assert row["meta"]["entity_split"] == "train"
    assert len(row["jd_text"]) <= 1100
    assert len(row["resume_text"]) <= 1400


def test_build_faircv_rows_keeps_only_deidentified_technical_templates(tmp_path):
    source = tmp_path / "templates.json"
    source.write_text(
        json.dumps(
            {
                "resumes": [
                    {
                        "metadata": {
                            "position": "后端开发工程师",
                            "skill_level": "中",
                        },
                        "content": (
                            "### 个人信息\n姓名：张三\n年龄：22\n邮箱：demo@example.com\n"
                            "### 教育背景\n某大学 计算机科学与技术 本科\n"
                            "### 专业技能\n熟悉 Python、MySQL、Redis 和 Linux\n"
                            "### 工作经历\n负责后端服务开发、接口治理、监控告警、容量评估和线上故障复盘，"
                            "持续改进发布流程与服务稳定性。\n"
                            "### 项目经验\n项目名称：订单服务\n负责开发接口并优化数据库查询性能\n"
                            "实现缓存更新与消息重试机制，完成压力测试、慢查询治理和链路追踪接入。\n"
                            "### 其他亮点\n具有开源项目协作经验\n"
                            "### 自我评价\n重视技术方案中的证据、边界条件和可维护性，"
                            "能够独立完成需求分析、开发、测试和上线复盘。\n"
                        ),
                    },
                    {
                        "metadata": {
                            "position": "业务产品经理",
                            "skill_level": "中",
                        },
                        "content": "### 教育背景\n某大学 本科\n### 项目经验\n产品项目",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    schema = {
        "skill_alias": {
            "Python": ["python"],
            "MySQL": ["mysql"],
            "Redis": ["redis"],
            "Linux": ["linux"],
        }
    }

    rows, report = build_faircv_resume_rows(source, schema=schema)

    assert len(rows) == 1
    assert report["non_technical_or_product_rows"] == 1
    row = rows[0]
    assert row["label"]["目标岗位"] == "后端开发"
    assert row["label"]["核心技能"] == ["Python", "MySQL", "Redis", "Linux"]
    assert "张三" not in row["text"]
    assert "demo@example.com" not in row["text"]
    assert "个人信息" not in row["text"]
    assert row["meta"]["language"] == "zh"
    assert row["meta"]["domain"] == "technical_jobs"


def test_public_html_extraction_skips_scripts_and_private_sections():
    visible = html_to_visible_text(
        """
        <html><script>教师资格证 试讲 班级管理</script><body>
        <h2>基本信息</h2><p>姓名：张三</p><p>手机：13800138000</p>
        <h2>教育背景</h2><p>某大学 软件工程 本科</p>
        <h2>专业技能</h2><p>熟悉 Java、MySQL、Redis 和 Linux</p>
        <p>项目管理工具：Jira</p><p>个人微信</p>
        <h2>工作经历</h2><p>新东方教育科技集团有限公司</p>
        <p>负责设计测试用例并完成缺陷跟踪和回归验证。</p>
        <h2>项目经历</h2><p>订单系统：负责接口开发并优化数据库查询。</p>
        </body></html>
        """.encode()
    )

    sections = extract_public_resume_sections(visible)

    assert sections["education"] == ["某大学 软件工程 本科"]
    assert sections["skills"] == ["熟悉 Java、MySQL、Redis 和 Linux", "项目管理工具：Jira"]
    assert sections["work"] == [
        "新东方教育科技集团有限公司",
        "负责设计测试用例并完成缺陷跟踪和回归验证。",
    ]
    assert sections["projects"] == ["订单系统：负责接口开发并优化数据库查询。"]
    assert "教师资格证" not in visible
    assert all("张三" not in line for lines in sections.values() for line in lines)
    assert all("13800138000" not in line for lines in sections.values() for line in lines)


def test_public_sections_drop_publication_author_lists_and_footer_identity():
    text = """教育背景
示例大学 计算机硕士
科研成果
Henghua Zhang, Jue Chen, Yuhang Wu, and Yujie Xiong. Paper title, 2026.
© 2026 张三 · GitHub Pages
工作经历
负责推荐服务开发与性能优化
"""

    sections = extract_public_resume_sections(text)

    flattened = "\n".join(line for values in sections.values() for line in values)
    assert "Henghua Zhang" not in flattened
    assert "张三" not in flattened
    assert "负责推荐服务开发与性能优化" in flattened


def test_public_sections_accept_markdown_and_org_headings():
    text = """## 教育背景
某大学 计算机科学与技术 本科
* 专业技能
熟悉 Go、MySQL、Redis 和 Linux
##### 项目经历
订单系统：使用 Go 开发接口并通过 Redis 优化查询性能。
"""

    sections = extract_public_resume_sections(text)

    assert sections["education"] == ["某大学 计算机科学与技术 本科"]
    assert sections["skills"] == ["熟悉 Go、MySQL、Redis 和 Linux"]
    assert sections["projects"] == [
        "订单系统：使用 Go 开发接口并通过 Redis 优化查询性能。"
    ]


def test_build_public_web_resume_rows_uses_extracts_not_generated_labels(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        """
sources:
  - id: backend_one
    url: https://example.test/resume.png
    source_page_url: https://example.test/public-post
    format: html
    target_direction: 后端开发
    license_status: not_stated_no_explicit_prohibition
    training_decision: allowed_after_deidentification
""",
        encoding="utf-8",
    )
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    snapshot_dir.joinpath("backend_one.json").write_text(
        json.dumps(
            {
                "id": "backend_one",
                "url": "https://example.test/resume",
                "fetched_at": "2026-09-01",
                "content_sha256": "abc",
                "visible_text": "\n".join(
                    [
                        "基本信息",
                        "姓名：张三",
                        "邮箱：person@example.com",
                        "教育背景",
                        "某大学 软件工程 本科",
                        "专业技能",
                        "熟悉 Java、MySQL、Redis、Linux 和 Docker，能够完成服务开发与排障。",
                        "工作经历",
                        "在技术团队负责接口治理、监控告警和线上故障复盘。",
                        "参与需求评审、方案设计、代码审查、灰度发布和容量评估，维护服务稳定性。",
                        "项目经历",
                        "订单服务项目：使用 Java 和 MySQL 开发接口并优化慢查询。",
                        "设计 Redis 缓存与消息重试机制，完成压力测试和链路追踪接入。",
                        "通过索引治理和批处理改造降低接口延迟，并补充自动化回归测试。",
                        "其他亮点",
                        "持续参与开源协作，重视可验证证据和工程质量。",
                    ]
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    schema = {
        "skill_alias": {
            "Java": ["java"],
            "MySQL": ["mysql"],
            "Redis": ["redis"],
            "Linux": ["linux"],
            "Docker": ["docker"],
        }
    }

    rows, report = build_public_web_resume_rows(manifest, snapshot_dir, schema=schema)

    assert report["selected_rows"] == 1
    assert report["selected_direction_counts"] == {"后端开发": 1}
    assert report["selected_extraction_method_counts"] == {"unknown": 1}
    row = rows[0]
    assert row["source_type"] == "public_real_self_published_anonymized"
    assert row["label"]["目标岗位"] == "后端开发"
    assert row["label"]["教育背景"] == ["某大学 软件工程 本科"]
    assert row["label"]["项目经历"][0].startswith("订单服务项目")
    assert row["label"]["核心技能"] == ["Java", "MySQL", "Redis", "Linux", "Docker"]
    assert "张三" not in row["text"]
    assert "person@example.com" not in row["text"]
    assert row["meta"]["annotation_status"] == "human_target_mapping_plus_extractive_labels"
    assert row["meta"]["source_url"] == "https://example.test/public-post"
    assert row["meta"]["source_asset_url"] == "https://example.test/resume.png"


def test_public_resume_with_work_evidence_does_not_require_student_project(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        """
sources:
  - id: test_engineer
    urls:
      - https://example.test/page-1.jpg
      - https://example.test/page-2.jpg
    source_page_url: https://example.test/public-post
    format: image
    target_direction: 测试开发
    training_decision: allowed_after_deidentification
""",
        encoding="utf-8",
    )
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    snapshot_dir.joinpath("test_engineer.json").write_text(
        json.dumps(
            {
                "sections": {
                    "education": ["某大学 电子信息专业 硕士研究生"],
                    "skills": ["熟悉 Python、Linux、Git、自动化测试和接口测试"],
                    "work": [
                        "负责搭建 Python 自动化测试框架，持续运行回归任务并定位偶发缺陷。",
                        "设计接口与压力测试场景，完成缺陷跟踪、回归验证和测试报告。",
                        "基于 Linux 环境完成日志采集、进程监控和性能数据分析，推动研发修复问题。",
                        "参与需求评审并设计边界值、异常链路和兼容性用例，维护持续集成回归任务。",
                        "使用 Git 管理测试脚本，整理环境部署文档并复盘线上缺陷的根因和影响范围。",
                    ],
                },
                "content_sha256": "abc",
                "fetched_at": "2026-09-01",
                "extraction_method": "image_ocr",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    schema = {
        "skill_alias": {
            "Python": ["python"],
            "Linux": ["linux"],
            "Git": ["git"],
        }
    }

    rows, report = build_public_web_resume_rows(manifest, snapshot_dir, schema=schema)

    assert report["selected_rows"] == 1
    assert rows[0]["label"]["项目经历"] == []
    assert rows[0]["meta"]["source_asset_urls"] == [
        "https://example.test/page-1.jpg",
        "https://example.test/page-2.jpg",
    ]


def test_public_resume_recovers_explicit_education_from_merged_work_section(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        """
sources:
  - id: merged_headings
    url: https://example.test/resume.pdf
    format: pdf
    target_direction: AI Infra
    training_decision: allowed_after_deidentification
""",
        encoding="utf-8",
    )
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    snapshot_dir.joinpath("merged_headings.json").write_text(
        json.dumps(
            {
                "sections": {
                    "work": [
                        "2022.09 - 2025.06 某理工大学 计算机系统结构 硕士研究生",
                        "负责 Python 训练任务编排、Linux 环境排障和 GPU 资源监控。",
                        "实现失败任务重试与队列优先级，并完成容量压测和故障复盘。",
                        "参与训练作业的发布、监控和异常排查，整理资源使用报表并优化告警阈值。",
                        "维护容器镜像与配置清单，跟进上线后的任务成功率和排队时间。",
                    ],
                    "skills": ["Python、Linux、Docker、Kubernetes"],
                    "projects": [
                        "训练平台项目：基于 Kubernetes 实现任务发布、失败重试和资源回收。"
                    ],
                },
                "content_sha256": "abc",
                "fetched_at": "2026-09-02",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    schema = {
        "skill_alias": {
            "Python": ["python"],
            "Linux": ["linux"],
            "Docker": ["docker"],
            "Kubernetes": ["kubernetes"],
        }
    }

    rows, report = build_public_web_resume_rows(manifest, snapshot_dir, schema=schema)

    assert report["selected_rows"] == 1
    assert rows[0]["label"]["教育背景"] == [
        "2022.09 - 2025.06 某理工大学 计算机系统结构 硕士研究生"
    ]


def test_public_resume_quality_report_exposes_rejection_reasons(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        """
sources:
  - id: incomplete
    url: https://example.test/incomplete
    format: html
    target_direction: 后端开发
    training_decision: allowed_after_deidentification
""",
        encoding="utf-8",
    )
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    snapshot_dir.joinpath("incomplete.json").write_text(
        json.dumps({"sections": {"skills": ["熟悉 Python"]}}, ensure_ascii=False),
        encoding="utf-8",
    )

    rows, report = build_public_web_resume_rows(manifest, snapshot_dir, schema={})

    assert rows == []
    assert report["quality_filtered"] == 1
    assert report["quality_filtered_short_text"] == 1
    assert report["quality_filtered_insufficient_skills"] == 1
    assert report["quality_filtered_missing_education"] == 1
    assert report["quality_filtered_missing_project_or_work_evidence"] == 1


def test_human_reviewed_public_pair_is_training_only_and_not_observed_outcome(tmp_path):
    manifest = tmp_path / "pairs.yaml"
    manifest.write_text(
        """
pairs:
  - id: pair_one
    resume_source_id: resume_one
    jd_id: jd_one
    jd_direction: 后端开发
    level: 较匹配
    direction_match: true
    education_match: true
    experience_match: false
    matched_skills: [Java, MySQL]
    missing_skills: [Redis]
    matched_projects: [订单服务]
    rationale: 技能和项目相关，但没有满足岗位要求的正式工作年限。
""",
        encoding="utf-8",
    )
    jd_pool = tmp_path / "jds.jsonl"
    jd_pool.write_text(
        json.dumps(
            {
                "id": "jd_one",
                "raw_text": (
                    "岗位名称：Java后端工程师\n任务类型：从岗位中提取学历\n岗位描述："
                    + "负责订单服务开发，要求本科、Java、MySQL、Redis 和三年经验。" * 3
                    + "\n学历提示：本科"
                ),
                "meta": {"source_file": "https://example.test/jobs"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    resume_rows = [
        {
            "text": "目标岗位：后端开发\n教育背景：某大学本科\n技能：Java、MySQL\n项目：订单服务接口开发。",
            "label": {"目标岗位": "后端开发"},
            "meta": {
                "source_reference_id": "resume_one",
                "source_url": "https://example.test/resume",
            },
        }
    ]

    rows, report = build_human_reviewed_public_match_rows(manifest, resume_rows, jd_pool)

    assert report["selected_pairs"] == 1
    row = rows[0]
    assert "任务类型" not in row["jd_text"]
    assert "学历提示" not in row["jd_text"]
    assert row["analysis"]["匹配结论"] == "技能和项目相关，但没有满足岗位要求的正式工作年限。"
    assert row["meta"]["entity_split"] == "train"
    assert row["meta"]["observed_outcome"] is False
    assert row["meta"]["annotation_status"] == "human_reviewed_pair_v1"
