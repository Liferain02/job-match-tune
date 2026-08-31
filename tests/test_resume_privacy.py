from jobmatch_tune.resume.privacy import (
    build_resume_privacy_report,
    detect_resume_pii,
    redact_resume_pii,
    redact_resume_metadata,
    sanitize_resume_row,
    sanitize_resume_text_for_training,
)


def test_detect_resume_pii_finds_common_private_fields():
    text = "\n".join(
        [
            "李四",
            "电话：13812345678",
            "邮箱：candidate@example.com",
            "微信: liferain02",
            "QQ：123456789",
            "身份证号：110101199001011234",
            "通讯地址：北京市海淀区示例路 1 号",
            "年龄：23岁",
            "专业技能：Java Spring Redis",
        ]
    )
    findings = detect_resume_pii(text)
    counts = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    assert counts["phone"] == 1
    assert counts["email"] == 1
    assert counts["wechat"] == 1
    assert counts["qq"] == 1
    assert counts["id_number"] == 1
    assert counts["address"] == 1
    assert counts["age"] == 1
    assert counts["name"] == 1


def test_redact_resume_pii_masks_values_without_dropping_resume_content():
    text = "张三\n电话：13812345678\n邮箱：candidate@example.com\n项目经历：RAG 系统"
    redacted = redact_resume_pii(text)
    assert "13812345678" not in redacted
    assert "candidate@example.com" not in redacted
    assert "张三" not in redacted
    assert "[手机号]" in redacted
    assert "[邮箱]" in redacted
    assert "[姓名]" in redacted
    assert "RAG 系统" in redacted


def test_training_sanitizer_removes_personal_profile_links():
    text = (
        "GitHub：https://github.com/example-user\n"
        "个人主页：www.example-user.dev\n"
        "核心技能：Python、FastAPI"
    )

    findings = detect_resume_pii(text)
    sanitized = sanitize_resume_text_for_training(text)

    assert sum(finding.kind == "personal_url" for finding in findings) == 2
    assert "example-user" not in sanitized
    assert "核心技能：Python、FastAPI" in sanitized


def test_redact_resume_pii_masks_name_near_contact_block():
    text = "项目经历：RAG 系统\n李四\n电话：13812345678 | 邮箱：candidate@example.com"
    redacted = redact_resume_pii(text)
    assert "李四" not in redacted
    assert "[姓名]" in redacted
    assert "RAG 系统" in redacted


def test_detect_resume_pii_does_not_treat_role_short_line_as_name():
    text = "## 目标岗位\n后端开发\n## 核心技能\nJava Spring Redis"
    assert detect_resume_pii(text) == []


def test_redact_resume_metadata_masks_name_in_file_name():
    assert redact_resume_metadata("个人简历-李四.pdf") == "个人简历-[姓名].pdf"
    assert redact_resume_metadata("张三简历.pdf") == "[姓名]简历.pdf"


def test_sanitize_resume_row_redacts_text_fields_and_sections():
    row = {
        "id": "r1",
        "file_name": "个人简历-王五.pdf",
        "file_path": "docs/个人简历-王五.pdf",
        "clean_text": "王五\n电话：13900001111\n核心技能：Python",
        "sections": {"header": "王五\n电话：13900001111", "skills": "Python"},
    }
    sanitized = sanitize_resume_row(row)
    assert sanitized["privacy"]["has_pii"] is True
    assert sanitized["privacy"]["redacted"] is True
    assert sanitized["file_name"] == "个人简历-[姓名].pdf"
    assert sanitized["file_path"] == "docs/个人简历-[姓名].pdf"
    assert "13900001111" not in sanitized["clean_text"]
    assert "王五" not in sanitized["sections"]["header"]
    assert sanitized["sections"]["skills"] == "Python"


def test_build_resume_privacy_report_summarizes_rows():
    rows = [
        {"id": "r1", "clean_text": "电话：13900001111"},
        {"id": "r2", "clean_text": "核心技能：Java"},
    ]
    report = build_resume_privacy_report(rows=rows)
    assert report["row_count"] == 2
    assert report["rows_with_pii"] == 1
    assert report["pii_counts"]["phone"] == 1


def test_training_sanitizer_removes_markdown_profile_and_sensitive_attributes():
    text = "\n".join(
        [
            "- **姓名**：张斌",
            "- **电话**：138-1234-5678",
            "- **邮箱**：张斌@163.com",
            "- **婚姻状况**：离异",
            "- **身体状况**：肢体四级残疾",
            "- **政治面貌**：群众",
            "- **身份证号**：110101199001011234",
            "- **现居住地**：北京市海淀区示例路 1 号",
            "### 教育背景",
            "本科，计算机科学与技术",
            "### 项目经历",
            "负责 Python 服务开发",
        ]
    )

    sanitized = sanitize_resume_text_for_training(text)

    for private_value in (
        "张斌",
        "138-1234-5678",
        "离异",
        "肢体四级残疾",
        "群众",
        "110101199001011234",
        "北京市海淀区示例路 1 号",
    ):
        assert private_value not in sanitized
    assert "本科，计算机科学与技术" in sanitized
    assert "负责 Python 服务开发" in sanitized
