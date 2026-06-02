from __future__ import annotations

from jobmatch_tune.api.server import (
    ModelService,
    match_uploaded_inputs,
    parse_uploaded_document_bytes,
    parse_uploaded_resume_bytes,
)


def test_parse_uploaded_resume_text(monkeypatch) -> None:
    service = ModelService()

    def fake_parse(request):
        assert request.task == "resume_parse"
        assert "教育背景" in request.text
        return {
            "ok": True,
            "data": {
                "目标岗位": "后端开发",
                "教育背景": ["本科，软件工程"],
                "核心技能": ["Python", "FastAPI"],
                "实习经历": [],
                "项目经历": ["服务平台开发"],
                "优势标签": ["工程能力扎实"],
            },
            "raw_output": "{}",
            "latency_seconds": 0.01,
        }

    monkeypatch.setattr(service, "parse", fake_parse)

    result = parse_uploaded_resume_bytes(
        service,
        file_name="resume.txt",
        content=(
            "姓名：张三\n"
            "目标岗位：后端开发\n"
            "教育背景\n"
            "本科，软件工程\n"
            "核心技能\n"
            "Python FastAPI\n"
            "项目经历\n"
            "服务平台开发"
        ).encode("utf-8"),
    )

    assert result["ok"] is True
    assert result["file_name"] == "resume.txt"
    assert result["ingest"]["source_type"] == "text"
    assert result["data"]["目标岗位"] == "后端开发"


def test_parse_uploaded_resume_image_without_ocr() -> None:
    service = ModelService()
    result = parse_uploaded_resume_bytes(
        service,
        file_name="resume.png",
        content=b"fake-image",
    )
    assert result["ok"] is False
    assert result["needs_ocr"] is True
    assert result["error"] == "image_resume_requires_ocr"


def test_parse_uploaded_resume_image_with_ocr_sidecar(monkeypatch) -> None:
    service = ModelService()

    def fake_parse(request):
        return {
            "ok": True,
            "data": {
                "目标岗位": "AI应用开发",
                "教育背景": ["本科，计算机科学与技术"],
                "核心技能": ["Python", "RAG"],
                "实习经历": [],
                "项目经历": ["问答系统"],
                "优势标签": ["有项目经验"],
            },
            "raw_output": "{}",
            "latency_seconds": 0.01,
        }

    monkeypatch.setattr(service, "parse", fake_parse)

    result = parse_uploaded_resume_bytes(
        service,
        file_name="resume.png",
        content=b"fake-image",
        ocr_text="目标岗位：AI应用开发\n教育背景\n本科，计算机科学与技术\n项目经历\n问答系统",
    )
    assert result["ok"] is True
    assert result["ingest"]["ocr_used"] is True
    assert result["ingest"]["ocr_source"] == "sidecar"


def test_parse_uploaded_jd_text_file(monkeypatch) -> None:
    service = ModelService()

    def fake_parse(request):
        assert request.task == "jd_parse"
        assert "岗位职责" in request.text
        return {
            "ok": True,
            "data": {
                "岗位方向": "后端开发",
                "核心职责": ["负责服务端接口开发"],
                "必备技能": ["Python"],
                "加分项": [],
                "经验要求": "三年以上工作经验",
                "学历要求": "本科及以上",
            },
            "raw_output": "{}",
            "latency_seconds": 0.01,
        }

    monkeypatch.setattr(service, "parse", fake_parse)

    result = parse_uploaded_document_bytes(
        service,
        task="jd_parse",
        file_name="jd.txt",
        content="岗位职责：负责服务端接口开发\n任职要求：熟悉 Python".encode(),
    )

    assert result["ok"] is True
    assert result["task"] == "jd_parse"
    assert result["ingest"]["source_type"] == "text"
    assert result["data"]["岗位方向"] == "后端开发"


def test_match_uploaded_inputs_with_jd_and_resume_files(monkeypatch) -> None:
    service = ModelService()

    def fake_match(request):
        assert "岗位职责" in request.jd_text
        assert "项目经历" in request.resume_text
        return {
            "ok": True,
            "jd_parse": {"岗位方向": "AI应用开发"},
            "resume_parse": {"目标岗位": "AI应用开发"},
            "rule_result": {"匹配分数": 90, "匹配等级": "高匹配"},
            "analysis": {"匹配结论": "高度匹配"},
            "latency_seconds": 0.02,
        }

    monkeypatch.setattr(service, "match", fake_match)

    result = match_uploaded_inputs(
        service,
        jd_file_name="jd.txt",
        jd_content="岗位职责：负责 RAG 应用开发".encode(),
        resume_file_name="resume.txt",
        resume_content="目标岗位：AI应用开发\n项目经历\n知识库问答系统".encode(),
    )

    assert result["ok"] is True
    assert result["inputs"]["jd"]["source"] == "file"
    assert result["inputs"]["resume"]["source"] == "file"
    assert result["rule_result"]["匹配等级"] == "高匹配"
