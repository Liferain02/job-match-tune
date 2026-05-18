from __future__ import annotations

from jobmatch_tune.api.server import ModelService, parse_uploaded_resume_bytes


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
