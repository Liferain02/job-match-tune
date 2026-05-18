from pathlib import Path
from types import SimpleNamespace
import sys

from jobmatch_tune.resume.ocr import (
    available_ocr_backend,
    collect_ocr_targets,
    generate_sidecar_ocr,
    sidecar_output_path,
)


def test_available_ocr_backend_prefers_rapidocr(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=object))
    assert available_ocr_backend() == "rapidocr"


def test_sidecar_output_path_uses_out_dir(tmp_path: Path):
    source = tmp_path / "resume.pdf"
    out_dir = tmp_path / "ocr"
    path = sidecar_output_path(source, out_dir=out_dir)
    assert path == out_dir / "resume.pdf.ocr.txt"


def test_collect_ocr_targets_filters_files(tmp_path: Path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.png").write_bytes(b"img")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    targets = collect_ocr_targets(tmp_path)
    assert [path.name for path in targets] == ["a.pdf", "b.png"]


def test_generate_sidecar_ocr_for_image(monkeypatch, tmp_path: Path):
    image = tmp_path / "resume.png"
    image.write_bytes(b"img")
    monkeypatch.setattr("jobmatch_tune.resume.ocr.ocr_image_file", lambda path: "目标岗位：后端开发工程师")
    out = generate_sidecar_ocr(image, out_dir=tmp_path / "out")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "目标岗位：后端开发工程师"


def test_generate_sidecar_ocr_for_pdf(monkeypatch, tmp_path: Path):
    pdf = tmp_path / "resume.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr("jobmatch_tune.resume.ocr.ocr_pdf_file", lambda path: "教育背景：本科")
    out = generate_sidecar_ocr(pdf, out_dir=tmp_path / "out")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "教育背景：本科"
