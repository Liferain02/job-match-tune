from __future__ import annotations

import io
import zipfile

import pytest
from PIL import Image
from pypdf import PdfWriter

from jobmatch_tune.api.upload_validation import UploadValidationError, validate_upload_content


def png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(output, format="PNG")
    return output.getvalue()


def pdf_bytes() -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(output)
    return output.getvalue()


def docx_bytes(*, valid: bool = True) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        if valid:
            archive.writestr(
                "word/document.xml",
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>',
            )
    return output.getvalue()


@pytest.mark.parametrize(
    ("name", "content", "detected"),
    [
        ("resume.txt", "中文简历\nPython".encode(), "text"),
        ("resume.md", b"# Resume\nPython", "text"),
        ("resume.pdf", pdf_bytes(), "pdf"),
        ("resume.docx", docx_bytes(), "docx"),
        ("resume.png", png_bytes(), "image/png"),
    ],
)
def test_valid_supported_content(name: str, content: bytes, detected: str) -> None:
    assert validate_upload_content(name, content).detected_type == detected


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("resume.txt", pdf_bytes()),
        ("resume.pdf", png_bytes()),
        ("resume.png", pdf_bytes()),
        ("resume.jpg", png_bytes()),
    ],
)
def test_extension_content_mismatch_is_rejected(name: str, content: bytes) -> None:
    with pytest.raises(UploadValidationError) as captured:
        validate_upload_content(name, content)
    assert captured.value.code == "extension_content_mismatch"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("resume.pdf", b"not a pdf"),
        ("resume.docx", docx_bytes(valid=False)),
        ("resume.png", b"not an image"),
        ("resume.txt", b"binary\x00payload"),
        ("resume.txt", b"\xff\xfe"),
    ],
)
def test_corrupt_content_is_classified(name: str, content: bytes) -> None:
    with pytest.raises(UploadValidationError) as captured:
        validate_upload_content(name, content)
    assert captured.value.code == "corrupt_file"


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(UploadValidationError) as captured:
        validate_upload_content("resume.exe", b"MZ payload")
    assert captured.value.code == "unsupported_type"
