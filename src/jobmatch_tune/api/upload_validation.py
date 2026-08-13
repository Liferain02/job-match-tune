from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | {".pdf", ".docx"}
IMAGE_FORMAT_EXTENSIONS = {
    "PNG": {".png"},
    "JPEG": {".jpg", ".jpeg"},
    "WEBP": {".webp"},
    "BMP": {".bmp"},
}


class UploadValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class UploadDescriptor:
    extension: str
    detected_type: str
    text_encoding: str = ""


def _looks_like_pdf(content: bytes) -> bool:
    return content.startswith(b"%PDF-")


def _looks_like_zip(content: bytes) -> bool:
    return content.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))


def _detected_image_format(content: bytes) -> str:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as image:
            image.verify()
            return str(image.format or "").upper()
    except Exception:
        return ""


def _detect_supported_type(content: bytes) -> str:
    if _looks_like_pdf(content):
        return "pdf"
    if _looks_like_zip(content):
        return "zip"
    image_format = _detected_image_format(content)
    if image_format:
        return f"image/{image_format.lower()}"
    return "unknown"


def _validate_pdf(content: bytes) -> None:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        len(reader.pages)
    except Exception as exc:
        raise UploadValidationError("corrupt_file", "PDF container cannot be parsed") from exc


def _validate_docx(content: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = set(archive.namelist())
            required = {"[Content_Types].xml", "word/document.xml"}
            if not required.issubset(names):
                raise UploadValidationError(
                    "corrupt_file",
                    "DOCX ZIP container is missing required OOXML entries",
                )
            document = archive.read("word/document.xml")
            if b"<w:document" not in document and b"<document" not in document:
                raise UploadValidationError("corrupt_file", "DOCX document.xml is invalid")
    except UploadValidationError:
        raise
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise UploadValidationError("corrupt_file", "DOCX container cannot be parsed") from exc


def _validate_image(extension: str, content: bytes) -> str:
    image_format = _detected_image_format(content)
    if not image_format:
        raise UploadValidationError("corrupt_file", "image content cannot be decoded")
    allowed_extensions = IMAGE_FORMAT_EXTENSIONS.get(image_format, set())
    if extension not in allowed_extensions:
        raise UploadValidationError(
            "extension_content_mismatch",
            f"{extension} does not match decoded {image_format} image content",
        )
    return image_format


def _validate_text(content: bytes) -> str:
    if b"\x00" in content:
        raise UploadValidationError("corrupt_file", "text file contains NUL bytes")
    try:
        text = content.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError as exc:
        raise UploadValidationError("corrupt_file", "text file must be valid UTF-8") from exc
    if not text.strip():
        raise UploadValidationError("corrupt_file", "text file is empty after decoding")
    control_count = sum(ord(char) < 32 and char not in "\n\r\t" for char in text)
    if control_count / max(len(text), 1) > 0.02:
        raise UploadValidationError("corrupt_file", "text file contains too many control bytes")
    return encoding


def validate_upload_content(file_name: str, content: bytes) -> UploadDescriptor:
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError(
            "unsupported_type",
            f"unsupported extension: {extension or '<missing>'}",
        )
    if not content:
        raise UploadValidationError("corrupt_file", "uploaded file is empty")

    detected = _detect_supported_type(content)
    if extension in TEXT_EXTENSIONS:
        if detected != "unknown":
            raise UploadValidationError(
                "extension_content_mismatch",
                f"text extension does not match detected {detected} content",
            )
        encoding = _validate_text(content)
        return UploadDescriptor(extension, "text", encoding)
    if extension == ".pdf":
        if detected != "pdf":
            if detected != "unknown":
                raise UploadValidationError(
                    "extension_content_mismatch",
                    f".pdf does not match detected {detected} content",
                )
            raise UploadValidationError("corrupt_file", "PDF signature is missing")
        _validate_pdf(content)
        return UploadDescriptor(extension, "pdf")
    if extension == ".docx":
        if detected != "zip":
            if detected != "unknown":
                raise UploadValidationError(
                    "extension_content_mismatch",
                    f".docx does not match detected {detected} content",
                )
            raise UploadValidationError("corrupt_file", "DOCX ZIP signature is missing")
        _validate_docx(content)
        return UploadDescriptor(extension, "docx")

    if detected in {"pdf", "zip"}:
        raise UploadValidationError(
            "extension_content_mismatch",
            f"image extension does not match detected {detected} content",
        )
    image_format = _validate_image(extension, content)
    return UploadDescriptor(extension, f"image/{image_format.lower()}")
