from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any

from jobmatch_tune.utils.io import write_text


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def detect_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def find_sidecar_ocr_text(path: Path, ocr_dir: Path | None = None) -> str:
    candidates = []
    if ocr_dir is not None:
        candidates.extend(
            [
                ocr_dir / f"{path.name}.ocr.txt",
                ocr_dir / f"{path.stem}.ocr.txt",
            ]
        )
    candidates.extend(
        [
            path.with_suffix(path.suffix + ".ocr.txt"),
            path.with_suffix(".ocr.txt"),
        ]
    )
    seen = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return ""


def sidecar_output_path(path: Path, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        return path.with_suffix(path.suffix + ".ocr.txt")
    return out_dir / f"{path.name}.ocr.txt"


def available_ocr_backend() -> str:
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return "rapidocr"
    except Exception:
        pass
    try:
        import paddleocr  # noqa: F401

        return "paddleocr"
    except Exception:
        pass
    return ""


def _parse_rapidocr_result(result: Any) -> list[str]:
    lines = []
    for item in result or []:
        if len(item) >= 2 and item[1]:
            lines.append(str(item[1]).strip())
    return [line for line in lines if line]


def _parse_paddleocr_result(result: Any) -> list[str]:
    lines = []
    for block in result or []:
        if not block:
            continue
        for item in block:
            if len(item) >= 2 and item[1]:
                text = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                lines.append(str(text).strip())
    return [line for line in lines if line]


def _ocr_image_with_backend(image_input: Any, backend: str) -> list[str]:
    if backend == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()
        result, _ = engine(image_input)
        return _parse_rapidocr_result(result)
    if backend == "paddleocr":
        from paddleocr import PaddleOCR

        engine = PaddleOCR(use_angle_cls=True, lang="ch")
        if isinstance(image_input, bytes):
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                tmp.write(image_input)
                tmp.flush()
                result = engine.ocr(tmp.name, cls=True)
        else:
            result = engine.ocr(image_input, cls=True)
        return _parse_paddleocr_result(result)
    raise RuntimeError("No OCR backend available. Install rapidocr_onnxruntime or paddleocr.")


def ocr_image_file(path: Path) -> str:
    backend = available_ocr_backend()
    lines = _ocr_image_with_backend(str(path), backend)
    return "\n".join(lines).strip()


def _render_pdf_page_images(path: Path) -> list[bytes]:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyMuPDF is required for PDF OCR rendering") from exc

    document = fitz.open(str(path))
    images: list[bytes] = []
    for page in document:
        pix = page.get_pixmap(dpi=200, alpha=False)
        images.append(pix.tobytes("png"))
    return images


def ocr_pdf_file(path: Path) -> str:
    backend = available_ocr_backend()
    page_images = _render_pdf_page_images(path)
    page_texts: list[str] = []
    for image_bytes in page_images:
        page_text = "\n".join(_ocr_image_with_backend(image_bytes, backend)).strip()
        if page_text:
            page_texts.append(page_text)
    return "\n\n".join(page_texts).strip()


def generate_sidecar_ocr(path: Path, out_dir: Path | None = None) -> Path:
    source_type = detect_source_type(path)
    if source_type not in {"image", "pdf"}:
        raise ValueError(f"OCR sidecar only supports image/pdf inputs, got: {source_type}")

    if source_type == "image":
        text = ocr_image_file(path)
    else:
        text = ocr_pdf_file(path)
    output_path = sidecar_output_path(path, out_dir=out_dir)
    write_text(output_path, text)
    return output_path


def collect_ocr_targets(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and (path.suffix.lower() in IMAGE_EXTENSIONS or path.suffix.lower() == ".pdf")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image/PDF file or directory")
    parser.add_argument("--out-dir", default=None, help="Directory to write OCR sidecar text files")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else None
    targets = collect_ocr_targets(input_path)
    if not targets:
        print("no image/pdf targets found")
        return
    for path in targets:
        output_path = generate_sidecar_ocr(path, out_dir=out_dir)
        print(f"{path} -> {output_path}")


if __name__ == "__main__":
    main()
