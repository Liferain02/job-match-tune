from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, Literal

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from jobmatch_tune.inference.predict import build_prompt, load_model
from jobmatch_tune.inference.postprocess_json import parse_json_output
from jobmatch_tune.inference.structured_output import build_response_format
from jobmatch_tune.match.rule_engine import compute_match_rule_result
from jobmatch_tune.resume.ingest import ingest_resume
from jobmatch_tune.resume.normalize import normalize_ingest_row


DEFAULT_MODEL_PATH = "models/Qwen3-14B"
DEFAULT_ADAPTER_PATH = "outputs/checkpoints/qwen3-14b-jobmatch-dft-20260601"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8010/v1"
DEFAULT_VLLM_MODEL = "jobmatch-lora"
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def read_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class ParseRequest(BaseModel):
    task: Literal["jd_parse", "resume_parse"] = "jd_parse"
    text: str = Field(min_length=1, max_length=20000)
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)


class MatchRequest(BaseModel):
    jd_text: str = Field(min_length=1, max_length=20000)
    resume_text: str = Field(min_length=1, max_length=20000)
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)


class BatchParseRequest(BaseModel):
    task: Literal["jd_parse", "resume_parse"] = "jd_parse"
    texts: list[str] = Field(min_length=1, max_length=64)
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)


class BatchMatchItem(BaseModel):
    jd_text: str = Field(min_length=1, max_length=20000)
    resume_text: str = Field(min_length=1, max_length=20000)


class BatchMatchRequest(BaseModel):
    items: list[BatchMatchItem] = Field(min_length=1, max_length=32)
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)


class ModelService:
    def __init__(self) -> None:
        self.backend = os.getenv("JOBMATCH_INFERENCE_BACKEND", "transformers")
        self.model_path = os.getenv("JOBMATCH_MODEL_PATH", DEFAULT_MODEL_PATH)
        self.adapter_path = os.getenv("JOBMATCH_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
        self.load_4bit = os.getenv("JOBMATCH_LOAD_4BIT", "1") not in {"0", "false", "False"}
        self.vllm_base_url = os.getenv("JOBMATCH_VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL)
        self.vllm_api_key = os.getenv("JOBMATCH_VLLM_API_KEY", "-")
        self.vllm_model = os.getenv("JOBMATCH_VLLM_MODEL", DEFAULT_VLLM_MODEL)
        self.vllm_max_concurrency = read_positive_int_env("JOBMATCH_VLLM_MAX_CONCURRENCY", 4)
        self.parallel_match_parse = os.getenv("JOBMATCH_PARALLEL_MATCH_PARSE", "1") not in {
            "0",
            "false",
            "False",
        }
        self._tokenizer = None
        self._model = None
        self._client = None
        self._lock = Lock()
        self._vllm_slots = BoundedSemaphore(self.vllm_max_concurrency)

    @property
    def match_parse_mode(self) -> str:
        if self.backend == "vllm" and self.parallel_match_parse and self.vllm_max_concurrency >= 2:
            return "parallel"
        return "sequential"

    @property
    def batch_execution_mode(self) -> str:
        if self.backend == "vllm" and self.vllm_max_concurrency >= 2:
            return "parallel"
        return "sequential"

    @property
    def loaded(self) -> bool:
        if self.backend == "vllm":
            return self._client is not None
        return self._tokenizer is not None and self._model is not None

    def _load_transformers(self) -> None:
        self._tokenizer, self._model = load_model(
            self.model_path,
            self.adapter_path,
            self.load_4bit,
        )

    def _load_vllm(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.vllm_base_url, api_key=self.vllm_api_key)
        self._client.models.list()

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            if self.backend == "vllm":
                self._load_vllm()
            else:
                self._load_transformers()

    def _parse_with_transformers(self, request: ParseRequest) -> dict[str, Any]:
        messages = build_prompt(request.task, request.text)
        raw_output = self._complete_with_transformers(messages, request.max_new_tokens)
        result = parse_json_output(raw_output, context_text=request.text)
        result["raw_output"] = raw_output
        return result

    def _complete_with_transformers(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        assert self._tokenizer is not None
        assert self._model is not None
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._lock, torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _parse_with_vllm(self, request: ParseRequest) -> dict[str, Any]:
        assert self._client is not None
        messages = build_prompt(request.task, request.text)
        raw_output = self._complete_with_vllm(messages, request.max_new_tokens, request.task)
        result = parse_json_output(raw_output, context_text=request.text)
        result["raw_output"] = raw_output
        return result

    def _complete_with_vllm(self, messages: list[dict[str, str]], max_new_tokens: int, task: str) -> str:
        assert self._client is not None
        with self._vllm_slots:
            completion = self._client.chat.completions.create(
                model=self.vllm_model,
                messages=messages,
                temperature=0,
                max_tokens=max_new_tokens,
                response_format=build_response_format(task),
            )
        return completion.choices[0].message.content or ""

    def parse(self, request: ParseRequest) -> dict[str, Any]:
        self.load()

        started = time.perf_counter()
        if self.backend == "vllm":
            result = self._parse_with_vllm(request)
        else:
            result = self._parse_with_transformers(request)
        result["latency_seconds"] = round(time.perf_counter() - started, 3)
        return result

    def _match_with_transformers(self, request: MatchRequest, rule_result: dict[str, Any]) -> dict[str, Any]:
        messages = build_prompt(
            "match",
            request.jd_text,
            resume_text=request.resume_text,
            rule_result=jsonable(rule_result),
        )
        raw_output = self._complete_with_transformers(messages, request.max_new_tokens)
        result = parse_json_output(raw_output)
        result["raw_output"] = raw_output
        return result

    def _match_with_vllm(self, request: MatchRequest, rule_result: dict[str, Any]) -> dict[str, Any]:
        messages = build_prompt(
            "match",
            request.jd_text,
            resume_text=request.resume_text,
            rule_result=jsonable(rule_result),
        )
        raw_output = self._complete_with_vllm(messages, request.max_new_tokens, "match")
        result = parse_json_output(raw_output)
        result["raw_output"] = raw_output
        return result

    def _parse_match_inputs(
        self,
        request: MatchRequest,
    ) -> tuple[dict[str, Any], dict[str, Any], float]:
        jd_request = ParseRequest(
            task="jd_parse",
            text=request.jd_text,
            max_new_tokens=request.max_new_tokens,
        )
        resume_request = ParseRequest(
            task="resume_parse",
            text=request.resume_text,
            max_new_tokens=request.max_new_tokens,
        )
        started = time.perf_counter()
        if self.match_parse_mode == "parallel":
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="jobmatch-parse") as executor:
                jd_future = executor.submit(self.parse, jd_request)
                resume_future = executor.submit(self.parse, resume_request)
                jd_result = jd_future.result()
                resume_result = resume_future.result()
        else:
            jd_result = self.parse(jd_request)
            resume_result = self.parse(resume_request)
        return jd_result, resume_result, time.perf_counter() - started

    def match(self, request: MatchRequest) -> dict[str, Any]:
        self.load()
        started = time.perf_counter()

        jd_result, resume_result, parse_wall_seconds = self._parse_match_inputs(request)
        if not jd_result.get("ok") or not resume_result.get("ok"):
            raise ValueError("Failed to parse JD or resume before matching")

        rule_started = time.perf_counter()
        rule_result = compute_match_rule_result(
            jd_result["data"],
            resume_result["data"],
            jd_text=request.jd_text,
            resume_text=request.resume_text,
        )
        rule_seconds = time.perf_counter() - rule_started

        analysis_started = time.perf_counter()
        if self.backend == "vllm":
            analysis_result = self._match_with_vllm(request, rule_result)
        else:
            analysis_result = self._match_with_transformers(request, rule_result)
        analysis_seconds = time.perf_counter() - analysis_started
        total_seconds = time.perf_counter() - started

        return {
            "ok": analysis_result.get("ok", False),
            "jd_parse": jd_result["data"],
            "resume_parse": resume_result["data"],
            "rule_result": rule_result,
            "analysis": analysis_result.get("data"),
            "analysis_raw_output": analysis_result.get("raw_output", ""),
            "execution": {
                "backend": self.backend,
                "parse_mode": self.match_parse_mode,
            },
            "timings": {
                "jd_parse_seconds": jd_result.get("latency_seconds", 0.0),
                "resume_parse_seconds": resume_result.get("latency_seconds", 0.0),
                "parse_wall_seconds": round(parse_wall_seconds, 3),
                "rule_seconds": round(rule_seconds, 3),
                "analysis_seconds": round(analysis_seconds, 3),
                "total_seconds": round(total_seconds, 3),
            },
            "latency_seconds": round(total_seconds, 3),
        }

    def batch_parse(self, request: BatchParseRequest) -> dict[str, Any]:
        self.load()
        started = time.perf_counter()
        items: list[dict[str, Any]] = []

        def parse_one(index_and_text: tuple[int, str]) -> dict[str, Any]:
            index, text = index_and_text
            try:
                result = self.parse(
                    ParseRequest(
                        task=request.task,
                        text=text,
                        max_new_tokens=request.max_new_tokens,
                    )
                )
                return {"index": index, **result}
            except ValueError as exc:
                return {"index": index, "ok": False, "error": str(exc)}

        indexed_texts = list(enumerate(request.texts))
        if self.batch_execution_mode == "parallel" and len(indexed_texts) > 1:
            worker_count = min(self.vllm_max_concurrency, len(indexed_texts))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="jobmatch-batch-parse",
            ) as executor:
                items = list(executor.map(parse_one, indexed_texts))
        else:
            items = [parse_one(item) for item in indexed_texts]

        success_count = sum(1 for item in items if item.get("ok"))
        return {
            "ok": success_count == len(items),
            "task": request.task,
            "total": len(items),
            "success_count": success_count,
            "items": items,
            "execution": {"backend": self.backend, "mode": self.batch_execution_mode},
            "latency_seconds": round(time.perf_counter() - started, 3),
        }

    def batch_match(self, request: BatchMatchRequest) -> dict[str, Any]:
        self.load()
        started = time.perf_counter()
        items: list[dict[str, Any]] = []

        def match_one(index_and_item: tuple[int, BatchMatchItem]) -> dict[str, Any]:
            index, item = index_and_item
            try:
                result = self.match(
                    MatchRequest(
                        jd_text=item.jd_text,
                        resume_text=item.resume_text,
                        max_new_tokens=request.max_new_tokens,
                    )
                )
                return {"index": index, **result}
            except ValueError as exc:
                return {"index": index, "ok": False, "error": str(exc)}

        indexed_items = list(enumerate(request.items))
        if self.batch_execution_mode == "parallel" and len(indexed_items) > 1:
            worker_count = min(self.vllm_max_concurrency, len(indexed_items))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="jobmatch-batch-match",
            ) as executor:
                items = list(executor.map(match_one, indexed_items))
        else:
            items = [match_one(item) for item in indexed_items]

        success_count = sum(1 for item in items if item.get("ok"))
        return {
            "ok": success_count == len(items),
            "total": len(items),
            "success_count": success_count,
            "items": items,
            "execution": {"backend": self.backend, "mode": self.batch_execution_mode},
            "latency_seconds": round(time.perf_counter() - started, 3),
        }

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "load_4bit": self.load_4bit,
            "loaded": self.loaded,
            "cuda_available": torch.cuda.is_available(),
            "vllm_base_url": self.vllm_base_url if self.backend == "vllm" else "",
            "vllm_model": self.vllm_model if self.backend == "vllm" else "",
            "vllm_max_concurrency": self.vllm_max_concurrency if self.backend == "vllm" else 0,
            "match_parse_mode": self.match_parse_mode,
            "batch_execution_mode": self.batch_execution_mode,
        }


service = ModelService()
app = FastAPI(title="JobMatchTune API", version="0.1.0")
configured_cors_origins = os.getenv("JOBMATCH_CORS_ORIGINS", "").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        [origin.strip() for origin in configured_cors_origins.split(",") if origin.strip()]
        if configured_cors_origins
        else []
    ),
    allow_origin_regex=(
        None if configured_cors_origins else r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_uploaded_resume_bytes(
    service: ModelService,
    *,
    file_name: str,
    content: bytes,
    ocr_text: str = "",
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    suffix = Path(file_name).suffix or ".txt"
    with tempfile.TemporaryDirectory(prefix="jobmatch_resume_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        resume_path = tmpdir_path / f"resume{suffix}"
        resume_path.write_bytes(content)
        if ocr_text.strip():
            (tmpdir_path / "resume.ocr.txt").write_text(ocr_text.strip(), encoding="utf-8")

        ingest_row = ingest_resume(resume_path, ocr_dir=tmpdir_path)
        if not ingest_row.get("parse_ok"):
            return {
                "ok": False,
                "file_name": file_name,
                "ingest": ingest_row,
                "needs_ocr": ingest_row.get("needs_ocr", False),
                "error": ingest_row.get("parse_error", "resume_ingest_failed"),
            }

        normalized = normalize_ingest_row(ingest_row)
        parse_result = service.parse(
            ParseRequest(
                task="resume_parse",
                text=normalized["normalized_text"],
                max_new_tokens=max_new_tokens,
            )
        )
        return {
            "ok": parse_result.get("ok", False),
            "file_name": file_name,
            "ingest": {
                "source_type": ingest_row.get("source_type", ""),
                "pdf_kind": ingest_row.get("pdf_kind", ""),
                "ocr_used": ingest_row.get("ocr_used", False),
                "ocr_source": ingest_row.get("ocr_source", ""),
                "extraction_method": ingest_row.get("extraction_method", ""),
                "page_count": ingest_row.get("page_count", 1),
                "text_char_count": ingest_row.get("text_char_count", 0),
                "needs_ocr": ingest_row.get("needs_ocr", False),
            },
            "normalized_text": normalized["normalized_text"],
            "sections": normalized["sections"],
            "data": parse_result.get("data"),
            "raw_output": parse_result.get("raw_output", ""),
            "latency_seconds": parse_result.get("latency_seconds", 0.0),
        }


def parse_uploaded_document_bytes(
    service: ModelService,
    *,
    task: Literal["jd_parse", "resume_parse"],
    file_name: str,
    content: bytes,
    ocr_text: str = "",
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    extracted = extract_uploaded_document_text(
        file_name=file_name,
        content=content,
        ocr_text=ocr_text,
    )
    if not extracted.get("ok"):
        return extracted
    parse_result = service.parse(
        ParseRequest(
            task=task,
            text=extracted["text"],
            max_new_tokens=max_new_tokens,
        )
    )
    return {
        "ok": parse_result.get("ok", False),
        "file_name": file_name,
        "task": task,
        "ingest": extracted["ingest"],
        "text": extracted["text"],
        "sections": extracted.get("sections", {}),
        "data": parse_result.get("data"),
        "raw_output": parse_result.get("raw_output", ""),
        "latency_seconds": parse_result.get("latency_seconds", 0.0),
    }


def extract_uploaded_document_text(
    *,
    file_name: str,
    content: bytes,
    ocr_text: str = "",
) -> dict[str, Any]:
    suffix = Path(file_name).suffix or ".txt"
    with tempfile.TemporaryDirectory(prefix="jobmatch_doc_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        document_path = tmpdir_path / f"document{suffix}"
        document_path.write_bytes(content)
        if ocr_text.strip():
            (tmpdir_path / "document.ocr.txt").write_text(ocr_text.strip(), encoding="utf-8")

        ingest_row = ingest_resume(document_path, ocr_dir=tmpdir_path)
        if not ingest_row.get("parse_ok"):
            return {
                "ok": False,
                "file_name": file_name,
                "ingest": ingest_row,
                "needs_ocr": ingest_row.get("needs_ocr", False),
                "error": ingest_row.get("parse_error", "document_ingest_failed"),
            }

        normalized = normalize_ingest_row(ingest_row)
        text = normalized["normalized_text"] if normalized["source_type"] == "image" else normalized["clean_text"]
        if not text.strip():
            return {
                "ok": False,
                "file_name": file_name,
                "ingest": ingest_row,
                "needs_ocr": ingest_row.get("needs_ocr", False),
                "error": "empty_text_after_extraction",
            }
        return {
            "ok": True,
            "file_name": file_name,
            "text": text,
            "sections": normalized["sections"],
            "ingest": {
                "source_type": ingest_row.get("source_type", ""),
                "pdf_kind": ingest_row.get("pdf_kind", ""),
                "ocr_used": ingest_row.get("ocr_used", False),
                "ocr_source": ingest_row.get("ocr_source", ""),
                "extraction_method": ingest_row.get("extraction_method", ""),
                "page_count": ingest_row.get("page_count", 1),
                "text_char_count": ingest_row.get("text_char_count", 0),
                "needs_ocr": ingest_row.get("needs_ocr", False),
            },
        }


def match_uploaded_inputs(
    service: ModelService,
    *,
    jd_text: str = "",
    resume_text: str = "",
    jd_file_name: str = "",
    jd_content: bytes | None = None,
    resume_file_name: str = "",
    resume_content: bytes | None = None,
    jd_ocr_text: str = "",
    resume_ocr_text: str = "",
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    jd_payload: dict[str, Any] = {"source": "text", "text": jd_text.strip()}
    resume_payload: dict[str, Any] = {"source": "text", "text": resume_text.strip()}
    if jd_content is not None:
        jd_payload = extract_uploaded_document_text(
            file_name=jd_file_name or "jd.txt",
            content=jd_content,
            ocr_text=jd_ocr_text,
        )
        if not jd_payload.get("ok"):
            return {"ok": False, "stage": "jd_ingest", **jd_payload}
    if resume_content is not None:
        resume_payload = extract_uploaded_document_text(
            file_name=resume_file_name or "resume.txt",
            content=resume_content,
            ocr_text=resume_ocr_text,
        )
        if not resume_payload.get("ok"):
            return {"ok": False, "stage": "resume_ingest", **resume_payload}

    resolved_jd_text = str(jd_payload.get("text") or "").strip()
    resolved_resume_text = str(resume_payload.get("text") or "").strip()
    if not resolved_jd_text:
        raise ValueError("JD text is empty")
    if not resolved_resume_text:
        raise ValueError("Resume text is empty")

    match_result = service.match(
        MatchRequest(
            jd_text=resolved_jd_text,
            resume_text=resolved_resume_text,
            max_new_tokens=max_new_tokens,
        )
    )
    return {
        **match_result,
        "inputs": {
            "jd": {
                "source": "file" if jd_content is not None else "text",
                "file_name": jd_file_name,
                "ingest": jd_payload.get("ingest", {}),
                "text_char_count": len(resolved_jd_text),
            },
            "resume": {
                "source": "file" if resume_content is not None else "text",
                "file_name": resume_file_name,
                "ingest": resume_payload.get("ingest", {}),
                "text_char_count": len(resolved_resume_text),
            },
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, **service.status()}


@app.post("/api/warmup")
def warmup() -> dict[str, Any]:
    started = time.perf_counter()
    service.load()
    return {
        "ok": True,
        "loaded": service.loaded,
        "latency_seconds": round(time.perf_counter() - started, 3),
        **service.status(),
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {"ok": True, **service.status()}


@app.post("/api/parse")
def parse(request: ParseRequest) -> dict[str, Any]:
    try:
        return service.parse(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/match")
def match(request: MatchRequest) -> dict[str, Any]:
    try:
        return service.match(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/batch_parse")
def batch_parse(request: BatchParseRequest) -> dict[str, Any]:
    try:
        return service.batch_parse(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/batch_match")
def batch_match(request: BatchMatchRequest) -> dict[str, Any]:
    try:
        return service.batch_match(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/resume_file_parse")
async def resume_file_parse(
    file: UploadFile = File(...),
    max_new_tokens: int = Form(default=1024),
    ocr_text: str = Form(default=""),
) -> dict[str, Any]:
    try:
        content = await read_upload_content(file)
        return await run_in_threadpool(
            parse_uploaded_resume_bytes,
            service,
            file_name=file.filename or "resume.txt",
            content=content,
            ocr_text=ocr_text,
            max_new_tokens=max_new_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jd_file_parse")
async def jd_file_parse(
    file: UploadFile = File(...),
    max_new_tokens: int = Form(default=1024),
    ocr_text: str = Form(default=""),
) -> dict[str, Any]:
    try:
        content = await read_upload_content(file)
        return await run_in_threadpool(
            parse_uploaded_document_bytes,
            service,
            task="jd_parse",
            file_name=file.filename or "jd.txt",
            content=content,
            ocr_text=ocr_text,
            max_new_tokens=max_new_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/match_files")
async def match_files(
    jd_text: str = Form(default=""),
    resume_text: str = Form(default=""),
    jd_file: UploadFile | None = File(default=None),
    resume_file: UploadFile | None = File(default=None),
    jd_ocr_text: str = Form(default=""),
    resume_ocr_text: str = Form(default=""),
    max_new_tokens: int = Form(default=1024),
) -> dict[str, Any]:
    try:
        jd_content = await read_upload_content(jd_file) if jd_file is not None else None
        resume_content = await read_upload_content(resume_file) if resume_file is not None else None
        return await run_in_threadpool(
            match_uploaded_inputs,
            service,
            jd_text=jd_text,
            resume_text=resume_text,
            jd_file_name=jd_file.filename if jd_file is not None else "",
            jd_content=jd_content,
            resume_file_name=resume_file.filename if resume_file is not None else "",
            resume_content=resume_content,
            jd_ocr_text=jd_ocr_text,
            resume_ocr_text=resume_ocr_text,
            max_new_tokens=max_new_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def jsonable(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


async def read_upload_content(file: UploadFile) -> bytes:
    max_bytes = int(os.getenv("JOBMATCH_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))
    if max_bytes <= 0:
        raise ValueError("JOBMATCH_MAX_UPLOAD_BYTES must be positive")
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"Uploaded file exceeds the {max_bytes}-byte limit")
    if not content:
        raise ValueError("Uploaded file is empty")
    return content
