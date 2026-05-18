from __future__ import annotations

import json
import os
import time
from threading import Lock
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from jobmatch_tune.inference.predict import build_prompt, load_model
from jobmatch_tune.inference.postprocess_json import parse_json_output
from jobmatch_tune.inference.structured_output import build_response_format
from jobmatch_tune.match.rule_engine import compute_match_rule_result


DEFAULT_MODEL_PATH = "models/Qwen3-14B"
DEFAULT_ADAPTER_PATH = "outputs/checkpoints/qwen3-14b-jobmatch-qlora"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8010/v1"
DEFAULT_VLLM_MODEL = "jobmatch-lora"


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
        self._tokenizer = None
        self._model = None
        self._client = None
        self._lock = Lock()

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

    def match(self, request: MatchRequest) -> dict[str, Any]:
        self.load()
        started = time.perf_counter()

        jd_result = self.parse(ParseRequest(task="jd_parse", text=request.jd_text, max_new_tokens=request.max_new_tokens))
        resume_result = self.parse(ParseRequest(task="resume_parse", text=request.resume_text, max_new_tokens=request.max_new_tokens))
        if not jd_result.get("ok") or not resume_result.get("ok"):
            raise ValueError("Failed to parse JD or resume before matching")

        rule_result = compute_match_rule_result(
            jd_result["data"],
            resume_result["data"],
            jd_text=request.jd_text,
            resume_text=request.resume_text,
        )

        if self.backend == "vllm":
            analysis_result = self._match_with_vllm(request, rule_result)
        else:
            analysis_result = self._match_with_transformers(request, rule_result)

        return {
            "ok": analysis_result.get("ok", False),
            "jd_parse": jd_result["data"],
            "resume_parse": resume_result["data"],
            "rule_result": rule_result,
            "analysis": analysis_result.get("data"),
            "analysis_raw_output": analysis_result.get("raw_output", ""),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }

    def batch_parse(self, request: BatchParseRequest) -> dict[str, Any]:
        self.load()
        started = time.perf_counter()
        items: list[dict[str, Any]] = []

        for index, text in enumerate(request.texts):
            try:
                result = self.parse(
                    ParseRequest(
                        task=request.task,
                        text=text,
                        max_new_tokens=request.max_new_tokens,
                    )
                )
                items.append({"index": index, **result})
            except ValueError as exc:
                items.append({"index": index, "ok": False, "error": str(exc)})

        success_count = sum(1 for item in items if item.get("ok"))
        return {
            "ok": success_count == len(items),
            "task": request.task,
            "total": len(items),
            "success_count": success_count,
            "items": items,
            "latency_seconds": round(time.perf_counter() - started, 3),
        }

    def batch_match(self, request: BatchMatchRequest) -> dict[str, Any]:
        self.load()
        started = time.perf_counter()
        items: list[dict[str, Any]] = []

        for index, item in enumerate(request.items):
            try:
                result = self.match(
                    MatchRequest(
                        jd_text=item.jd_text,
                        resume_text=item.resume_text,
                        max_new_tokens=request.max_new_tokens,
                    )
                )
                items.append({"index": index, **result})
            except ValueError as exc:
                items.append({"index": index, "ok": False, "error": str(exc)})

        success_count = sum(1 for item in items if item.get("ok"))
        return {
            "ok": success_count == len(items),
            "total": len(items),
            "success_count": success_count,
            "items": items,
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
        }


service = ModelService()
app = FastAPI(title="JobMatchTune API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("JOBMATCH_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def jsonable(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)
