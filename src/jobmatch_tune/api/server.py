from __future__ import annotations

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


DEFAULT_MODEL_PATH = "models/Qwen3-14B"
DEFAULT_ADAPTER_PATH = "outputs/checkpoints/qwen3-14b-jobmatch-qlora"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8010/v1"
DEFAULT_VLLM_MODEL = "jobmatch-lora"


class ParseRequest(BaseModel):
    task: Literal["jd_parse", "resume_parse"] = "jd_parse"
    text: str = Field(min_length=1, max_length=20000)
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
        assert self._tokenizer is not None
        assert self._model is not None

        messages = build_prompt(request.task, request.text)
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
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_output = self._tokenizer.decode(generated, skip_special_tokens=True)
        result = parse_json_output(raw_output, context_text=request.text)
        result["raw_output"] = raw_output
        return result

    def _parse_with_vllm(self, request: ParseRequest) -> dict[str, Any]:
        assert self._client is not None
        messages = build_prompt(request.task, request.text)
        completion = self._client.chat.completions.create(
            model=self.vllm_model,
            messages=messages,
            temperature=0,
            max_tokens=request.max_new_tokens,
            response_format=build_response_format(request.task),
        )
        raw_output = completion.choices[0].message.content or ""
        result = parse_json_output(raw_output, context_text=request.text)
        result["raw_output"] = raw_output
        return result

    def parse(self, request: ParseRequest) -> dict[str, Any]:
        self.load()

        started = time.perf_counter()
        if self.backend == "vllm":
            result = self._parse_with_vllm(request)
        else:
            result = self._parse_with_transformers(request)
        result["latency_seconds"] = round(time.perf_counter() - started, 3)
        return result

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
