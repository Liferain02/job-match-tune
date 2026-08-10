from __future__ import annotations

import threading
import time

from jobmatch_tune.api.server import (
    BatchMatchItem,
    BatchMatchRequest,
    BatchParseRequest,
    MatchRequest,
    ModelService,
)
from jobmatch_tune.inference.predict import build_prompt


def test_build_prompt_for_match() -> None:
    messages = build_prompt(
        "match",
        "岗位名称：AI应用开发工程师",
        resume_text="目标岗位：AI应用开发工程师",
        rule_result='{"匹配分数":82,"匹配等级":"较匹配"}',
    )
    assert messages[0]["role"] == "system"
    assert "规则评分结果" in messages[1]["content"]
    assert "匹配分数" in messages[1]["content"]


def test_batch_parse_endpoint(monkeypatch) -> None:
    service = ModelService()

    def fake_parse(request):
        return {
            "ok": True,
            "data": {"任务": request.task, "文本长度": len(request.text)},
            "latency_seconds": 0.01,
        }

    monkeypatch.setattr(service, "parse", fake_parse)
    monkeypatch.setattr(service, "load", lambda: None)

    payload = service.batch_parse(
        BatchParseRequest(
            task="jd_parse",
            texts=["岗位一", "岗位二"],
            max_new_tokens=256,
        )
    )
    assert payload["ok"] is True
    assert payload["success_count"] == 2
    assert payload["items"][0]["data"]["任务"] == "jd_parse"
    assert payload["execution"]["mode"] == "sequential"


def test_batch_match_endpoint(monkeypatch) -> None:
    service = ModelService()

    def fake_match(request):
        assert isinstance(request, MatchRequest)
        return {
            "ok": True,
            "jd_parse": {"岗位方向": "后端开发"},
            "resume_parse": {"目标岗位": "后端开发"},
            "rule_result": {"匹配分数": 88, "匹配等级": "较匹配"},
            "analysis": {"匹配结论": "基本匹配"},
            "latency_seconds": 0.02,
        }

    monkeypatch.setattr(service, "match", fake_match)
    monkeypatch.setattr(service, "load", lambda: None)

    payload = service.batch_match(
        BatchMatchRequest(
            items=[
                BatchMatchItem(jd_text="JD-A", resume_text="Resume-A"),
                BatchMatchItem(jd_text="JD-B", resume_text="Resume-B"),
            ],
            max_new_tokens=256,
        )
    )
    assert payload["ok"] is True
    assert payload["success_count"] == 2
    assert payload["items"][1]["analysis"]["匹配结论"] == "基本匹配"


def test_vllm_match_parses_jd_and_resume_in_parallel(monkeypatch) -> None:
    service = ModelService()
    service.backend = "vllm"
    service.parallel_match_parse = True
    service.vllm_max_concurrency = 4
    monkeypatch.setattr(service, "load", lambda: None)

    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def fake_parse(request):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with active_lock:
            active -= 1
        data = (
            {"岗位方向": "后端开发", "必备技能": ["Python"]}
            if request.task == "jd_parse"
            else {"目标岗位": "后端开发", "核心技能": ["Python"]}
        )
        return {"ok": True, "data": data, "latency_seconds": 0.03}

    monkeypatch.setattr(service, "parse", fake_parse)
    monkeypatch.setattr(
        service,
        "_match_with_vllm",
        lambda request, rule_result: {
            "ok": True,
            "data": {"匹配结论": "基本匹配"},
            "raw_output": "{}",
        },
    )

    payload = service.match(
        MatchRequest(jd_text="JD", resume_text="Resume", max_new_tokens=256)
    )
    assert max_active == 2
    assert payload["execution"] == {"backend": "vllm", "parse_mode": "parallel"}
    assert payload["timings"]["parse_wall_seconds"] >= 0.0
    assert payload["timings"]["total_seconds"] == payload["latency_seconds"]


def test_transformers_match_keeps_shared_model_generation_sequential(monkeypatch) -> None:
    service = ModelService()
    service.backend = "transformers"
    monkeypatch.setattr(service, "load", lambda: None)
    called_tasks: list[str] = []

    def fake_parse(request):
        called_tasks.append(request.task)
        data = (
            {"岗位方向": "后端开发", "必备技能": []}
            if request.task == "jd_parse"
            else {"目标岗位": "后端开发", "核心技能": []}
        )
        return {"ok": True, "data": data, "latency_seconds": 0.01}

    monkeypatch.setattr(service, "parse", fake_parse)
    monkeypatch.setattr(
        service,
        "_match_with_transformers",
        lambda request, rule_result: {
            "ok": True,
            "data": {"匹配结论": "基本匹配"},
            "raw_output": "{}",
        },
    )

    payload = service.match(
        MatchRequest(jd_text="JD", resume_text="Resume", max_new_tokens=256)
    )
    assert called_tasks == ["jd_parse", "resume_parse"]
    assert payload["execution"]["parse_mode"] == "sequential"


def test_vllm_batch_parse_uses_bounded_parallel_execution(monkeypatch) -> None:
    service = ModelService()
    service.backend = "vllm"
    service.vllm_max_concurrency = 2
    monkeypatch.setattr(service, "load", lambda: None)

    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def fake_parse(request):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with active_lock:
            active -= 1
        return {"ok": True, "data": {"text": request.text}, "latency_seconds": 0.02}

    monkeypatch.setattr(service, "parse", fake_parse)
    payload = service.batch_parse(
        BatchParseRequest(task="jd_parse", texts=["A", "B", "C"], max_new_tokens=64)
    )
    assert max_active == 2
    assert [item["data"]["text"] for item in payload["items"]] == ["A", "B", "C"]
    assert payload["execution"] == {"backend": "vllm", "mode": "parallel"}
