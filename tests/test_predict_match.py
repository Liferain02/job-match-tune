from __future__ import annotations

from jobmatch_tune.api.server import BatchMatchItem, BatchMatchRequest, BatchParseRequest, MatchRequest, ModelService
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
