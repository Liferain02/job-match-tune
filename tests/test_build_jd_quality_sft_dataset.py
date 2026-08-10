from __future__ import annotations

import json

from jobmatch_tune.dataset.build_jd_quality_sft_dataset import (
    _with_quality_score_meta,
    _sanitize_normalized_row,
    build_quality_profile,
    build_quality_rows,
    build_quality_weak_rows,
)
from jobmatch_tune.dataset.build_sft_dataset import build_jd_parse_sample


def _strict_row(row_id: str) -> dict:
    return {
        "id": row_id,
        "source": "careers.tencent.com",
        "language": "zh-cn",
        "sft_ready": True,
        "job_title": "后端开发工程师",
        "clean_text": "岗位职责\n负责后端系统开发\n任职要求\n本科及以上，3年以上经验，熟悉Java和Python。" * 6,
        "sections": {"responsibilities": "负责后端系统开发", "requirements": "本科及以上，3年以上经验"},
        "labels": {"岗位方向": "后端开发", "学历要求": "本科", "经验要求": "3年以上", "必备技能": ["Java"]},
    }


def _candidate_row(row_id: str) -> dict:
    return {
        "id": row_id,
        "source": "github_workaggregation_test",
        "job_title": "Python后端开发工程师",
        "company": "测试公司",
        "location": "北京",
        "raw_text": "岗位职责\n负责Python后端系统开发和接口设计。\n任职要求\n本科及以上，3年以上经验，熟悉Python、SQL、Linux。" * 4,
        "meta": {"training_eligible": True, "intended_usage": "weak_supervision_only"},
    }


def _weak_candidate_row(row_id: str) -> dict:
    return {
        "id": row_id,
        "source": "hf_job_educational_train_2026_05_17",
        "job_title": "Java后端开发工程师",
        "company": "测试公司",
        "location": "北京",
        "raw_text": (
            "岗位名称：Java后端开发工程师\n"
            "任务类型：从岗位中提取学历\n"
            "岗位描述：\n"
            "岗位职责：负责Java后端服务、接口平台、数据链路和稳定性建设，参与需求评审、系统设计、上线保障，"
            "推动核心链路容量评估、监控告警、故障复盘、性能优化和跨团队技术方案落地。\n"
            "岗位要求：本科及以上学历，熟悉Java、MySQL、Redis、Linux，理解分布式系统和服务治理，"
            "具备接口设计、数据库建模、缓存治理、线上问题定位和高并发服务调优能力。"
        ),
        "meta": {
            "language": "zh",
            "sft_ready": False,
            "training_eligible": True,
            "intended_usage": "weak_supervision_only",
        },
    }


def test_build_quality_rows_prioritizes_strict_then_enhanced() -> None:
    schema = {
        "job_directions": ["后端开发", "算法工程"],
        "skill_alias": {"Python": ["python"], "SQL": ["sql"], "Linux": ["linux"], "Java": ["java"]},
    }
    rows, stats = build_quality_rows(
        strict_rows=[_strict_row("s1"), _strict_row("s2")],
        candidate_rows=[_candidate_row("c1"), _candidate_row("c2")],
        schema=schema,
        target_total=3,
        seed=42,
    )
    assert len(rows) == 3
    assert stats["strict"] == 2
    assert stats["strict_plus"] >= 1
    assert {row["meta"]["quality_tier"] for row in rows} >= {"strict", "strict_plus"}


def test_build_quality_rows_accepts_single_pass_strict_rows() -> None:
    schema = {
        "job_directions": ["后端开发", "算法工程"],
        "skill_alias": {"Python": ["python"], "SQL": ["sql"], "Linux": ["linux"], "Java": ["java"]},
    }

    rows, stats = build_quality_rows(
        strict_rows=iter([_strict_row("s1"), _strict_row("s2")]),
        candidate_rows=[_candidate_row("c1")],
        schema=schema,
        target_total=2,
        seed=42,
    )

    assert len(rows) == 2
    assert stats["strict"] == 2


def test_build_quality_rows_excludes_holdout_ids() -> None:
    schema = {
        "job_directions": ["后端开发", "算法工程"],
        "skill_alias": {"Python": ["python"], "SQL": ["sql"], "Linux": ["linux"], "Java": ["java"]},
    }
    rows, _ = build_quality_rows(
        strict_rows=[_strict_row("holdout"), _strict_row("safe")],
        candidate_rows=[_candidate_row("candidate")],
        schema=schema,
        target_total=2,
        seed=42,
        excluded_ids={"holdout"},
        max_weak_external_ratio=1.0,
    )

    assert {row["id"] for row in rows} == {"safe", "candidate"}


def test_build_quality_weak_rows_sanitizes_degree_only_experience_and_repairs_direction() -> None:
    schema = {
        "job_directions": ["后端开发", "前端开发"],
        "skill_alias": {
            "Java": ["java"],
            "MySQL": ["mysql"],
            "Redis": ["redis"],
            "Linux": ["linux"],
        },
    }
    rows = build_quality_weak_rows([_weak_candidate_row("w1")], schema)
    assert len(rows) == 1
    assert rows[0]["labels"]["岗位方向"] == "后端开发"
    assert rows[0]["labels"]["经验要求"] == ""

    sample = build_jd_parse_sample(rows[0])
    assistant = json.loads(sample["messages"][-1]["content"])
    assert sample["meta"]["quality_tier"] == "quality_weak"
    assert assistant["岗位方向"] == "后端开发"
    assert assistant["经验要求"] == ""


def test_sanitize_normalized_row_keeps_network_security_in_security_direction() -> None:
    row = _strict_row("security")
    row["source"] = "hf_job_educational_train_2026_05_17"
    row["job_title"] = "网络与信息安全研究岗"
    row["labels"]["岗位方向"] = "网络与基础设施"

    sanitized = _sanitize_normalized_row(row)

    assert sanitized["labels"]["岗位方向"] == "安全工程"


def test_sanitize_normalized_row_repairs_responsibility_requirement_boundary() -> None:
    row = _strict_row("s1")
    row["sections"] = {
        "responsibilities": "负责服务开发。任职要求：本科及以上，熟悉 Java。",
        "requirements": "",
    }

    sanitized = _sanitize_normalized_row(row)

    assert sanitized["sections"]["responsibilities"] == "负责服务开发。"
    assert "任职要求" in sanitized["sections"]["requirements"]


def test_build_quality_profile_reports_tiers_and_empty_rates() -> None:
    row = _strict_row("s1")
    row["meta"] = {"quality_tier": "strict", "quality_reason": "high_trust"}
    row = _with_quality_score_meta(row)
    profile = build_quality_profile([row], {"strict": 1})

    assert profile["total"] == 1
    assert profile["tier_counts"] == {"strict": 1}
    assert profile["reason_counts"] == {"high_trust": 1}
    assert profile["quality_score_avg"] > 0
    assert profile["risk_score_counts"]


def test_quality_score_meta_tracks_risk_reasons() -> None:
    row = _strict_row("s1")
    row["meta"] = {"quality_tier": "strict", "quality_reason": "high_trust"}
    row["sections"]["responsibilities"] = "负责后端系统开发。" * 260

    scored = _with_quality_score_meta(row)

    assert scored["meta"]["quality_risk_score"] >= 1
    assert "oversized_single_responsibility" in scored["meta"]["quality_risk_reasons"]
    assert scored["meta"]["quality_score"] < 100


def test_build_quality_rows_caps_weak_external_family() -> None:
    schema = {
        "job_directions": ["后端开发", "算法工程"],
        "skill_alias": {"Python": ["python"], "SQL": ["sql"], "Linux": ["linux"], "Java": ["java"]},
    }
    weak_rows = [_weak_candidate_row(f"w{index}") for index in range(4)]
    rows, _ = build_quality_rows(
        strict_rows=[_strict_row("s1"), _strict_row("s2")],
        candidate_rows=weak_rows,
        schema=schema,
        target_total=4,
        seed=42,
        max_weak_external_ratio=0.5,
    )
    weak_count = sum(
        (row.get("meta") or {}).get("intended_usage") == "weak_supervision_only"
        for row in rows
    )
    assert len(rows) == 4
    assert weak_count <= 2
