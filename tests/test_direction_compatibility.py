from __future__ import annotations

import pytest

from jobmatch_tune.match.direction_compatibility import (
    DirectionRelation,
    evaluate_direction_compatibility,
)


@pytest.mark.parametrize(
    ("jd_direction", "resume_direction"),
    [
        ("后端开发", "后端开发工程师"),
        ("算法工程", "算法工程师"),
        ("AI应用开发", "AI应用开发工程师"),
        ("数据开发", "数据工程师"),
        ("测试开发", "测试开发工程师"),
        ("前端开发", "前端工程师"),
        ("AI Infra", "AI Infra工程师"),
        ("模型服务", "推理工程师"),
    ],
)
def test_exact_direction_cases(jd_direction: str, resume_direction: str) -> None:
    decision = evaluate_direction_compatibility(jd_direction, resume_direction)
    assert decision.relation is DirectionRelation.EXACT
    assert decision.matches is True


@pytest.mark.parametrize(
    ("jd_direction", "resume_direction", "jd_context", "resume_context", "skills"),
    [
        (
            "算法工程",
            "AI后端",
            "建设模型 serving 平台 API，使用 PyTorch、FastAPI 和 Docker",
            "负责模型部署与在线推理 API，掌握 PyTorch、FastAPI 和 Docker",
            ["PyTorch", "FastAPI", "Docker"],
        ),
        (
            "模型服务",
            "后端开发",
            "建设在线推理服务和容器化平台",
            "开发模型服务平台，负责 Docker 部署",
            ["Python", "PyTorch", "Docker"],
        ),
        (
            "AI Infra",
            "后端开发",
            "维护 Kubernetes 推理服务平台",
            "负责模型服务和 Kubernetes 后端",
            ["Python", "PyTorch", "Kubernetes"],
        ),
        (
            "算法工程",
            "AI应用开发",
            "负责 RAG 与 Agent 应用落地",
            "开发 RAG 智能体工作流",
            ["Python", "RAG", "Agent"],
        ),
        (
            "数据开发",
            "后端开发",
            "建设 Flink 数据平台与数据服务",
            "负责数据平台后端和 ETL 管道",
            ["Java", "Flink", "Kafka"],
        ),
        (
            "算法平台",
            "AI Infra",
            "建设模型服务平台和 Kubernetes 推理服务",
            "维护模型部署平台与 Kubernetes 集群",
            ["Python", "PyTorch", "Kubernetes"],
        ),
    ],
)
def test_compatible_direction_requires_role_and_skill_evidence(
    jd_direction: str,
    resume_direction: str,
    jd_context: str,
    resume_context: str,
    skills: list[str],
) -> None:
    decision = evaluate_direction_compatibility(
        jd_direction,
        resume_direction,
        jd_context=jd_context,
        resume_context=resume_context,
        shared_skills=skills,
    )
    assert decision.relation is DirectionRelation.COMPATIBLE
    assert decision.matches is True


@pytest.mark.parametrize(
    ("jd_direction", "resume_direction", "jd_context", "resume_context", "skills"),
    [
        ("算法工程", "后端开发", "训练 CV 模型并优化 PyTorch", "Java 微服务与 MySQL", []),
        ("算法工程", "后端开发", "建设模型 serving 平台", "普通订单服务", ["Python"]),
        ("测试开发", "后端开发", "自动化测试平台", "Java 后端服务", ["Java", "MySQL"]),
        ("前端开发", "后端开发", "Vue 页面", "FastAPI 服务", ["Python", "FastAPI"]),
        ("数据开发", "后端开发", "离线数仓", "订单服务", ["Java", "MySQL"]),
        ("算法工程", "AI应用开发", "CV 模型训练", "RAG 应用", ["Python", "PyTorch"]),
        ("AI Infra", "后端开发", "GPU 集群", "普通 CRUD", ["Python", "Linux"]),
        ("产品经理", "后端开发", "需求管理", "服务开发", ["SQL", "Git"]),
        ("安全工程", "后端开发", "代码审计", "微服务", ["Python", "Linux"]),
        ("客户端开发", "后端开发", "Android 客户端", "Java 服务", ["Java", "Git"]),
    ],
)
def test_cross_role_mismatch_without_sufficient_evidence(
    jd_direction: str,
    resume_direction: str,
    jd_context: str,
    resume_context: str,
    skills: list[str],
) -> None:
    decision = evaluate_direction_compatibility(
        jd_direction,
        resume_direction,
        jd_context=jd_context,
        resume_context=resume_context,
        shared_skills=skills,
    )
    assert decision.relation is DirectionRelation.MISMATCH
    assert decision.matches is False
