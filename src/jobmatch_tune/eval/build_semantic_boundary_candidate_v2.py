from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _case(
    number: int,
    *,
    jd: str,
    resume: str,
    tags: list[str],
    draft_label: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"semantic_boundary_v2_{number:03d}",
        "task": "match",
        "source_type": "semantic_boundary_candidate_v2",
        "source_group": f"semantic_boundary_v2_{number:03d}",
        "jd_text": jd,
        "resume_text": resume,
        "label": draft_label,
        "meta": {
            "annotation_status": "needs_human_review",
            "annotation_provenance": "codex_draft_for_independent_human_review",
            "training_eligible": False,
            "intended_usage": "semantic_boundary_candidate_v2_review_only",
            "difficulty_tags": tags,
            "annotator_id": "",
            "reviewed_at": "",
            "rationale": "",
        },
    }


def build_candidates() -> list[dict[str, Any]]:
    """Return new review candidates; these draft labels are not human Gold."""

    return [
        _case(1, jd="质量平台工程师。任职要求：掌握 Python、Pytest、Selenium。", resume="目标测试开发；技能：Py thon、Py test、Selenium。", tags=["OCR词内断开", "技能同义词"], draft_label={"命中技能": ["Python", "Pytest", "Selenium"], "缺失技能": []}),
        _case(2, jd="云原生平台开发。任职要求：熟悉 Kubernetes、MySQL、C++。", resume="技能清单：Kubernet  es、My SOL、C + +。项目为容器平台。", tags=["OCR词内断开"], draft_label={"命中技能": ["Kubernetes", "MySQL", "C++"], "缺失技能": []}),
        _case(3, jd="前端工程师。任职要求：掌握 Vue、TypeScript、Vite。", resume="使用 Vue 3 与 Type Script 开发后台，熟悉 Vite。", tags=["OCR词内断开", "技能同义词"], draft_label={"命中技能": ["Vue", "TypeScript", "Vite"], "缺失技能": []}),
        _case(4, jd="后端开发。任职要求：熟悉 Node.js、PostgreSQL。", resume="项目使用 Node . js；日常维护 Postman collection。", tags=["OCR误报控制"], draft_label={"命中技能": ["Node.js"], "缺失技能": ["PostgreSQL"]}),
        _case(5, jd="测试工具开发。任职要求：掌握 Pytest。", resume="项目文档中只有 pytest-like helper 的概念描述，没有使用测试框架。", tags=["OCR误报控制"], draft_label={"命中技能": [], "缺失技能": ["Pytest"]}),
        _case(6, jd="智能客服研发。岗位职责：负责 Agent 流程开发。任职要求：熟悉 Python 和 FastAPI。", resume="AI 应用方向；掌握 Python、FastAPI，未提及智能体。", tags=["职责与要求边界"], draft_label={"命中技能": ["Python", "FastAPI"], "缺失技能": []}),
        _case(7, jd="数据服务工程师。岗位职责：维护 Kafka 数据链路。任职要求：掌握 Java、MySQL。", resume="后端方向；技能 Java、MySQL，没有 Kafka 经历。", tags=["职责与要求边界"], draft_label={"命中技能": ["Java", "MySQL"], "缺失技能": []}),
        _case(8, jd="推理服务开发。岗位职责：建设 Docker 平台。任职要求：熟悉 Docker、Kubernetes 和 Python。", resume="负责容器平台，使用 Docker、Kubernetes、Python。", tags=["职责要求重复"], draft_label={"命中技能": ["Docker", "Kubernetes", "Python"], "缺失技能": []}),
        _case(9, jd="搜索后端工程师。岗位职责：使用 Redis 完成缓存治理。加分项：有 Kafka 经验优先。任职要求：掌握 Go。", resume="Go 服务开发，熟悉 Redis，没有 Kafka 项目。", tags=["加分项非硬门槛", "上下文技能"], draft_label={"命中技能": ["Go"], "缺失技能": []}),
        _case(10, jd="机器学习平台研发。岗位职责：研究 PyTorch 训练任务。任职要求：具备 Python 工程能力。", resume="后端开发方向，使用 Python 建设任务系统。", tags=["职责与要求边界", "项目证据"], draft_label={"命中技能": ["Python"], "缺失技能": []}),
        _case(11, jd="算法平台工程师；建设模型 serving、推理 API 和容器化发布；要求 Python、PyTorch、FastAPI、Docker。", resume="目标 AI 后端；完成在线推理 API 与模型部署；技能 Python、PyTorch、FastAPI、Docker。", tags=["跨岗位方向", "模型服务"], draft_label={"岗位方向匹配": True, "命中技能": ["Python", "PyTorch", "FastAPI", "Docker"], "缺失技能": []}),
        _case(12, jd="视觉算法工程师；负责 CV 模型训练优化；要求 PyTorch、OpenCV。", resume="目标 Java 后端；项目为订单微服务；技能 Java、Spring Boot、MySQL。", tags=["跨岗位方向", "不兼容方向"], draft_label={"岗位方向匹配": False, "命中技能": [], "缺失技能": ["PyTorch", "OpenCV"]}),
        _case(13, jd="模型服务工程师；维护 Kubernetes 在线推理平台；要求 Python、PyTorch、Kubernetes。", resume="目标 AI Infra；负责模型部署平台和 GPU 服务；掌握 Python、PyTorch、Kubernetes。", tags=["跨岗位方向", "AI基础设施"], draft_label={"岗位方向匹配": True, "命中技能": ["Python", "PyTorch", "Kubernetes"], "缺失技能": []}),
        _case(14, jd="数据平台工程师；建设 Flink 数据管道和数据服务；要求 Java、Flink、Kafka。", resume="目标后端开发；负责数据平台 API 和 ETL；掌握 Java、Flink、Kafka。", tags=["跨岗位方向", "数据平台"], draft_label={"岗位方向匹配": True, "命中技能": ["Java", "Flink", "Kafka"], "缺失技能": []}),
        _case(15, jd="后端开发工程师；负责交易服务；要求 Java、Spring Boot、MySQL。", resume="目标测试开发；负责接口自动化；技能 Java、Pytest、Selenium。", tags=["跨岗位方向", "不兼容方向"], draft_label={"岗位方向匹配": False, "命中技能": ["Java"], "缺失技能": ["Spring Boot", "MySQL"]}),
        _case(16, jd="大模型应用工程师；负责 RAG 工作流落地；要求 Python、RAG、LangChain。", resume="目标算法工程；项目是企业知识助手和 RAG 应用；掌握 Python、RAG、LangChain。", tags=["跨岗位方向", "AI应用"], draft_label={"岗位方向匹配": True, "命中技能": ["Python", "RAG", "LangChain"], "缺失技能": []}),
        _case(17, jd="推理性能工程师；要求 CUDA、C++、Linux；负责算子优化。", resume="高性能计算方向；项目中用 C++、CUDA 优化推理算子，环境为 Linux。", tags=["项目证据", "可迁移技能"], draft_label={"命中技能": ["CUDA", "C++", "Linux"], "缺失技能": []}),
        _case(18, jd="平台后端工程师；任职要求：熟悉 FastAPI、Redis、Docker。", resume="技能只有 Python；项目描述为参与平台讨论，没有 FastAPI、Redis 或 Docker 实现证据。", tags=["项目证据", "硬技能缺失"], draft_label={"命中技能": [], "缺失技能": ["FastAPI", "Redis", "Docker"]}),
    ]


def write_candidates(path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"review file already exists and was not overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(json.dumps(row, ensure_ascii=False) for row in build_candidates()) + "\n"
    path.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Semantic Boundary Candidate V2")
    parser.add_argument("--out", default="data/private/semantic_boundary_candidate_v2.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    write_candidates(Path(args.out), overwrite=args.overwrite)
    print(json.dumps({"out": args.out, "rows": len(build_candidates()), "status": "needs_human_review"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
