from __future__ import annotations

from typing import Any

from jobmatch_tune.dataset.curated_resume_training_data import CURATED_RESUME_TRAIN_ROWS


_RESUMES = {int(row["id"].rsplit("_", 1)[-1]): row for row in CURATED_RESUME_TRAIN_ROWS}


def _row(
    number: int,
    *,
    jd_text: str,
    level: str,
    direction_match: bool,
    education_match: bool,
    experience_match: bool,
    matched_skills: list[str],
    missing_skills: list[str],
    matched_projects: list[str],
    jd_direction: str,
    entity_split: str,
    rationale: str,
) -> dict[str, Any]:
    resume = _RESUMES[number]
    return {
        "id": f"match_curated_train_{number:03d}",
        "task": "match",
        "source_type": "curated_fictional_pair",
        "source_group": f"match_curated_train_{number:03d}",
        "jd_text": jd_text,
        "resume_text": resume["text"],
        "label": {
            "匹配等级": level,
            "岗位方向匹配": direction_match,
            "学历匹配": education_match,
            "经验匹配": experience_match,
            "命中技能": matched_skills,
            "缺失技能": missing_skills,
            "命中项目": matched_projects,
        },
        "meta": {
            "language": "zh",
            "provenance": "repository_curated_fictional_pair_v1",
            "annotation_status": "repository_curated_unverified",
            "annotation_provenance": "ai_assisted_individual_case_authoring",
            "generator": "ai_assisted_individual_case_authoring",
            "pair_type": "curated_fictional_pair",
            "training_eligible": True,
            "contains_real_person_data": False,
            "jd_direction": jd_direction,
            "resume_direction": resume["label"]["目标岗位"],
            "entity_split": entity_split,
            "annotation_rationale": rationale,
        },
    }


# Each pair is authored independently. There is no template expansion, label flip,
# or rule-generated negative sampling. The fixed split keeps resume/JD entities from
# crossing train, validation, and test sets. These rows are useful for pipeline
# development only; they are not evidence of quality on real observed applications.
CURATED_MATCH_TRAIN_ROWS = [
    _row(
        1,
        jd_text=(
            "岗位：Java 后端工程师\n职责：负责订单与会员服务的设计、开发和线上治理。\n"
            "要求：本科及以上，3 年以上后端经验；熟悉 Java、Spring Boot、MySQL、Redis、Kafka，"
            "有容器化部署和性能优化经验。"
        ),
        level="高匹配",
        direction_match=True,
        education_match=True,
        experience_match=True,
        matched_skills=["Java", "Spring Boot", "MySQL", "Redis", "Kafka", "Docker", "Kubernetes"],
        missing_skills=[],
        matched_projects=["订单查询链路重构", "消息幂等和失败补偿机制"],
        jd_direction="后端开发",
        entity_split="train",
        rationale="方向、学历、4 年经验和全部核心技术栈均有直接证据。",
    ),
    _row(
        2,
        jd_text=(
            "岗位：Go 服务端工程师\n职责：建设高并发检索接口和服务可观测体系。\n"
            "要求：本科及以上，3 年以上服务端经验；掌握 Go、gRPC、Redis、PostgreSQL、Prometheus。"
        ),
        level="较匹配",
        direction_match=True,
        education_match=True,
        experience_match=False,
        matched_skills=["Go", "gRPC", "Redis", "PostgreSQL", "Prometheus"],
        missing_skills=[],
        matched_projects=["并发流水线改造", "慢查询治理和指标告警"],
        jd_direction="后端开发",
        entity_split="train",
        rationale="技能和项目高度相关，但只有 2 年经验，未满足明确的 3 年门槛。",
    ),
    _row(
        3,
        jd_text=(
            "岗位：大模型应用开发工程师\n职责：开发企业知识库问答和模型服务接口。\n"
            "要求：本科及以上，2 年以上相关工作经验；熟悉 Python、FastAPI、RAG、向量数据库和 MySQL。"
        ),
        level="较匹配",
        direction_match=True,
        education_match=True,
        experience_match=False,
        matched_skills=["Python", "FastAPI", "RAG", "Milvus", "MySQL"],
        missing_skills=[],
        matched_projects=["技术文档问答助手", "引用溯源和无答案拒答"],
        jd_direction="AI应用开发",
        entity_split="train",
        rationale="实习和项目覆盖核心能力，但没有 2 年正式相关工作经验。",
    ),
    _row(
        4,
        jd_text=(
            "岗位：搜索算法工程师\n职责：优化语义召回、排序模型并建设离线评测。\n"
            "要求：计算机相关硕士；熟悉 Python、PyTorch、Transformers、信息检索和实验分析，工作经验不限。"
        ),
        level="高匹配",
        direction_match=True,
        education_match=True,
        experience_match=True,
        matched_skills=["Python", "PyTorch", "Transformers", "信息检索", "离线评估"],
        missing_skills=[],
        matched_projects=["中文语义匹配模型", "Recall@K 和 MRR 评测流程"],
        jd_direction="算法工程",
        entity_split="train",
        rationale="岗位接受应届生，硕士背景、检索项目和评测经验均直接对应。",
    ),
    _row(
        5,
        jd_text=(
            "岗位：AI Infra 工程师\n职责：负责训练任务编排、GPU 资源治理和推理发布平台。\n"
            "要求：本科及以上，3 年以上平台研发经验；熟悉 Python、Go、Kubernetes、Docker、Prometheus。"
        ),
        level="高匹配",
        direction_match=True,
        education_match=True,
        experience_match=True,
        matched_skills=["Python", "Go", "Kubernetes", "Docker", "Prometheus", "GPU"],
        missing_skills=[],
        matched_projects=["训练任务队列和失败重试", "GPU 利用率治理"],
        jd_direction="AI Infra",
        entity_split="train",
        rationale="方向、3 年经验、平台职责和基础设施技术栈完整对应。",
    ),
    _row(
        6,
        jd_text=(
            "岗位：实时数据开发工程师\n职责：建设实时指标链路和离线数仓。\n"
            "要求：本科及以上，3 年以上数据开发经验；熟悉 SQL、Flink、Kafka、Spark、Hive 和 Airflow。"
        ),
        level="较匹配",
        direction_match=True,
        education_match=True,
        experience_match=False,
        matched_skills=["SQL", "Flink", "Kafka", "Spark", "Hive", "Airflow"],
        missing_skills=[],
        matched_projects=["用户行为宽表重构", "实时链路质量校验和延迟告警"],
        jd_direction="数据开发",
        entity_split="train",
        rationale="技术和项目满足岗位，但 2 年经验低于明确的 3 年要求。",
    ),
    _row(
        7,
        jd_text=(
            "岗位：React 前端工程师\n职责：开发中后台复杂交互和可视化页面。\n"
            "要求：本科及以上，1 年以上正式前端工作经验；掌握 TypeScript、React、Vite、ECharts 和自动化测试。"
        ),
        level="较匹配",
        direction_match=True,
        education_match=True,
        experience_match=False,
        matched_skills=["TypeScript", "React", "Vite", "ECharts", "Playwright"],
        missing_skills=[],
        matched_projects=["可视化流程编辑器", "运营中台实习"],
        jd_direction="前端开发",
        entity_split="train",
        rationale="技能与项目相关，但候选人仍在读且只有实习，没有 1 年正式工作经验。",
    ),
    _row(
        8,
        jd_text=(
            "岗位：测试开发工程师\n职责：建设接口自动化、性能测试和持续集成质量门禁。\n"
            "要求：本科及以上，3 年以上测试开发经验；熟悉 Python、Pytest、Selenium、JMeter、Jenkins。"
        ),
        level="高匹配",
        direction_match=True,
        education_match=True,
        experience_match=True,
        matched_skills=["Python", "Pytest", "Selenium", "JMeter", "Jenkins"],
        missing_skills=[],
        matched_projects=["接口回归流水线", "故障注入和降级验证"],
        jd_direction="测试开发",
        entity_split="train",
        rationale="3 年经验、自动化测试栈和稳定性项目均直接对应。",
    ),
    _row(
        9,
        jd_text=(
            "岗位：SRE 工程师\n职责：负责发布治理、可观测性和生产故障响应。\n"
            "要求：本科及以上，3 年以上相关经验；熟悉 Linux、Python、Kubernetes、Prometheus、Grafana。"
        ),
        level="高匹配",
        direction_match=True,
        education_match=True,
        experience_match=True,
        matched_skills=["Linux", "Python", "Kubernetes", "Prometheus", "Grafana"],
        missing_skills=[],
        matched_projects=["变更前检查工具", "SLI 看板和告警路由"],
        jd_direction="运维开发",
        entity_split="train",
        rationale="4 年经验与发布、监控、故障响应职责高度一致。",
    ),
    _row(
        10,
        jd_text=(
            "岗位：安全研发工程师\n职责：开展代码审计、攻击检测和安全自动化建设。\n"
            "要求：本科及以上，3 年以上安全经验；熟悉 Python、Web 安全、代码审计、流量分析和 Linux。"
        ),
        level="较匹配",
        direction_match=True,
        education_match=True,
        experience_match=False,
        matched_skills=["Python", "Web安全", "代码审计", "流量分析", "Linux"],
        missing_skills=[],
        matched_projects=["主机基线巡检工具", "攻击链检测规则"],
        jd_direction="安全工程",
        entity_split="train",
        rationale="专业能力匹配，但 2 年经验未满足 3 年硬门槛。",
    ),
    _row(
        12,
        jd_text=(
            "岗位：高性能计算工程师\n职责：优化 GPU 算子和多机集合通信。\n"
            "要求：博士学历，工业经验不限；熟悉 C++、CUDA、MPI、NCCL、Linux 和性能分析工具。"
        ),
        level="较匹配",
        direction_match=True,
        education_match=False,
        experience_match=True,
        matched_skills=["C++", "CUDA", "MPI", "NCCL", "Linux", "Nsight Systems"],
        missing_skills=[],
        matched_projects=["CUDA 归约算子", "集合通信负载分析"],
        jd_direction="高性能计算",
        entity_split="train",
        rationale="研究技能完整且岗位不限工业经验，但硕士学历未达到明确的博士门槛。",
    ),
    _row(
        16,
        jd_text=(
            "岗位：AI 产品经理\n职责：负责知识助手需求规划、效果指标和跨团队交付。\n"
            "要求：本科及以上，3 年以上产品经验；掌握需求分析、PRD、原型、SQL、埋点和 A/B 测试。"
        ),
        level="较匹配",
        direction_match=True,
        education_match=True,
        experience_match=False,
        matched_skills=["需求分析", "PRD", "Axure", "SQL", "埋点设计", "A/B测试"],
        missing_skills=[],
        matched_projects=["知识助手反馈和引用流程", "检索失败样本复盘"],
        jd_direction="产品经理",
        entity_split="train",
        rationale="能力项与项目匹配，但只有 1 年产品经验，未达到 3 年要求。",
    ),
    _row(
        11,
        jd_text=(
            "岗位：嵌入式软件工程师\n职责：开发 MCU 固件、外设驱动并完成板级联调。\n"
            "要求：本科及以上，3 年以上经验；熟悉 C、C++、RTOS、STM32、CAN、SPI 和示波器。"
        ),
        level="高匹配",
        direction_match=True,
        education_match=True,
        experience_match=True,
        matched_skills=["C", "C++", "FreeRTOS", "STM32", "CAN", "SPI", "示波器"],
        missing_skills=[],
        matched_projects=["CAN 异常恢复", "采样任务调度优化"],
        jd_direction="嵌入式开发",
        entity_split="valid",
        rationale="经验年限、固件技能与板级问题定位经历均满足要求。",
    ),
    _row(
        13,
        jd_text=(
            "岗位：iOS 客户端工程师\n职责：负责 iOS 应用架构、离线缓存和性能优化。\n"
            "要求：硕士及以上，2 年以上经验；熟悉 Swift、UIKit、SwiftUI、Core Data 和 Instruments。"
        ),
        level="低匹配",
        direction_match=False,
        education_match=False,
        experience_match=False,
        matched_skills=["性能优化"],
        missing_skills=["Swift", "UIKit", "SwiftUI", "Core Data", "Instruments"],
        matched_projects=["客户端首屏性能优化"],
        jd_direction="iOS客户端开发",
        entity_split="valid",
        rationale="Android 经验有少量可迁移性，但学历、平台方向和全部 iOS 核心技术均不匹配。",
    ),
    _row(
        14,
        jd_text=(
            "岗位：硬件研发工程师\n职责：负责控制板原理图、电源设计、调试和可靠性验证。\n"
            "要求：硕士及以上，3 年以上经验；熟悉 Cadence、示波器、电源设计、EMC 和 ESD 防护。"
        ),
        level="较匹配",
        direction_match=True,
        education_match=False,
        experience_match=True,
        matched_skills=["Cadence", "示波器", "电源设计", "EMC", "ESD防护"],
        missing_skills=[],
        matched_projects=["电源纹波问题定位", "接口 ESD 防护验证"],
        jd_direction="硬件研发",
        entity_split="test",
        rationale="经验、工具和项目满足要求，但本科学历未达到明确的硕士门槛。",
    ),
    _row(
        15,
        jd_text=(
            "岗位：Java 后端工程师\n职责：开发交易服务和消息处理链路。\n"
            "要求：本科及以上，2 年以上后端经验；熟悉 Java、Spring Boot、MySQL、Redis 和 Kafka。"
        ),
        level="低匹配",
        direction_match=False,
        education_match=True,
        experience_match=False,
        matched_skills=[],
        missing_skills=["Java", "Spring Boot", "MySQL", "Redis", "Kafka"],
        matched_projects=[],
        jd_direction="后端开发",
        entity_split="test",
        rationale="智驾实习与后端交易岗位方向不同，且没有核心 Java 技术和 2 年后端经验。",
    ),
]
