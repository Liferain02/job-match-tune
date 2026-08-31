from __future__ import annotations

from typing import Any


def _row(
    number: int,
    *,
    text: str,
    target: str,
    education: list[str],
    skills: list[str],
    internships: list[str],
    projects: list[str],
    strengths: list[str],
) -> dict[str, Any]:
    row_id = f"resume_curated_train_{number:03d}"
    return {
        "id": row_id,
        "task": "resume_parse",
        "source_type": "curated_fictional",
        "source_group": row_id,
        "text": text,
        "label": {
            "目标岗位": target,
            "教育背景": education,
            "核心技能": skills,
            "实习经历": internships,
            "项目经历": projects,
            "优势标签": strengths,
        },
        "meta": {
            "language": "zh",
            "provenance": "repository_curated_fictional_v1",
            "curation_status": "repository_curated_unverified",
            "training_eligible": True,
            "contains_real_person_data": False,
            "generator": "ai_assisted_individual_case_authoring",
        },
    }


CURATED_RESUME_TRAIN_ROWS = [
    _row(
        1,
        text=(
            "求职方向：Java 后端开发\n"
            "教育经历：某理工大学，软件工程本科\n"
            "工作经历：4 年后端研发经验。负责会员与订单服务，参与服务拆分、容量评估和线上故障复盘。\n"
            "技术栈：Java、Spring Boot、MySQL、Redis、Kafka、Docker、Kubernetes\n"
            "项目：主导订单查询链路重构，将核心接口 P99 延迟从 420ms 降至 170ms；设计消息幂等和失败补偿机制。\n"
            "特点：能够独立完成系统设计、压测和上线治理。"
        ),
        target="后端开发",
        education=["某理工大学，软件工程本科"],
        skills=["Java", "Spring Boot", "MySQL", "Redis", "Kafka", "Docker", "Kubernetes"],
        internships=[],
        projects=[
            "主导订单查询链路重构，将核心接口 P99 延迟从 420ms 降至 170ms。",
            "设计消息幂等和失败补偿机制。",
        ],
        strengths=["系统设计", "性能优化", "线上治理"],
    ),
    _row(
        2,
        text=(
            "目标岗位 Go 服务端工程师\n"
            "学历 计算机技术硕士\n"
            "两年工作经验，维护广告检索服务和特征查询接口。\n"
            "技能 Go / gRPC / PostgreSQL / Redis / Prometheus / Linux\n"
            "代表工作\n"
            "- 将批量查询接口改为并发流水线，吞吐量提升约 45%\n"
            "- 完成慢查询治理、指标埋点和告警分级"
        ),
        target="后端开发",
        education=["计算机技术硕士"],
        skills=["Go", "gRPC", "PostgreSQL", "Redis", "Prometheus", "Linux"],
        internships=[],
        projects=[
            "将批量查询接口改为并发流水线，吞吐量提升约 45%。",
            "完成慢查询治理、指标埋点和告警分级。",
        ],
        strengths=["Go服务开发", "可观测性", "数据库优化"],
    ),
    _row(
        3,
        text=(
            "应聘：大模型应用开发\n"
            "本科｜计算机科学与技术\n"
            "实习：在企业智能化团队参与知识库问答项目，负责文档解析、召回评测与 FastAPI 接口。\n"
            "技能：Python、FastAPI、RAG、LangChain、Milvus、MySQL\n"
            "项目经历：构建技术文档问答助手，整理 300 条问题集评估召回率；实现引用溯源和无答案拒答。\n"
            "优势：重视评测，能把模型能力接入业务系统。"
        ),
        target="AI应用开发",
        education=["计算机科学与技术本科"],
        skills=["Python", "FastAPI", "RAG", "LangChain", "Milvus", "MySQL"],
        internships=["在企业智能化团队参与知识库问答项目，负责文档解析、召回评测与 FastAPI 接口。"],
        projects=[
            "构建技术文档问答助手，整理 300 条问题集评估召回率。",
            "实现引用溯源和无答案拒答。",
        ],
        strengths=["RAG评测", "业务系统集成", "问题分析"],
    ),
    _row(
        4,
        text=(
            "研究方向：自然语言处理 / 检索排序\n"
            "教育背景\n某综合大学 人工智能 硕士\n"
            "核心能力\nPython，PyTorch，Transformers，信息检索，模型蒸馏，离线评估\n"
            "科研与项目\n1. 训练中文语义匹配模型，设计困难负例采样并完成消融实验。\n"
            "2. 为检索系统搭建 Recall@K、MRR 与人工 badcase 复核流程。\n"
            "曾在搜索算法团队实习，负责样本清洗和训练结果分析。"
        ),
        target="算法工程",
        education=["某综合大学，人工智能硕士"],
        skills=["Python", "PyTorch", "Transformers", "信息检索", "模型蒸馏", "离线评估"],
        internships=["在搜索算法团队实习，负责样本清洗和训练结果分析。"],
        projects=[
            "训练中文语义匹配模型，设计困难负例采样并完成消融实验。",
            "为检索系统搭建 Recall@K、MRR 与人工 badcase 复核流程。",
        ],
        strengths=["实验设计", "困难样本分析", "模型评测"],
    ),
    _row(
        5,
        text=(
            "目标岗位：AI Infra\n"
            "教育：计算机工程本科\n"
            "经历：3 年模型平台研发，负责训练任务编排、GPU 配额和推理服务发布链路。\n"
            "Skills: Python, Go, Kubernetes, Docker, Slurm, Prometheus, NVIDIA GPU\n"
            "项目成果：实现队列优先级与失败重试；建设 GPU 利用率看板，定位空转任务并降低资源浪费。"
        ),
        target="AI Infra",
        education=["计算机工程本科"],
        skills=["Python", "Go", "Kubernetes", "Docker", "Slurm", "Prometheus", "GPU"],
        internships=[],
        projects=[
            "实现训练任务队列优先级与失败重试。",
            "建设 GPU 利用率看板，定位空转任务并降低资源浪费。",
        ],
        strengths=["训练平台", "GPU调度", "资源治理"],
    ),
    _row(
        6,
        text=(
            "数据开发工程师｜2 年经验\n"
            "教育背景：数据科学本科\n"
            "熟悉 SQL、Flink、Kafka、Spark、Hive、Airflow、ClickHouse。\n"
            "负责实时指标链路和离线数仓任务，处理口径变更、延迟排查与数据回补。\n"
            "项目亮点：重构用户行为宽表；为实时链路增加数据质量校验和延迟告警。"
        ),
        target="数据开发",
        education=["数据科学本科"],
        skills=["SQL", "Flink", "Kafka", "Spark", "Hive", "Airflow", "ClickHouse"],
        internships=[],
        projects=["重构用户行为宽表。", "为实时链路增加数据质量校验和延迟告警。"],
        strengths=["数仓建模", "实时计算", "数据质量"],
    ),
    _row(
        7,
        text=(
            "个人方向 前端开发\n"
            "本科在读，软件工程\n"
            "技术：TypeScript / React / Vite / Zustand / ECharts / Playwright\n"
            "实习内容：参与运营中台开发，负责表单组件、权限配置和接口联调。\n"
            "课程项目：实现可视化流程编辑器，支持节点拖拽、撤销重做和配置校验。\n"
            "关注工程化、可访问性和页面性能。"
        ),
        target="前端开发",
        education=["软件工程本科在读"],
        skills=["TypeScript", "React", "Vite", "Zustand", "ECharts", "Playwright"],
        internships=["参与运营中台开发，负责表单组件、权限配置和接口联调。"],
        projects=["实现可视化流程编辑器，支持节点拖拽、撤销重做和配置校验。"],
        strengths=["前端工程化", "组件设计", "页面性能"],
    ),
    _row(
        8,
        text=(
            "求职意向：测试开发\n"
            "学历：软件工程本科\n"
            "职业经历：3 年质量平台研发。维护接口自动化、测试数据构造和持续集成任务。\n"
            "工具与语言：Python、Pytest、Selenium、JMeter、Jenkins、MySQL\n"
            "工作成果：将核心接口回归接入流水线；编写故障注入脚本验证降级和重试策略。"
        ),
        target="测试开发",
        education=["软件工程本科"],
        skills=["Python", "Pytest", "Selenium", "JMeter", "Jenkins", "MySQL"],
        internships=[],
        projects=["将核心接口回归接入流水线。", "编写故障注入脚本验证降级和重试策略。"],
        strengths=["自动化测试", "持续集成", "稳定性验证"],
    ),
    _row(
        9,
        text=(
            "SRE / 运维开发，4 年工作经验\n"
            "计算机相关专业本科\n"
            "能力栈：Linux、Python、Shell、Kubernetes、Prometheus、Grafana、ELK\n"
            "负责发布巡检、容量管理和值班故障响应。\n"
            "实践：开发变更前检查工具；统一服务 SLI 看板和告警路由；推动三类重复故障形成自动化处置。"
        ),
        target="运维开发",
        education=["计算机相关专业本科"],
        skills=["Linux", "Python", "Shell", "Kubernetes", "Prometheus", "Grafana", "ELK"],
        internships=[],
        projects=[
            "开发变更前检查工具。",
            "统一服务 SLI 看板和告警路由。",
            "推动三类重复故障形成自动化处置。",
        ],
        strengths=["故障响应", "可观测性", "自动化运维"],
    ),
    _row(
        10,
        text=(
            "目标：安全工程师\n"
            "教育经历：网络空间安全硕士\n"
            "两年安全运营与研发经验，参与漏洞复现、代码审计和告警处置。\n"
            "核心技能：Python、Web安全、代码审计、流量分析、YARA、Linux\n"
            "项目经历：编写主机基线巡检工具；为常见攻击链补充检测规则并完成误报复盘。"
        ),
        target="安全工程",
        education=["网络空间安全硕士"],
        skills=["Python", "Web安全", "代码审计", "流量分析", "YARA", "Linux"],
        internships=[],
        projects=["编写主机基线巡检工具。", "为常见攻击链补充检测规则并完成误报复盘。"],
        strengths=["漏洞分析", "检测规则", "安全自动化"],
    ),
    _row(
        11,
        text=(
            "嵌入式软件开发｜3 年\n"
            "自动化专业本科\n"
            "C、C++、FreeRTOS、STM32、CAN、SPI、示波器\n"
            "从事电机控制板固件和外设驱动开发，负责板级联调与现场问题定位。\n"
            "完成 CAN 通信异常恢复机制；优化采样任务调度，解决高负载下的数据丢失。"
        ),
        target="嵌入式开发",
        education=["自动化专业本科"],
        skills=["C", "C++", "FreeRTOS", "STM32", "CAN", "SPI", "示波器"],
        internships=[],
        projects=["完成 CAN 通信异常恢复机制。", "优化采样任务调度，解决高负载下的数据丢失。"],
        strengths=["固件开发", "板级调试", "故障定位"],
    ),
    _row(
        12,
        text=(
            "申请方向：高性能计算\n"
            "计算机系统结构硕士\n"
            "研究与工程技能：C++、CUDA、MPI、NCCL、Linux、Nsight Systems\n"
            "实验室经历：参与多机训练通信性能分析。\n"
            "项目：实现 CUDA 归约算子并比较不同 block 配置；定位集合通信中的负载不均问题。"
        ),
        target="高性能计算",
        education=["计算机系统结构硕士"],
        skills=["C++", "CUDA", "MPI", "NCCL", "Linux", "Nsight Systems"],
        internships=["在实验室参与多机训练通信性能分析。"],
        projects=["实现 CUDA 归约算子并比较不同 block 配置。", "定位集合通信中的负载不均问题。"],
        strengths=["CUDA优化", "性能分析", "并行计算"],
    ),
    _row(
        13,
        text=(
            "Android 客户端开发，2 年经验\n"
            "学历：通信工程本科\n"
            "Kotlin、Android、Jetpack Compose、Coroutines、Room、性能优化\n"
            "负责播放器业务页面与离线缓存模块。\n"
            "代表项目：改造首屏加载链路并减少主线程阻塞；建设崩溃聚合和版本回归看板。"
        ),
        target="客户端开发",
        education=["通信工程本科"],
        skills=["Kotlin", "Android", "Jetpack Compose", "Coroutines", "Room", "性能优化"],
        internships=[],
        projects=["改造首屏加载链路并减少主线程阻塞。", "建设崩溃聚合和版本回归看板。"],
        strengths=["Android开发", "性能优化", "稳定性治理"],
    ),
    _row(
        14,
        text=(
            "硬件研发工程师\n"
            "电子信息工程本科；3 年硬件设计与调试经验。\n"
            "工具技能：Cadence、Altium Designer、示波器、逻辑分析仪、电源设计、EMC\n"
            "工作内容：完成控制板原理图和器件选型，跟进打样、焊接与可靠性测试。\n"
            "项目结果：解决电源纹波超标问题；完成接口 ESD 防护方案验证。"
        ),
        target="硬件研发",
        education=["电子信息工程本科"],
        skills=["Cadence", "Altium Designer", "示波器", "逻辑分析仪", "电源设计", "EMC"],
        internships=[],
        projects=["解决电源纹波超标问题。", "完成接口 ESD 防护方案验证。"],
        strengths=["原理图设计", "硬件调试", "可靠性验证"],
    ),
    _row(
        15,
        text=(
            "求职方向 汽车软件 / 智驾研发\n"
            "控制科学与工程硕士\n"
            "实习经历：在泊车团队负责场景回放、问题定位和回归验证。\n"
            "技术能力：Python、C++、ROS、CAN、自动驾驶仿真、车端日志分析\n"
            "项目经历：搭建泊车场景批量回放脚本；分析感知结果与车辆状态时间戳偏差。"
        ),
        target="汽车软件/智驾研发",
        education=["控制科学与工程硕士"],
        skills=["Python", "C++", "ROS", "CAN", "自动驾驶仿真", "车端日志分析"],
        internships=["在泊车团队负责场景回放、问题定位和回归验证。"],
        projects=["搭建泊车场景批量回放脚本。", "分析感知结果与车辆状态时间戳偏差。"],
        strengths=["场景验证", "日志分析", "问题闭环"],
    ),
    _row(
        16,
        text=(
            "应聘岗位：AI 产品经理\n"
            "信息管理本科，1 年产品工作经验\n"
            "能力：需求分析、PRD、Axure、SQL、埋点设计、A/B测试\n"
            "负责知识助手的需求拆解、版本规划和效果指标复盘。\n"
            "项目：设计答案反馈与引用查看流程；结合检索失败样本推动知识库更新机制。"
        ),
        target="产品经理",
        education=["信息管理本科"],
        skills=["需求分析", "PRD", "Axure", "SQL", "埋点设计", "A/B测试"],
        internships=[],
        projects=["设计答案反馈与引用查看流程。", "结合检索失败样本推动知识库更新机制。"],
        strengths=["需求拆解", "数据分析", "跨团队协作"],
    ),
]
