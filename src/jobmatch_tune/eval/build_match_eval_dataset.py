from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from jobmatch_tune.dataset.curated_match_training_data import CURATED_MATCH_TRAIN_ROWS
from jobmatch_tune.utils.io import write_jsonl


ROWS = [
    {
        "id": "match_eval_001",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：AI 应用开发工程师\n岗位职责：负责知识库问答、Agent 应用和后端接口开发。\n任职要求：熟悉 Python、FastAPI、RAG、MySQL、Redis。本科及以上学历，三年以上工作经验。",
        "resume_text": "目标岗位：AI应用开发\n教育背景：本科，计算机科学与技术\n核心技能：Python、FastAPI、RAG、MySQL、Redis、Docker\n项目经历：负责企业知识库问答系统开发；参与工单智能分派 Agent 应用建设。\n实习经历：后端开发实习，负责接口开发。",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python", "FastAPI", "RAG", "MySQL", "Redis"],
            "缺失技能": [],
        },
    },
    {
        "id": "match_eval_002",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：后端开发工程师\n岗位职责：负责交易链路服务开发与治理。\n任职要求：熟悉 Java、Spring Boot、MySQL、Redis、Kafka。本科及以上学历，三年以上工作经验。",
        "resume_text": "目标岗位：后端开发\n教育背景：硕士，软件工程\n核心技能：Java、Spring Boot、MySQL、Redis、Docker\n项目经历：负责订单中心重构；参与服务限流治理。\n实习经历：支付平台接口开发。",
        "label": {
            "匹配等级": "较匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Java", "Spring Boot", "MySQL", "Redis"],
            "缺失技能": ["Kafka"],
        },
    },
    {
        "id": "match_eval_003",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：前端开发工程师\n岗位职责：负责中后台和可视化页面开发。\n任职要求：熟悉 TypeScript、React、Vite、ECharts。本科及以上学历。",
        "resume_text": "目标岗位：前端开发\n教育背景：本科，计算机科学与技术\n核心技能：Vue、TypeScript、ECharts、Vite、Playwright\n项目经历：开发 BI 看板组件库；参与复杂表格界面开发。",
        "label": {
            "匹配等级": "较匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["TypeScript", "Vite", "ECharts"],
            "缺失技能": ["React"],
        },
    },
    {
        "id": "match_eval_004",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：AI Infra 工程师\n岗位职责：负责训练平台、推理平台和 GPU 调度系统研发。\n任职要求：熟悉 Python、Go、Kubernetes、Docker、Linux、Prometheus。本科及以上学历。",
        "resume_text": "目标岗位：AI Infra\n教育背景：本科，计算机工程\n核心技能：Python、Go、Kubernetes、Docker、Linux、Prometheus、GPU、Slurm\n项目经历：开发模型训练任务编排系统；负责推理服务部署链路监控告警。",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python", "Go", "Kubernetes", "Docker", "Linux", "Prometheus"],
            "缺失技能": [],
        },
    },
    {
        "id": "match_eval_005",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：高性能计算工程师\n岗位职责：负责 GPU 多机通信优化和 HPC 集群性能调优。\n任职要求：熟悉 C++、CUDA、MPI、NCCL、Linux。硕士及以上学历。",
        "resume_text": "目标岗位：高性能计算\n教育背景：硕士，计算机系统结构\n核心技能：C++、CUDA、MPI、Linux、性能分析\n项目经历：完成 GPU 多机训练通信链路优化。",
        "label": {
            "匹配等级": "较匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["C++", "CUDA", "MPI", "Linux"],
            "缺失技能": ["NCCL"],
        },
    },
    {
        "id": "match_eval_006",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：硬件研发工程师\n岗位职责：负责电源板原理图设计和硬件调试。\n任职要求：熟悉 Cadence、示波器、电源设计、EMC、C语言。本科及以上学历。",
        "resume_text": "目标岗位：硬件研发\n教育背景：本科，电子信息工程\n核心技能：Altium Designer、Cadence、示波器、电源设计、EMC、C语言\n项目经历：负责功率板硬件原理图设计；参与硬件调试与可靠性测试。",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Cadence", "示波器", "电源设计", "EMC", "C语言"],
            "缺失技能": [],
        },
    },
    {
        "id": "match_eval_007",
        "task": "match",
        "source_type": "ocr_like",
        "jd_text": "岗位名称：运维开发工程师\n岗位职责：负责监控、告警、巡检和自动化运维平台开发。\n任职要求：熟悉 Python、Shell、Linux、Kubernetes、Prometheus、Grafana。本科及以上学历。",
        "resume_text": "目标岗位:运维开发\n教育背景:本科,计算机科学与技术\n核心技能:Python Shell Linux Kubernetes Prometheus ELK\n项目经历:开发服务巡检机器人;参与日志平台接入\n实习经历:在 SRE 团队参与监控告警和发布巡检",
        "label": {
            "匹配等级": "较匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python", "Shell", "Linux", "Kubernetes", "Prometheus"],
            "缺失技能": ["Grafana"],
        },
    },
    {
        "id": "match_eval_008",
        "task": "match",
        "source_type": "ocr_like",
        "jd_text": "岗位名称：测试开发工程师\n岗位职责：负责接口自动化和稳定性验证。\n任职要求：熟悉 Python、Pytest、Selenium、JMeter、MySQL。本科及以上学历。",
        "resume_text": "目标岗位:测试开发\n教育背景:本科,信息安全\n核心技能:Python Py test Selenium Postman My SOL\n项目经历:搭建接口回归测试平台;开发稳定性巡检脚本",
        "label": {
            "匹配等级": "较匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python", "Selenium"],
            "缺失技能": ["Pytest", "JMeter", "MySQL"],
        },
    },
    {
        "id": "match_eval_009",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：数据开发工程师\n岗位职责：负责实时数仓建设和 ETL 链路开发。\n任职要求：熟悉 SQL、Spark、Hive、Flink、Airflow。本科及以上学历。",
        "resume_text": "目标岗位：数据开发\n教育背景：本科，数据科学与大数据技术\n核心技能：Python、SQL、Spark、Hive、Airflow、ClickHouse\n项目经历：开发实时指标 ETL 链路；参与数仓宽表建设。",
        "label": {
            "匹配等级": "较匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["SQL", "Spark", "Hive", "Airflow"],
            "缺失技能": ["Flink"],
        },
    },
    {
        "id": "match_eval_010",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：网络与基础设施工程师\n岗位职责：负责网络规划、巡检自动化和链路优化。\n任职要求：熟悉 Linux、TCP/IP、BGP、OSPF、Python、Ansible。本科及以上学历。",
        "resume_text": "目标岗位：网络与基础设施\n教育背景：本科，网络工程\n核心技能：Linux、TCP/IP、BGP、OSPF、Python、Ansible、Prometheus\n项目经历：负责网络巡检自动化脚本开发；参与边缘机房容量评估。",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Linux", "TCP/IP", "BGP", "OSPF", "Python", "Ansible"],
            "缺失技能": [],
        },
    },
    {
        "id": "match_eval_011",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：嵌入式开发工程师\n岗位职责：负责驱动开发和板级调试。\n任职要求：熟悉 C、C++、RTOS、STM32、CAN。本科及以上学历。",
        "resume_text": "目标岗位：嵌入式开发\n教育背景：本科，自动化\n核心技能：C、C++、RTOS、STM32、CAN、驱动开发\n项目经历：负责电机控制板固件开发；参与车载通信模块驱动适配。",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["C", "C++", "RTOS", "STM32", "CAN"],
            "缺失技能": [],
        },
    },
    {
        "id": "match_eval_012",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：汽车软件/智驾研发工程师\n岗位职责：负责自动泊车功能开发和智驾软件集成。\n任职要求：熟悉 Python、C++、ROS、自动驾驶仿真。本科及以上学历。",
        "resume_text": "目标岗位：汽车软件/智驾研发\n教育背景：硕士，车辆工程\n核心技能：Python、C++、ROS、感知融合、自动驾驶仿真、CANape\n项目经历：参与感知融合模块开发；负责自动泊车功能回归测试。",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python", "C++", "ROS", "自动驾驶仿真"],
            "缺失技能": [],
        },
    },
    {
        "id": "match_eval_013",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：安全工程师\n岗位职责：负责漏洞挖掘与安全巡检自动化建设。\n任职要求：熟悉 Python、Web安全、代码审计、流量分析。本科及以上学历。",
        "resume_text": "目标岗位：安全工程\n教育背景：本科，信息安全\n核心技能：Python、漏洞挖掘、Web安全、代码审计、YARA、流量分析\n项目经历：开发安全巡检脚本；参与漏洞复现和应急响应演练。",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python", "Web安全", "代码审计", "流量分析"],
            "缺失技能": [],
        },
    },
    {
        "id": "match_eval_014",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：高性能计算工程师\n岗位职责：负责 GPU 通信优化和集群性能调优。\n任职要求：熟悉 C++、CUDA、MPI、NCCL、Linux。硕士及以上学历。",
        "resume_text": "目标岗位：AI Infra\n教育背景：本科，计算机工程\n核心技能：Python、Go、Kubernetes、Docker、Linux、Prometheus、GPU\n项目经历：开发模型训练任务编排系统；负责 GPU 监控告警建设。",
        "label": {
            "匹配等级": "低匹配",
            "岗位方向匹配": False,
            "学历匹配": False,
            "经验匹配": True,
            "命中技能": ["Linux"],
            "缺失技能": ["C++", "CUDA", "MPI", "NCCL"],
        },
    },
    {
        "id": "match_eval_015",
        "task": "match",
        "source_type": "ocr_like",
        "jd_text": "岗位名称：客户端开发工程师\n岗位职责：负责 Android 客户端功能开发和性能优化。\n任职要求：熟悉 Kotlin、Android、Jetpack、性能优化。本科及以上学历。",
        "resume_text": "目标岗位:客户端开发\n教育背景:本科,软件工程\n核心技能:Kotlin Android Jetpack C + + FFmpeg\n项目经历:负责播放器首屏优化;参与音视频 SDK 封装",
        "label": {
            "匹配等级": "高匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Kotlin", "Android", "Jetpack", "性能优化"],
            "缺失技能": [],
        },
    },
    {
        "id": "match_eval_016",
        "task": "match",
        "source_type": "ocr_like",
        "jd_text": "岗位名称：AI Infra 工程师\n岗位职责：负责训练平台和 GPU 调度平台研发。\n任职要求：熟悉 Python、Go、Kubernetes、Docker、Linux、Prometheus。本科及以上学历。",
        "resume_text": "目标岗位:AI Infra\n教育背景:硕士,计算机技术\n核心技能:Python Go Kubernet es Docker Linux Terraform\n项目经历:参与训练任务编排服务开发;负责 GPU 资源池监控告警",
        "label": {
            "匹配等级": "较匹配",
            "岗位方向匹配": True,
            "学历匹配": True,
            "经验匹配": True,
            "命中技能": ["Python", "Go", "Docker", "Linux"],
            "缺失技能": ["Kubernetes", "Prometheus"],
        },
    },
]


LEGACY_DIFFICULTY_TAGS = {
    "match_eval_001": ["AI算法后端交叉"],
    "match_eval_002": ["单项硬门槛不满足"],
    "match_eval_003": ["可迁移技能"],
    "match_eval_004": ["AI算法后端交叉"],
    "match_eval_005": ["技能仅在项目中"],
    "match_eval_006": ["技能同义词"],
    "match_eval_007": ["OCR噪声"],
    "match_eval_008": ["OCR噪声"],
    "match_eval_009": ["技能仅在项目中"],
    "match_eval_010": ["相近岗位方向"],
    "match_eval_011": ["技能仅在项目中"],
    "match_eval_012": ["相近岗位方向"],
    "match_eval_013": ["技能仅在项目中"],
    "match_eval_014": ["学历不完全满足", "相近岗位方向"],
    "match_eval_015": ["OCR噪声"],
    "match_eval_016": ["OCR噪声", "AI算法后端交叉"],
}


CHALLENGE_ROWS = [
    {
        "id": "match_gold_candidate_001",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：数据平台后端工程师\n任职要求：熟悉 Python、PostgreSQL、Redis，本科及以上学历。",
        "resume_text": "目标岗位：后端开发\n教育背景：本科\n核心技能：Python、Postgres、Redis\n项目经历：使用 Postgres 开发数据服务。",
        "label": {"匹配等级": "高匹配", "岗位方向匹配": True, "学历匹配": True, "经验匹配": True, "命中技能": ["Python", "PostgreSQL", "Redis"], "缺失技能": []},
        "difficulty_tags": ["技能同义词"],
        "draft_rationale": "Postgres 是 PostgreSQL 的常见同义表达。",
    },
    {
        "id": "match_gold_candidate_002",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：React 前端工程师\n任职要求：熟悉 TypeScript、React、Vite，本科及以上学历。",
        "resume_text": "目标岗位：前端开发\n教育背景：本科\n核心技能：TypeScript、Vue、Vite\n项目经历：独立开发大型 Vue 中后台并维护组件库。",
        "label": {"匹配等级": "较匹配", "岗位方向匹配": True, "学历匹配": True, "经验匹配": True, "命中技能": ["TypeScript", "Vite"], "缺失技能": ["React"]},
        "difficulty_tags": ["可迁移技能"],
        "draft_rationale": "Vue 经验可迁移，但不能当作已经掌握 React。",
    },
    {
        "id": "match_gold_candidate_003",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：AI 应用后端工程师\n岗位职责：负责 RAG 服务和 API 开发。\n任职要求：Python、FastAPI、RAG。",
        "resume_text": "目标岗位：Python 后端开发\n核心技能：Python、FastAPI、MySQL\n项目经历：开发检索增强问答系统并上线接口服务。",
        "label": {"匹配等级": "高匹配", "岗位方向匹配": True, "学历匹配": True, "经验匹配": True, "命中技能": ["Python", "FastAPI", "RAG"], "缺失技能": []},
        "difficulty_tags": ["相近岗位方向", "AI算法后端交叉", "技能仅在项目中"],
        "draft_rationale": "目标岗位名称不同，但项目内容与 AI 应用后端核心职责一致。",
    },
    {
        "id": "match_gold_candidate_004",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：后端开发工程师\n任职要求：Java、Spring Boot、MySQL，三年以上工作经验，本科及以上学历。",
        "resume_text": "目标岗位：后端开发\n教育背景：本科\n核心技能：Java、Spring Boot、MySQL\n工作经历：一年后端开发\n项目经历：独立负责百万日订单系统重构和压测。",
        "label": {"匹配等级": "较匹配", "岗位方向匹配": True, "学历匹配": True, "经验匹配": False, "命中技能": ["Java", "Spring Boot", "MySQL"], "缺失技能": []},
        "difficulty_tags": ["年限不足但项目强", "单项硬门槛不满足"],
        "draft_rationale": "项目证据强，但明确的一年经历仍未满足三年硬要求。",
    },
    {
        "id": "match_gold_candidate_005",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：测试开发工程师\n任职要求：Python、Pytest、Selenium，本科及以上学历。",
        "resume_text": "目标岗位：测试开发\n教育背景：大专，软件技术\n核心技能：Python、Pytest、Selenium\n项目经历：搭建接口与 UI 自动化测试平台。",
        "label": {"匹配等级": "较匹配", "岗位方向匹配": True, "学历匹配": False, "经验匹配": True, "命中技能": ["Python", "Pytest", "Selenium"], "缺失技能": []},
        "difficulty_tags": ["学历不完全满足", "单项硬门槛不满足"],
        "draft_rationale": "技能和方向符合，但大专不满足本科硬要求。",
    },
    {
        "id": "match_gold_candidate_006",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：Python 后端工程师\n任职要求：Python、FastAPI、Redis。",
        "resume_text": "目标岗位：后端开发\n核心技能：Python、Redis\n项目经历：使用 FastAPI 开发网关服务，承担鉴权和限流模块。",
        "label": {"匹配等级": "高匹配", "岗位方向匹配": True, "学历匹配": True, "经验匹配": True, "命中技能": ["Python", "FastAPI", "Redis"], "缺失技能": []},
        "difficulty_tags": ["技能仅在项目中"],
        "draft_rationale": "FastAPI 未列在技能栏，但项目中有明确使用证据。",
    },
    {
        "id": "match_gold_candidate_007",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：算法平台工程师\n岗位职责：负责模型服务、特征流水线和平台 API。\n任职要求：Python、PyTorch、FastAPI、Docker。",
        "resume_text": "目标岗位：AI 后端工程师\n核心技能：Python、PyTorch、FastAPI、Docker\n项目经历：部署分类模型并开发在线推理 API。",
        "label": {"匹配等级": "高匹配", "岗位方向匹配": True, "学历匹配": True, "经验匹配": True, "命中技能": ["Python", "PyTorch", "FastAPI", "Docker"], "缺失技能": []},
        "difficulty_tags": ["AI算法后端交叉", "相近岗位方向"],
        "draft_rationale": "算法、平台和后端交叉岗位，应依据职责和项目而非岗位名称判定。",
    },
    {
        "id": "match_gold_candidate_008",
        "task": "match",
        "source_type": "ocr_like",
        "jd_text": "岗位名称：AI Infra 工程师\n任职要求：Python、Kubernetes、C++、MySQL。",
        "resume_text": "目标岗位:AI Infra\n核心技能:Py thon Kubernet es C + + My SOL\n项目经历:负责 GPU 任务调度和集群监控",
        "label": {"匹配等级": "高匹配", "岗位方向匹配": True, "学历匹配": True, "经验匹配": True, "命中技能": ["Python", "Kubernetes", "C++", "MySQL"], "缺失技能": []},
        "difficulty_tags": ["OCR噪声"],
        "draft_rationale": "四个技能均有典型 OCR 断词或误识别形式。",
    },
    {
        "id": "match_gold_candidate_009",
        "task": "match",
        "source_type": "text",
        "jd_text": "岗位名称：高性能计算工程师\n任职要求：C++、CUDA、NCCL、Linux，硕士及以上学历，五年以上经验。",
        "resume_text": "目标岗位：高性能计算\n教育背景：硕士\n核心技能：C++、CUDA、NCCL、Linux\n工作经历：两年 GPU 通信优化\n项目经历：优化多机训练集合通信性能。",
        "label": {"匹配等级": "较匹配", "岗位方向匹配": True, "学历匹配": True, "经验匹配": False, "命中技能": ["C++", "CUDA", "NCCL", "Linux"], "缺失技能": []},
        "difficulty_tags": ["单项硬门槛不满足", "年限不足但项目强"],
        "draft_rationale": "唯一明确不满足的是五年经验门槛，不应掩盖其余强匹配证据。",
    },
]


def to_ocr_like(text: str) -> str:
    text = text.replace("：", ":")
    text = text.replace("，", ",")
    text = text.replace("；", ";")
    text = text.replace("。", "")
    text = text.replace("、", " ")
    text = text.replace("MySQL", "My SOL")
    text = text.replace("Pytest", "Py test")
    text = text.replace("Kubernetes", "Kubernet es")
    text = text.replace("C++", "C + +")
    return text


def build_variant_rows(rows: list[dict]) -> list[dict]:
    variants = []
    for row in rows:
        copied = deepcopy(row)
        copied["id"] = f"{row['id']}_alt"
        copied["jd_text"] = (
            copied["jd_text"]
            .replace("岗位职责：", "工作内容：")
            .replace("任职要求：", "职位要求：")
        )
        copied["resume_text"] = to_ocr_like(copied["resume_text"]) if row.get("source_type") == "ocr_like" else copied["resume_text"].replace("目标岗位：", "求职方向：")
        variants.append(copied)
        copied2 = deepcopy(row)
        copied2["id"] = f"{row['id']}_ocr"
        copied2["source_type"] = "ocr_like"
        copied2["jd_text"] = to_ocr_like(copied2["jd_text"])
        copied2["resume_text"] = to_ocr_like(copied2["resume_text"])
        variants.append(copied2)
        copied3 = deepcopy(row)
        copied3["id"] = f"{row['id']}_compact"
        copied3["jd_text"] = (
            copied3["jd_text"]
            .replace("岗位名称：", "职位：")
            .replace("岗位职责：", "职责：")
            .replace("任职要求：", "要求：")
        )
        copied3["resume_text"] = (
            copied3["resume_text"]
            .replace("目标岗位：", "求职岗位：")
            .replace("教育背景：", "教育：")
            .replace("核心技能：", "技能：")
            .replace("项目经历：", "项目：")
            .replace("实习经历：", "实习：")
        )
        variants.append(copied3)
    return variants


def build_legacy_review_rows(rows: list[dict]) -> list[dict]:
    reviewed = []
    for row in rows:
        copied = deepcopy(row)
        copied["source_group"] = row["id"]
        copied["meta"] = {
            "annotation_status": "legacy_unverified",
            "annotation_provenance": "repository_seed_without_reviewer_record",
            "training_eligible": False,
            "intended_usage": "evaluation_review_only",
            "difficulty_tags": LEGACY_DIFFICULTY_TAGS[row["id"]],
        }
        reviewed.append(copied)
    return reviewed


def build_gold_review_candidates(rows: list[dict]) -> list[dict]:
    candidates = build_legacy_review_rows(rows)
    for row in CHALLENGE_ROWS:
        copied = deepcopy(row)
        difficulty_tags = copied.pop("difficulty_tags")
        draft_rationale = copied.pop("draft_rationale")
        copied["source_group"] = copied["id"]
        copied["meta"] = {
            "annotation_status": "needs_human_review",
            "annotation_provenance": "ai_draft_for_human_review",
            "annotator_id": "",
            "reviewed_at": "",
            "rationale": draft_rationale,
            "difficulty_tags": difficulty_tags,
            "training_eligible": False,
            "intended_usage": "gold_candidate_review_only",
        }
        candidates.append(copied)
    return candidates


def main() -> None:
    review_rows = build_legacy_review_rows(ROWS)
    gold_candidates = build_gold_review_candidates(ROWS)
    robustness_rows = build_variant_rows(review_rows)
    write_jsonl("data/eval/match_manual_eval_seed.jsonl", review_rows)
    write_jsonl("data/eval/match_gold_review_candidates.jsonl", gold_candidates)
    write_jsonl("data/eval/match_format_robustness_seed.jsonl", robustness_rows)
    write_jsonl("data/eval/match_curated_train_pool.jsonl", CURATED_MATCH_TRAIN_ROWS)
    legacy_train_pool = Path("data/eval/match_manual_train_pool.jsonl")
    removed_legacy_train_pool = legacy_train_pool.exists()
    legacy_train_pool.unlink(missing_ok=True)
    print(
        f"wrote {len(review_rows)} legacy review rows, {len(gold_candidates)} Gold review candidates, "
        f"{len(robustness_rows)} robustness rows and "
        f"{len(CURATED_MATCH_TRAIN_ROWS)} independent curated training pairs; "
        f"removed_leaked_train_pool={removed_legacy_train_pool}"
    )


if __name__ == "__main__":
    main()
