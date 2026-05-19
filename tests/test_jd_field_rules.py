from __future__ import annotations

import yaml

from jobmatch_tune.preprocess.jd_field_rules import infer_job_direction


def _schema() -> dict:
    with open("configs/label_schema.yaml", "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_infer_job_direction_returns_empty_for_non_tech_business_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "百度集团公关（支持智能驾驶事业群）",
        "岗位职责：负责品牌传播、公关合作和对外沟通。",
        schema,
    )
    assert direction == ""


def test_infer_job_direction_does_not_fallback_to_first_schema_class() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "国际费用结算",
        "岗位职责：负责费用结算、单据处理和流程跟进。",
        schema,
    )
    assert direction == ""


def test_infer_job_direction_keeps_real_backend_research_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "OLAP引擎研发",
        "岗位职责：负责 OLAP 引擎研发、数据库优化和分布式查询性能调优。",
        schema,
    )
    assert direction == "后端开发"


def test_infer_job_direction_accepts_algorithm_engineer_english_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "Text To Speech (TTS) Algorithm Engineer",
        "岗位职责：负责 TTS 模型训练、推理优化和语音生成效果提升。",
        schema,
    )
    assert direction == "算法工程"


def test_infer_job_direction_accepts_security_ops_engineer_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "安全运营工程师",
        "岗位职责：负责安全告警分析、漏洞处置和安全运营平台建设。",
        schema,
    )
    assert direction == "安全工程"


def test_infer_job_direction_accepts_security_ops_post_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "大模型安全运营",
        "岗位职责：负责大模型安全治理、风险监控与安全运营平台建设。",
        schema,
    )
    assert direction == "安全工程"


def test_infer_job_direction_accepts_backend_rd_post_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "B端招商入驻研发岗",
        "岗位职责：负责平台招商系统研发、服务端能力建设和数据库设计。",
        schema,
    )
    assert direction == "后端开发"


def test_infer_job_direction_accepts_hardware_rd_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "功率硬件工程师",
        "岗位职责：负责功率硬件设计、板级调试和硬件验证。",
        schema,
    )
    assert direction == "硬件研发"


def test_infer_job_direction_accepts_network_and_infra_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "网络规划专家",
        "岗位职责：负责网络规划、基础架构设计与网络容量建设。",
        schema,
    )
    assert direction == "网络与基础设施"


def test_infer_job_direction_accepts_ai_infra_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "AI Infra研发工程师",
        "岗位职责：负责机器学习平台、训练平台和推理平台建设。",
        schema,
    )
    assert direction == "AI Infra"


def test_infer_job_direction_accepts_hpc_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "高性能计算研发工程师",
        "岗位职责：负责高性能计算集群、分布式计算和 GPU 集群优化。",
        schema,
    )
    assert direction == "高性能计算"


def test_infer_job_direction_accepts_autonomous_driving_software_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "智驾系统架构工程师",
        "岗位职责：负责智驾系统架构设计、软件模块集成和功能开发。",
        schema,
    )
    assert direction == "汽车软件/智驾研发"


def test_infer_job_direction_accepts_logistics_sre_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "秒送物流SRE",
        "岗位职责：负责系统稳定性、发布巡检、监控告警和自动化运维平台建设。",
        schema,
    )
    assert direction == "运维开发"


def test_infer_job_direction_accepts_software_quality_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "开发质量工程师（软件）",
        "岗位职责：负责软件质量保障、测试流程建设和自动化验证。",
        schema,
    )
    assert direction == "测试开发"


def test_infer_job_direction_accepts_architect_post_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "资深架构师岗（电商业务）",
        "岗位职责：负责电商服务端架构设计、数据库治理和系统演进。",
        schema,
    )
    assert direction == "后端开发"


def test_infer_job_direction_accepts_server_dev_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "服务器开发工程师（UGC业务）",
        "岗位职责：负责游戏服务器端架构设计、开发和性能优化。",
        schema,
    )
    assert direction == "后端开发"


def test_infer_job_direction_accepts_network_software_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "交换机软件研发工程师",
        "岗位职责：负责交换机软件设计、开发、测试和维护工作。",
        schema,
    )
    assert direction == "网络与基础设施"


def test_infer_job_direction_accepts_ops_dev_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "运营开发工程师-EdgeOne",
        "岗位职责：负责边缘云健康探测系统架构设计、核心功能开发与性能优化。",
        schema,
    )
    assert direction == "运维开发"


def test_infer_job_direction_accepts_compiler_optimization_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "AI编译优化工程师(北京/上海/深圳)",
        "岗位职责：负责高性能图和算子编译器研发与生成算子优化。",
        schema,
    )
    assert direction == "高性能计算"


def test_infer_job_direction_rejects_business_project_manager_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "业务项目经理（保险运营）",
        "岗位职责：负责项目推进、跨团队协同和业务交付。",
        schema,
    )
    assert direction == ""


def test_infer_job_direction_rejects_platform_operations_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "平台运营岗",
        "岗位职责：负责平台业务运营、活动策划和商家协同。",
        schema,
    )
    assert direction == ""
