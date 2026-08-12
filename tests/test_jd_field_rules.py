from __future__ import annotations

import yaml

from jobmatch_tune.preprocess.jd_field_rules import (
    extract_education_requirement,
    extract_experience_requirement,
    extract_skills_from_text,
    infer_job_direction,
)


def _schema() -> dict:
    with open("configs/label_schema.yaml", "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def test_extract_experience_accepts_common_chinese_spacing_and_prefixes() -> None:
    assert extract_experience_requirement("任职要求：具备 3 年以上 Android 开发经验") == (
        "具备 3 年以上 Android 开发经验"
    )
    assert extract_experience_requirement("至少两年 iOS 开发经验") == "至少两年 iOS 开发经验"
    assert extract_experience_requirement("工作年限3～5年") == "3～5年"


def test_extract_experience_rejects_graduation_date_as_year_range() -> None:
    text = "任职资格：1-2024年10月1日至2025年9月30日期间毕业，本科及以上学历"
    assert extract_experience_requirement(text) == ""


def test_extract_education_covers_degree_and_postgraduate_phrases() -> None:
    assert extract_education_requirement("相关专业研究生及以上学历") == "研究生及以上学历"
    assert extract_education_requirement("相关领域的学士及以上学位") == "学士及以上学位"
    assert extract_education_requirement("专科及以上学历，五年以上经验") == "专科及以上学历"
    assert extract_education_requirement("招聘要求：博士研究生学历") == "博士研究生学历"
    assert extract_education_requirement("招聘要求：硕士研究生学历") == "硕士研究生学历"


def test_extract_education_preserves_preferred_semantics() -> None:
    assert extract_education_requirement("研究生及以上学历优先") == "研究生及以上学历优先"
    assert extract_education_requirement("硕士优先") == "硕士优先"
    assert extract_education_requirement("学士及学士以上的学历") == "学士及学士以上的学历"


def test_extract_education_uses_baseline_before_later_preference() -> None:
    assert extract_education_requirement("本科及以上学历，计算机专业，硕士优先") == "本科及以上"


def test_extract_skills_covers_common_client_game_and_hardware_tools() -> None:
    skills = extract_skills_from_text(
        "使用 JavaScript、Vue.js、Android、Unity3D、UE5、MATLAB、FPGA 和 SystemVerilog 开发",
        _schema(),
    )

    assert {"JavaScript", "Vue", "Android", "Unity", "Unreal Engine", "MATLAB", "FPGA", "Verilog"} <= set(
        skills
    )


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


def test_infer_job_direction_does_not_treat_domain_researcher_as_algorithm_role() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "微生物研究员",
        "负责菌株培养、微生物检测和实验结果记录。",
        schema,
    )
    assert direction == ""


def test_domain_researcher_does_not_use_incidental_technical_body_word() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "理化助理研究员",
        "负责实验室样品检测、仪器系统记录和设备日常运维。",
        schema,
    )
    assert direction == ""


def test_technical_researcher_can_use_explicit_algorithm_context() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "机器视觉研究员",
        "负责计算机视觉算法、深度学习模型训练和推理优化。",
        schema,
    )
    assert direction == "算法工程"


def test_security_researcher_keeps_security_direction() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "鸿蒙安全研究员",
        "负责操作系统漏洞挖掘、安全攻防和检测工具开发。",
        schema,
    )
    assert direction == "安全工程"


def test_domain_safety_researcher_is_not_network_security() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "食品安全方向研究员",
        "负责食品安全风险评估和微生物实验研究。",
        schema,
    )
    assert direction == ""


def test_world_model_researcher_keeps_algorithm_direction() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "实时全模态交互世界模型研究员",
        "负责多模态基础模型训练、推理和生成算法研发。",
        schema,
    )
    assert direction == "算法工程"


def test_generic_researcher_requires_multiple_strong_model_signals() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "高级研究员",
        "研究游戏大模型架构，负责模型训练、多模态和扩散模型方案。",
        schema,
    )
    assert direction == "算法工程"


def test_domain_researcher_is_not_rescued_by_incidental_deep_learning_reference() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "生物信息多组学研究员",
        "负责多组学实验分析，协助AI工程师构建深度学习模型。",
        schema,
    )
    assert direction == ""


def test_ai_industry_researcher_is_not_treated_as_algorithm_engineer() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "AI行业研究员",
        "跟踪生成式AI和大模型市场趋势，撰写投资研究和赛道分析报告。",
        schema,
    )
    assert direction == ""


def test_ai_technology_researcher_with_chinese_prefix_keeps_algorithm_direction() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "行业AI技术研究员",
        "负责机器学习、深度学习算法研发，使用PyTorch训练模型。",
        schema,
    )
    assert direction == "算法工程"


def test_hardware_research_and_development_title_keeps_hardware_direction() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "硬件研发工程师",
        "负责电源板原理图设计、硬件调试和EMC测试。",
        schema,
    )
    assert direction == "硬件研发"


def test_infer_job_direction_maps_explicit_product_intern_to_product_manager() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "AIGC产品实习生",
        "负责用户需求分析、产品方案和版本迭代，协同算法团队。",
        schema,
    )
    assert direction == "产品经理"


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


def test_infer_job_direction_prioritizes_algorithm_role_over_security_department() -> None:
    direction = infer_job_direction(
        "微信安全-多模态大模型高级算法工程师",
        "岗位职责：负责多模态模型训练和内容理解算法研发。",
        {"job_directions": ["算法工程", "安全工程"]},
    )
    assert direction == "算法工程"


def test_infer_job_direction_treats_network_security_as_security() -> None:
    direction = infer_job_direction(
        "网络与信息安全研究岗",
        "岗位职责：负责漏洞研究、安全攻防和风险分析。",
        {"job_directions": ["网络与基础设施", "安全工程"]},
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


def test_infer_job_direction_accepts_gameplay_dev_expert_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "游戏玩法开发专家",
        "岗位职责：负责角色控制、相机控制、NPC控制等模块的开发与相关技术预研。",
        schema,
    )
    assert direction == "后端开发"


def test_infer_job_direction_accepts_game_ai_dev_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "游戏AI开发工程师",
        "岗位职责：负责NPC、Bot与拟人AI的行为开发、感知、寻路与动作系统优化。",
        schema,
    )
    assert direction == "算法工程"


def test_infer_job_direction_accepts_conference_mobile_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "腾讯会议-Android研发工程师",
        "岗位职责：负责会议 Android 客户端功能研发与性能优化。",
        schema,
    )
    assert direction == "客户端开发"


def test_infer_job_direction_accepts_conference_sdk_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "腾讯会议-音视频引擎SDK开发工程师",
        "岗位职责：负责音视频引擎 SDK 模块设计开发与性能优化。",
        schema,
    )
    assert direction == "客户端开发"


def test_infer_job_direction_accepts_big_data_dev_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "大数据开发工程师",
        "岗位职责：负责离线数仓、ETL链路和数据平台开发。",
        schema,
    )
    assert direction == "数据开发"


def test_infer_job_direction_accepts_vehicle_control_title() -> None:
    schema = _schema()
    direction = infer_job_direction(
        "转向电控工程师",
        "岗位职责：负责转向电控系统方案设计、功能定义和联合标定开发。",
        schema,
    )
    assert direction == "汽车软件/智驾研发"


def test_extract_skills_from_text_supports_cross_domain_taxonomy() -> None:
    skills = extract_skills_from_text(
        "熟悉 Go、Linux、Kubernetes、Prometheus、CUDA、MPI、TCP/IP、BGP 和示波器。",
        _schema(),
    )
    assert skills == ["Go", "Linux", "Kubernetes", "Prometheus", "CUDA", "MPI", "TCP/IP", "BGP", "示波器"]


def test_extract_skills_from_text_uses_ascii_boundaries() -> None:
    skills = extract_skills_from_text(
        "负责 Golang 服务和 Cadence 电路设计，掌握 C 语言及 C++。",
        _schema(),
    )
    assert "Go" in skills
    assert "Cadence" in skills
    assert "C语言" in skills
    assert "C++" in skills
    assert "C" not in skills


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


def test_common_english_words_do_not_trigger_short_client_tokens() -> None:
    direction = infer_job_direction(
        "PR & Communications",
        "Support social content, daily administrative requests and reports.",
        _schema(),
    )
    assert direction == ""


def test_standalone_ue_title_still_maps_to_client_development() -> None:
    direction = infer_job_direction(
        "UE 开发工程师",
        "负责游戏客户端功能开发。",
        _schema(),
    )
    assert direction == "客户端开发"


def test_ue5_title_still_maps_to_client_development() -> None:
    direction = infer_job_direction(
        "UE5 资深 Gameplay 工程师",
        "负责战斗玩法和游戏引擎功能开发。",
        _schema(),
    )
    assert direction == "客户端开发"


def test_ue5_backend_title_stays_backend() -> None:
    direction = infer_job_direction(
        "UE5 后台开发工程师（DS 方向）",
        "负责游戏服务器和 UE5 Dedicated Server 开发。",
        _schema(),
    )
    assert direction == "后端开发"


def test_explicit_java_title_beats_incidental_frontend_body_terms() -> None:
    direction = infer_job_direction(
        "Java 开发工程师",
        "负责 Java 服务，也需要与 JavaScript、React 前端协作。",
        _schema(),
    )
    assert direction == "后端开发"


def test_android_and_unity3d_titles_stay_client_development() -> None:
    assert infer_job_direction("安卓 APP开发工程师", "负责移动应用。", _schema()) == "客户端开发"
    assert infer_job_direction("Unity3D开发工程师", "负责游戏功能。", _schema()) == "客户端开发"


def test_javascript_does_not_also_count_as_java_backend_evidence() -> None:
    direction = infer_job_direction(
        "软件工程师",
        "使用 JavaScript、TypeScript 和 React 负责 Web 页面开发。",
        _schema(),
    )
    assert direction == "前端开发"
