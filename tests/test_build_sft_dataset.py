from __future__ import annotations

from jobmatch_tune.dataset.build_sft_dataset import (
    build_jd_parse_sample,
    collect_sft_rows,
    get_effective_direction,
    is_high_trust_strong_row,
    is_high_confidence_weak_tech_row,
    is_tencent_short_tech_row,
)


def test_build_jd_parse_sample_uses_headers() -> None:
    row = {
        "id": "demo",
        "job_title": "后端开发工程师",
        "company": "示例公司",
        "location": "北京",
        "clean_text": "岗位职责：负责服务开发\n任职要求：熟悉Python",
        "sections": {"responsibilities": "负责服务开发", "bonus": "熟悉大模型"},
        "labels": {"岗位方向": "后端开发", "必备技能": ["Python"], "经验要求": "3-5年", "学历要求": "本科"},
    }
    sample = build_jd_parse_sample(row)
    assert sample["id"] == "demo_jd_parse"
    user_text = sample["messages"][1]["content"]
    assert "岗位名称：后端开发工程师" in user_text
    assert "公司：示例公司" in user_text
    assert "工作地点：北京" in user_text


def test_is_high_confidence_weak_tech_row_for_education_dataset() -> None:
    row = {
        "id": "weak_demo",
        "source": "hf_job_educational_train_2026_05_17",
        "language": "zh",
        "job_title": "后端开发工程师",
        "clean_text": (
            "岗位名称：后端开发工程师\n"
            "岗位职责：负责服务开发、接口设计、性能优化、数据库治理与接口稳定性建设，推动系统高可用落地。\n"
            "任职要求：本科及以上，3年以上 Python / Java 开发经验，熟悉分布式系统设计、数据库优化、缓存设计、消息队列和接口治理。\n"
            "技能要求：熟悉 Linux、SQL，具备日志分析、监控告警、线上故障排查、技术文档编写、自动化发布和容量评估能力。"
        ),
            "sections": {
                "responsibilities": "负责服务开发、接口设计和性能优化。",
                "requirements": "本科及以上，3年以上 Python / Java 开发经验。",
            },
            "labels": {"必备技能": ["Python", "Java", "Linux", "SQL"], "学历要求": "本科", "经验要求": "3年以上 Python / Java 开发经验"},
            "sft_ready": False,
        }
    assert is_high_confidence_weak_tech_row(row) is True


def test_collect_sft_rows_fills_to_target_with_weak_tech() -> None:
    strong_row = {
        "id": "strong_1",
        "source": "zhaopin.jd.com",
        "language": "zh",
        "job_title": "前端开发工程师",
        "clean_text": "岗位职责：负责前端开发\n任职要求：本科及以上，熟悉 Vue",
        "sections": {"responsibilities": "负责前端开发", "requirements": "本科及以上，熟悉 Vue"},
        "labels": {"岗位方向": "前端开发", "必备技能": ["Vue"]},
        "sft_ready": True,
    }
    weak_row = {
        "id": "weak_1",
        "source": "hf_job_educational_train_2026_05_17",
        "language": "zh",
        "job_title": "算法工程师",
        "clean_text": (
            "岗位职责：负责模型训练、推理优化、实验平台建设和数据分析链路治理，持续提升模型效果与线上稳定性。\n"
            "任职要求：硕士及以上，2年以上 Python 开发经验，熟悉 Linux 环境、数据处理链路、实验分析方法、评测流程和模型调优。\n"
            "技能要求：熟悉 Linux、SQL，能够编写分析脚本、维护训练实验文档、支持评测平台日常治理，并参与数据质量回溯、实验复盘和线上效果监控。"
        ),
        "sections": {"responsibilities": "负责模型训练与推理优化。", "requirements": "硕士及以上，2年以上 Python 开发经验。"},
        "labels": {"岗位方向": "算法工程", "必备技能": ["Python", "Linux", "SQL"], "学历要求": "硕士", "经验要求": "2年以上 Python 开发经验"},
        "sft_ready": False,
    }
    rows = collect_sft_rows(
        [strong_row, weak_row],
        include_weak_tech=True,
        target_total=2,
        seed=42,
        quality_profile="expanded",
    )
    assert [row["id"] for row in rows] == ["strong_1", "weak_1"]


def test_is_high_trust_strong_row_requires_trusted_source_and_fields() -> None:
    row = {
        "id": "trusted_1",
        "source": "zhaopin.jd.com",
        "language": "zh",
        "job_title": "后端开发工程师",
        "clean_text": "岗位职责：负责服务开发\n任职要求：本科及以上，熟悉 Python\n技能要求：Python",
        "sections": {"responsibilities": "负责服务开发", "requirements": "本科及以上，熟悉 Python"},
        "labels": {"岗位方向": "后端开发", "必备技能": ["Python"], "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True

    noisy = dict(row)
    noisy["source"] = "bebee.com"
    assert is_high_trust_strong_row(noisy) is False

    non_target = dict(row)
    non_target["job_title"] = "机械结构工程师"
    assert is_high_trust_strong_row(non_target) is False


def test_is_high_trust_strong_row_accepts_product_manager() -> None:
    row = {
        "id": "trusted_pm",
        "source": "talent.baidu.com",
        "language": "zh",
        "job_title": "大模型产品经理",
        "clean_text": "岗位职责：负责大模型产品设计\n任职要求：本科及以上，具备产品设计和数据分析能力",
        "sections": {"responsibilities": "负责大模型产品设计", "requirements": "本科及以上，具备产品设计和数据分析能力"},
        "labels": {"岗位方向": "产品经理", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_security_engineer() -> None:
    row = {
        "id": "trusted_sec",
        "source": "moka_threatbook",
        "language": "zh",
        "job_title": "安全工程师",
        "clean_text": "岗位职责：负责漏洞分析与安全攻防\n任职要求：本科及以上，具备安全研发经验",
        "sections": {"responsibilities": "负责漏洞分析与安全攻防", "requirements": "本科及以上，具备安全研发经验"},
        "labels": {"岗位方向": "安全工程", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_security_ops_exception() -> None:
    row = {
        "id": "trusted_sec_ops",
        "source": "zhaopin.jd.com",
        "language": "zh",
        "job_title": "大模型安全运营",
        "clean_text": "岗位职责：负责大模型安全治理与告警处置\n任职要求：本科及以上，具备安全运营经验",
        "sections": {"responsibilities": "负责大模型安全治理与告警处置", "requirements": "本科及以上，具备安全运营经验"},
        "labels": {"岗位方向": "安全工程", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_backend_rd_exception() -> None:
    row = {
        "id": "trusted_rd_post",
        "source": "zhaopin.jd.com",
        "language": "zh",
        "job_title": "B端招商入驻研发岗",
        "clean_text": "岗位职责：负责平台招商系统研发与接口开发\n任职要求：本科及以上，熟悉 Java 与数据库",
        "sections": {"responsibilities": "负责平台招商系统研发与接口开发", "requirements": "本科及以上，熟悉 Java 与数据库"},
        "labels": {"岗位方向": "后端开发", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_hardware_rd() -> None:
    row = {
        "id": "trusted_hw",
        "source": "moka_se",
        "language": "zh",
        "job_title": "功率硬件工程师",
        "clean_text": "岗位职责：负责功率硬件设计、板级调试和硬件验证\n任职要求：本科及以上，具备硬件开发经验",
        "sections": {"responsibilities": "负责功率硬件设计、板级调试和硬件验证", "requirements": "本科及以上，具备硬件开发经验"},
        "labels": {"岗位方向": "硬件研发", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_network_and_infra() -> None:
    row = {
        "id": "trusted_netinfra",
        "source": "zhaopin.jd.com",
        "language": "zh",
        "job_title": "网络规划专家",
        "clean_text": "岗位职责：负责网络规划、基础架构设计和容量评估\n任职要求：本科及以上，具备网络架构经验",
        "sections": {"responsibilities": "负责网络规划、基础架构设计和容量评估", "requirements": "本科及以上，具备网络架构经验"},
        "labels": {"岗位方向": "网络与基础设施", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_ai_infra() -> None:
    row = {
        "id": "trusted_aiinfra",
        "source": "careers.tencent.com",
        "language": "zh",
        "job_title": "AI Infra研发工程师",
        "clean_text": "岗位职责：负责机器学习平台、训练平台和推理平台建设\n任职要求：本科及以上，具备平台研发经验",
        "sections": {"responsibilities": "负责机器学习平台、训练平台和推理平台建设", "requirements": "本科及以上，具备平台研发经验"},
        "labels": {"岗位方向": "AI Infra", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_tencent_empty_language() -> None:
    row = {
        "id": "trusted_tencent_empty_lang",
        "source": "careers.tencent.com",
        "language": "",
        "job_title": "大模型推理框架研发工程师",
        "clean_text": "岗位职责：负责大模型推理框架研发与性能优化\n任职要求：本科及以上，三年以上工作经验，熟悉 C++ 与分布式系统",
        "sections": {"responsibilities": "负责大模型推理框架研发与性能优化", "requirements": "本科及以上，三年以上工作经验，熟悉 C++ 与分布式系统"},
        "labels": {"岗位方向": "后端开发", "学历要求": "本科", "经验要求": "三年以上工作经验"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_xiaomi_source() -> None:
    row = {
        "id": "trusted_xiaomi_android",
        "source": "hr.xiaomi.com",
        "language": "zh",
        "job_title": "Android Multimedia系统工程师（小米电视）",
        "clean_text": "岗位职责：负责 Android 系统框架开发与性能优化\n任职要求：本科及以上，3年以上 Android 开发经验，熟悉 Java",
        "sections": {"responsibilities": "负责 Android 系统框架开发与性能优化", "requirements": "本科及以上，3年以上 Android 开发经验，熟悉 Java"},
        "labels": {"岗位方向": "客户端开发", "学历要求": "本科", "经验要求": "3年以上 Android 开发经验", "必备技能": ["Java"]},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_hpc() -> None:
    row = {
        "id": "trusted_hpc",
        "source": "talent.baidu.com",
        "language": "zh",
        "job_title": "高性能计算研发工程师",
        "clean_text": "岗位职责：负责高性能计算集群研发与 GPU 资源调优\n任职要求：本科及以上，具备分布式计算经验",
        "sections": {"responsibilities": "负责高性能计算集群研发与 GPU 资源调优", "requirements": "本科及以上，具备分布式计算经验"},
        "labels": {"岗位方向": "高性能计算", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_autonomous_driving_software() -> None:
    row = {
        "id": "trusted_autodrive",
        "source": "moka_voyah",
        "language": "zh",
        "job_title": "智驾系统架构工程师",
        "clean_text": "岗位职责：负责智驾系统架构设计、软件模块集成和功能开发\n任职要求：本科及以上，具备自动驾驶软件开发经验",
        "sections": {"responsibilities": "负责智驾系统架构设计、软件模块集成和功能开发", "requirements": "本科及以上，具备自动驾驶软件开发经验"},
        "labels": {"岗位方向": "汽车软件/智驾研发", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_sre_with_logistics_keyword() -> None:
    row = {
        "id": "trusted_sre_logistics",
        "source": "zhaopin.jd.com",
        "language": "zh",
        "job_title": "秒送物流SRE",
        "clean_text": "岗位职责：负责系统稳定性、监控告警和容量规划\n任职要求：本科及以上，具备 SRE 或运维开发经验",
        "sections": {"responsibilities": "负责系统稳定性、监控告警和容量规划", "requirements": "本科及以上，具备 SRE 或运维开发经验"},
        "labels": {"岗位方向": "运维开发", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_rd_title_without_engineer_keyword() -> None:
    row = {
        "id": "trusted_rd",
        "source": "talent.baidu.com",
        "language": "zh",
        "job_title": "OLAP引擎研发",
        "clean_text": "岗位职责：负责 OLAP 引擎研发与性能优化\n任职要求：本科及以上，熟悉数据库与分布式系统",
        "sections": {"responsibilities": "负责 OLAP 引擎研发与性能优化", "requirements": "本科及以上，熟悉数据库与分布式系统"},
        "labels": {"岗位方向": "后端开发", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_model_lead_title() -> None:
    row = {
        "id": "trusted_model_lead",
        "source": "careers.tencent.com",
        "language": "",
        "job_title": "腾讯视频-视频生成模型负责人-(深圳)(杭州)",
        "clean_text": "岗位职责：负责视频生成模型研发与架构设计\n任职要求：本科及以上，三年以上生成模型或多模态算法经验",
        "sections": {"responsibilities": "负责视频生成模型研发与架构设计", "requirements": "本科及以上，三年以上生成模型或多模态算法经验"},
        "labels": {"岗位方向": "算法工程", "学历要求": "本科", "经验要求": "三年以上生成模型或多模态算法经验"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_ops_dev_title() -> None:
    row = {
        "id": "trusted_ops_dev",
        "source": "careers.tencent.com",
        "language": "",
        "job_title": "运营开发工程师-EdgeOne",
        "clean_text": "岗位职责：负责边缘云探测系统架构设计、核心功能开发与监控可视化建设\n任职要求：本科及以上，两年以上 SRE 或运维开发经验",
        "sections": {"responsibilities": "负责边缘云探测系统架构设计、核心功能开发与监控可视化建设", "requirements": "本科及以上，两年以上 SRE 或运维开发经验"},
        "labels": {"岗位方向": "运维开发", "学历要求": "本科", "经验要求": "两年以上 SRE 或运维开发经验"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_get_effective_direction_backfills_missing_direction() -> None:
    row = {
        "id": "trusted_backfill_dir",
        "source": "careers.tencent.com",
        "language": "zh",
        "job_title": "运营开发工程师-EdgeOne",
        "clean_text": "岗位职责：负责边缘云健康探测系统架构设计、核心功能开发与性能优化。",
        "sections": {"responsibilities": "负责边缘云系统开发", "requirements": "本科及以上，三年以上工作经验"},
        "labels": {"岗位方向": "", "学历要求": "本科", "经验要求": "三年以上工作经验"},
        "sft_ready": True,
    }
    assert get_effective_direction(row) == "运维开发"
    assert is_high_trust_strong_row(row) is True


def test_build_jd_parse_sample_uses_backfilled_direction() -> None:
    row = {
        "id": "trusted_backfilled_sample",
        "job_title": "云网络高级开发工程师",
        "company": "示例公司",
        "location": "北京",
        "clean_text": "岗位职责：负责云网关数据面的软件设计和开发\n任职要求：本科及以上，三年以上工作经验",
        "sections": {"responsibilities": "负责云网关数据面的软件设计和开发", "requirements": "本科及以上，三年以上工作经验"},
        "labels": {"岗位方向": "", "必备技能": [], "经验要求": "三年以上工作经验", "学历要求": "本科"},
    }
    sample = build_jd_parse_sample(row)
    assistant = sample["messages"][2]["content"]
    assert "网络与基础设施" in assistant


def test_is_tencent_short_tech_row_accepts_short_high_value_tencent_jd() -> None:
    row = {
        "id": "trusted_tencent_short",
        "source": "careers.tencent.com",
        "language": "",
        "job_title": "腾讯会议-Android研发工程师",
        "clean_text": (
            "岗位职责：负责腾讯会议Android客户端研发，难点攻坚以及新技术预研；\n"
            "2.负责Android端基础设施和技术方案设计，完成高质量交付和版本发布；\n"
            "3.负责腾讯会议C++跨平台逻辑开发与维护，持续推进端侧稳定性治理与工程效率优化。\n"
            "经验要求：三年以上工作经验"
        ),
        "sections": {
            "responsibilities": (
                "1.负责腾讯会议Android客户端研发，难点攻坚以及新技术预研；\n"
                "2.负责Android端基础设施和技术方案设计，完成高质量交付和版本发布；\n"
                "3.负责腾讯会议C++跨平台逻辑开发与维护，持续推进端侧稳定性治理与工程效率优化。"
            ),
            "requirements": "",
        },
        "labels": {"岗位方向": "", "学历要求": "", "经验要求": "三年以上工作经验", "必备技能": ["Android", "C++"]},
        "meta": {"category": "技术"},
        "sft_ready": True,
    }
    assert is_tencent_short_tech_row(row) is True
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_accepts_fallback_structure_from_high_trust_source() -> None:
    row = {
        "id": "trusted_fallback",
        "source": "careers.tencent.com",
        "language": "zh",
        "job_title": "数据开发工程师",
        "clean_text": (
            "岗位名称：数据开发工程师\n"
            "工作内容：负责离线数仓建设、ETL 任务开发、指标看板支持、任务调度治理和数据质量巡检，推动核心数据链路稳定运行并持续优化性能。\n"
            "职位要求：本科及以上学历，熟悉 SQL、Spark、Hive、Airflow，具备数据平台开发经验，能够独立完成批处理任务开发、问题排查、口径梳理和跨团队协作。\n"
            "加分项：有实时数仓和 Flink 开发经验。"
        ),
        "sections": {},
        "labels": {
            "岗位方向": "数据开发",
            "必备技能": ["SQL", "Spark", "Hive", "Airflow"],
            "学历要求": "本科及以上",
        },
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is True


def test_is_high_trust_strong_row_rejects_pr_role_even_if_model_predicted_tech() -> None:
    row = {
        "id": "trusted_pr",
        "source": "talent.baidu.com",
        "language": "zh",
        "job_title": "百度集团公关（支持智能驾驶事业群）",
        "clean_text": "岗位职责：负责公关传播与品牌合作\n任职要求：本科及以上，具备传播经验",
        "sections": {"responsibilities": "负责公关传播与品牌合作", "requirements": "本科及以上，具备传播经验"},
        "labels": {"岗位方向": "AI应用开发", "学历要求": "本科"},
        "sft_ready": True,
    }
    assert is_high_trust_strong_row(row) is False
