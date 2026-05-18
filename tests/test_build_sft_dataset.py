from __future__ import annotations

from jobmatch_tune.dataset.build_sft_dataset import (
    build_jd_parse_sample,
    collect_sft_rows,
    is_high_trust_strong_row,
    is_high_confidence_weak_tech_row,
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
