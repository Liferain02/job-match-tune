from jobmatch_tune.dataset.build_jd_train_pool_repairable import build_rows, is_repairable_row


def test_is_repairable_row_accepts_missing_direction_with_strong_text() -> None:
    row = {
        "id": "repair_1",
        "source": "careers.tencent.com",
        "language": "",
        "job_title": "腾讯会议-Android研发工程师",
        "clean_text": "岗位职责：负责会议 Android 客户端功能研发、性能优化、稳定性建设、多端协同能力落地、弱网优化、崩溃治理和质量保障。\n任职要求：本科及以上，三年以上工作经验，熟悉 Android 开发、性能分析、工程化实践、客户端架构设计和线上问题定位。",
        "sections": {"responsibilities": "负责会议 Android 客户端功能研发、性能优化、稳定性建设、多端协同能力落地、弱网优化、崩溃治理和质量保障。", "requirements": "本科及以上，三年以上工作经验，熟悉 Android 开发、性能分析、工程化实践、客户端架构设计和线上问题定位。"},
        "labels": {"岗位方向": "", "学历要求": "本科", "经验要求": "三年以上工作经验", "必备技能": ["Android"]},
        "sft_ready": True,
    }
    assert is_repairable_row(row) is True


def test_is_repairable_row_rejects_non_tech_row() -> None:
    row = {
        "id": "repair_2",
        "source": "zhaopin.jd.com",
        "language": "zh",
        "job_title": "选址开发",
        "clean_text": "岗位职责：负责新门店的选址与建设。",
        "sections": {"responsibilities": "负责新门店的选址与建设。", "requirements": "具备零售行业经验。"},
        "labels": {"岗位方向": "", "学历要求": "", "经验要求": "", "必备技能": []},
        "sft_ready": True,
    }
    assert is_repairable_row(row) is False


def test_build_rows_deduplicates() -> None:
    row = {
        "id": "repair_3",
        "source": "careers.tencent.com",
        "language": "",
        "job_title": "元宝-Android开发工程师",
        "company": "腾讯",
        "location": "深圳",
        "clean_text": "岗位职责：负责 Android 客户端功能研发、性能优化、崩溃治理、版本交付、监控建设、核心链路体验提升和端侧稳定性建设。\n任职要求：本科及以上，三年以上工作经验，熟悉 Android 工程架构、性能调优、组件化设计、线上问题定位和多端协同方案设计。",
        "sections": {"responsibilities": "负责 Android 客户端功能研发、性能优化、崩溃治理、版本交付、监控建设、核心链路体验提升和端侧稳定性建设。", "requirements": "本科及以上，三年以上工作经验，熟悉 Android 工程架构、性能调优、组件化设计、线上问题定位和多端协同方案设计。"},
        "labels": {"岗位方向": "", "学历要求": "本科", "经验要求": "三年以上工作经验", "必备技能": ["Android"]},
        "sft_ready": True,
    }
    built = build_rows([row, dict(row)])
    assert len(built) == 1
    assert built[0]["meta"]["pool_origin"] == "repairable_candidate"
