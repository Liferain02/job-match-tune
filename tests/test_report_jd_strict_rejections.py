from pathlib import Path

from jobmatch_tune.eval.report_jd_strict_rejections import build_samples, classify_rejection, summarize_rows


def _base_row() -> dict:
    return {
        "id": "row_1",
        "source": "careers.tencent.com",
        "language": "",
        "job_title": "大模型推理框架研发工程师",
        "clean_text": "岗位职责：负责推理框架研发与性能优化\n任职要求：本科及以上，三年以上工作经验，熟悉 C++ 与分布式系统",
        "sections": {"responsibilities": "负责推理框架研发与性能优化", "requirements": "本科及以上，三年以上工作经验，熟悉 C++ 与分布式系统"},
        "labels": {"岗位方向": "后端开发", "学历要求": "本科", "经验要求": "三年以上工作经验", "必备技能": ["C++"]},
        "sft_ready": True,
    }


def test_classify_rejection_detects_missing_title_signal():
    row = _base_row()
    row["job_title"] = "奇怪的技术岗位名"
    assert classify_rejection(row) == "missing_title_signal"


def test_summarize_rows_counts_reasons():
    rows = [_base_row()]
    rows[0]["job_title"] = "奇怪的技术岗位名"
    report = summarize_rows(rows)
    assert report["total_rejected"] == 1
    assert report["top_reasons"][0]["name"] == "missing_title_signal"


def test_build_samples_outputs_reason_and_preview():
    row = _base_row()
    row["job_title"] = "奇怪的技术岗位名"
    samples = build_samples([row], per_reason=2, seed=42)
    assert len(samples) == 1
    assert samples[0]["reason"] == "missing_title_signal"
    assert "clean_text_preview" in samples[0]


def test_classify_rejection_accepts_backfilled_direction():
    row = {
        "id": "row_backfilled",
        "source": "careers.tencent.com",
        "language": "",
        "job_title": "运营开发工程师-EdgeOne",
        "clean_text": "岗位职责：负责边缘云健康探测系统的架构设计与核心功能开发\n任职要求：本科及以上，三年以上工作经验",
        "sections": {"responsibilities": "负责边缘云健康探测系统的架构设计与核心功能开发", "requirements": "本科及以上，三年以上工作经验"},
        "labels": {"岗位方向": "", "学历要求": "本科", "经验要求": "三年以上工作经验", "必备技能": []},
        "sft_ready": True,
    }
    assert classify_rejection(row) == "accepted"


def test_classify_rejection_accepts_short_tencent_tech_row():
    row = {
        "id": "row_short_tencent",
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
    assert classify_rejection(row) == "accepted"
