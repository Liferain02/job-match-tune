from jobmatch_tune.eval.report_jd_strict_tech_candidates import build_report, is_tech_like_title


def test_is_tech_like_title_detects_engineer_title():
    assert is_tech_like_title("服务器开发工程师") is True
    assert is_tech_like_title("客户经理") is False


def test_build_report_filters_non_tech_rows():
    rows = [
        {
            "id": "1",
            "source": "careers.tencent.com",
            "language": "",
            "job_title": "服务器开发工程师",
            "clean_text": "岗位职责：负责服务端开发",
            "sections": {"responsibilities": "负责服务端开发"},
            "labels": {},
            "sft_ready": True,
        },
        {
            "id": "2",
            "source": "careers.tencent.com",
            "language": "",
            "job_title": "客户经理",
            "clean_text": "岗位职责：负责客户拓展",
            "sections": {"responsibilities": "负责客户拓展"},
            "labels": {},
            "sft_ready": True,
        },
    ]
    report = build_report(rows)
    assert report["total_tech_like_rejected"] == 1
    assert report["top_titles"][0]["name"] == "服务器开发工程师"
