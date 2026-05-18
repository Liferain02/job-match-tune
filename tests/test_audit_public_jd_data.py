from jobmatch_tune.eval.audit_public_jd_data import compute_report


def test_compute_public_jd_report():
    rows = [
        {
            "source": "github_workaggregation_test",
            "job_title": "后端开发工程师",
            "salary": "20-30K",
            "raw_text": "岗位名称：后端开发工程师\n薪资范围：20-30K\n经验要求：3年\n学历要求：本科",
            "meta": {"language": "zh", "sft_ready": False},
        }
    ]
    report = compute_report(rows)
    assert report["total_rows"] == 1
    assert report["salary_coverage"] == 1.0
    assert report["education_coverage"] == 1.0
