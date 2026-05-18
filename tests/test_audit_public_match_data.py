from jobmatch_tune.eval.audit_public_match_data import compute_report


def test_compute_public_match_report():
    rows = [
        {
            "source_type": "public_pair",
            "jd_text": "job description",
            "resume_text": "resume text",
            "label": {"raw_label": "fit", "raw_score": 0.9},
            "meta": {"source_name": "match_en", "language": "en"},
        }
    ]
    report = compute_report(rows)
    assert report["total_rows"] == 1
    assert report["score_coverage"] == 1.0
    assert report["raw_label_distribution_top20"][0][0] == "fit"
