from jobmatch_tune.eval.audit_public_match_data import compute_report


def test_compute_public_match_report():
    rows = [
        {
            "source_type": "public_pair",
            "jd_text": "job description",
            "resume_text": "resume text",
            "label": {"raw_label": "fit", "raw_score": 0.9},
            "meta": {
                "source_name": "match_en",
                "language": "en",
                "license_status": "confirmed",
                "intended_usage": "training",
                "provenance_status": "human_annotated",
            },
        }
    ]
    report = compute_report(rows)
    assert report["total_rows"] == 1
    assert report["score_coverage"] == 1.0
    assert report["raw_label_distribution_top20"][0][0] == "fit"
    assert report["training_eligible_rows"] == 1
    assert report["training_ready"] is True


def test_compute_public_match_report_blocks_unlicensed_rows_and_counts_duplicates():
    row = {
        "source_type": "public_pair",
        "jd_text": "job description",
        "resume_text": "resume text",
        "label": {"raw_label": "fit", "raw_score": ""},
        "meta": {
            "source_name": "unknown",
            "language": "en",
            "license_status": "unconfirmed",
            "intended_usage": "audit_only",
            "provenance_status": "undocumented",
        },
    }
    report = compute_report([row, row])
    assert report["training_eligible_rows"] == 0
    assert report["training_ready"] is False
    assert report["duplicate_pair_rows"] == 1
