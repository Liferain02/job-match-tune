from jobmatch_tune.eval.report_pool_profiles import build_report


def test_build_report_returns_three_sections():
    report = build_report()
    assert set(report.keys()) == {"jd", "resume", "match"}


def test_missing_files_are_treated_as_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = build_report()
    assert report["jd"]["count"] == 0
    assert report["resume"]["count"] == 0
    assert report["match"]["count"] == 0
