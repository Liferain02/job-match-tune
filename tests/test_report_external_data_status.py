from unittest.mock import patch

from jobmatch_tune.eval.report_external_data_status import build_report, describe_manifest


def test_describe_manifest_counts_existing_and_missing(tmp_path):
    existing = tmp_path / "exists.jsonl"
    existing.write_text("", encoding="utf-8")
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        "sources:\n"
        f"  - name: a\n    path: {existing}\n"
        "  - name: b\n    path: /tmp/not-found.jsonl\n",
        encoding="utf-8",
    )
    report = describe_manifest("demo", manifest)
    assert report["total_sources"] == 2
    assert report["existing_sources"] == 1
    assert report["missing_sources"] == 1


def test_describe_manifest_supports_public_job_local_path(tmp_path):
    existing = tmp_path / "jobs.csv"
    existing.write_text("title,description\n", encoding="utf-8")
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        "sources:\n" f"  - name: jobs\n    local_path: {existing}\n",
        encoding="utf-8",
    )
    report = describe_manifest("jobs", manifest)
    assert report["existing_sources"] == 1
    assert report["sources"][0]["path"] == str(existing)


def test_build_report_summarizes_ready_manifests():
    fake_sections = {
        "public_job_sources": {"total_sources": 1, "missing_sources": 0},
        "public_resume_sources": {"total_sources": 1, "missing_sources": 1},
        "public_match_sources": {"total_sources": 1, "missing_sources": 0},
    }
    with patch("jobmatch_tune.eval.report_external_data_status.describe_manifest") as mocked:
        mocked.side_effect = lambda name, path: {
            "manifest": name,
            "manifest_path": str(path),
            **fake_sections[name],
            "sources": [],
        }
        report = build_report()
    assert report["summary"]["all_manifests_ready"] is False
    assert "public_resume_sources" in report["summary"]["not_ready_manifests"]
