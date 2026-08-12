import json
from pathlib import Path

from jobmatch_tune.eval.report_resume_sft_profile import build_resume_sft_profile


def test_resume_sft_profile_counts_groups_and_variants(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    rows = [
        {"id": "resume_1_original", "source_group": "resume_1"},
        {"id": "resume_1_bullets", "source_group": "resume_1"},
        {"id": "resume_2_original", "source_group": "resume_2"},
    ]
    train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    report = build_resume_sft_profile([str(train)])

    assert report["total"] == 3
    assert report["unique_source_groups"] == 2
    assert report["variant_counts"] == {"original": 2, "bullets": 1}
    assert report["expansion_ratio"] == 1.5
    assert report["source_group_counts_by_category"] == {"resume": 2}
    assert report["split_unique_source_groups"] == {"train": 2}
