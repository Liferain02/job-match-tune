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
    assert report["real_resume_source_groups"] == 0
    assert report["supports_real_resume_quality_claim"] is False


def test_resume_sft_profile_counts_explicit_real_anonymized_sources(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    rows = [
        {
            "id": "real_1_original",
            "source_group": "real_1",
            "meta": {"data_origin": "public_real_anonymized"},
        },
        {
            "id": "curated_1_original",
            "source_group": "curated_1",
            "meta": {"data_origin": "curated_fictional"},
        },
    ]
    train.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = build_resume_sft_profile([str(train)])

    assert report["real_resume_source_groups"] == 1
    assert report["source_group_counts_by_category"] == {
        "curated_fictional": 1,
        "public_real_anonymized": 1,
    }
