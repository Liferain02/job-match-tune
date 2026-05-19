from pathlib import Path

from jobmatch_tune.eval.compare_jd_sft_tracks import build_report
from jobmatch_tune.utils.io import write_jsonl


def test_build_report_compares_three_tracks(tmp_path: Path):
    strict_dir = tmp_path / "strict"
    strict_plus_dir = tmp_path / "strict_plus"
    bootstrap_dir = tmp_path / "bootstrap"
    strict_dir.mkdir()
    strict_plus_dir.mkdir()
    bootstrap_dir.mkdir()
    row = {
        "id": "demo_jd_parse",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {
                "role": "assistant",
                "content": '{"岗位方向":"后端开发","核心职责":["a"],"必备技能":["Java"],"学历要求":"本科","经验要求":"3年"}',
            },
        ],
    }
    write_jsonl(strict_dir / "train.jsonl", [row])
    write_jsonl(strict_plus_dir / "train.jsonl", [row, row])
    write_jsonl(bootstrap_dir / "train.jsonl", [row, row])
    report = build_report(strict_dir, bootstrap_dir, strict_plus_dir)
    assert report["strict"]["total_samples"] == 1
    assert report["bootstrap"]["total_samples"] == 2
    assert report["strict_plus"]["total_samples"] == 2
    assert report["delta_strict_to_bootstrap"]["total_samples"] == 1
    assert report["delta_strict_to_strict_plus"]["total_samples"] == 1
    assert report["delta_strict_plus_to_bootstrap"]["total_samples"] == 0
