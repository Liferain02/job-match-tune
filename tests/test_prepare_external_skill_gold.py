from __future__ import annotations

from pathlib import Path

from jobmatch_tune.dataset.prepare_external_skill_gold import (
    bio_spans,
    convert_chinese_rows,
    isolate_splits,
    parse_chinese_tagged_sentence,
)


def test_parse_chinese_tagged_sentence_restores_offsets() -> None:
    text = "熟悉Python，具备沟通能力"
    spans = parse_chinese_tagged_sentence(
        text,
        "熟悉@@Python##K，具备@@沟通能力##T",
    )

    assert spans == [
        {"start": 2, "end": 8, "label": "K", "text": "Python"},
        {"start": 11, "end": 15, "label": "T", "text": "沟通能力"},
    ]


def test_bio_spans_extracts_contiguous_tokens() -> None:
    tokens = ["data", "analysis", "and", "Python"]
    starts = [0, 5, 14, 18]

    spans = bio_spans(tokens, ["B", "I", "O", "B"], "knowledge", starts)

    assert spans == [
        {"start": 0, "end": 13, "label": "knowledge", "text": "data analysis"},
        {"start": 18, "end": 24, "label": "knowledge", "text": "Python"},
    ]


def test_convert_chinese_rows_keeps_provenance_and_labels() -> None:
    rows = [
        {
            "id": 1,
            "input": "使用Python开发服务",
            "output": "@@使用Python开发服务##S",
            "meta": {"id": "job1-s0", "job_id": "job1", "source_domain": "技术招聘"},
        }
    ]
    source_cfg = {
        "language": "zh",
        "annotation": "human",
        "license_status": "unconfirmed",
        "intended_usage": "internal_evaluation_only",
    }

    converted, errors = convert_chinese_rows(
        rows,
        dataset_name="demo",
        split="test",
        source_cfg=source_cfg,
    )

    assert errors == []
    assert converted[0]["source_group"] == "demo:job1"
    assert converted[0]["label"]["专业技能"] == ["使用Python开发服务"]
    assert converted[0]["meta"]["license_status"] == "unconfirmed"


def test_isolate_splits_prioritizes_test_and_removes_leakage() -> None:
    def row(row_id: str, group: str, text: str) -> dict:
        return {"id": row_id, "source_group": group, "text": text, "spans": []}

    rows_by_split = {
        "train": [row("tr1", "g1", "train only"), row("tr2", "g2", "same text")],
        "dev": [row("d1", "g3", "dev only"), row("d2", "g4", "same text")],
        "test": [row("t1", "g1", "test only"), row("t2", "g5", "same text")],
    }

    kept, stats = isolate_splits(rows_by_split)

    assert [item["id"] for item in kept["test"]] == ["t1", "t2"]
    assert [item["id"] for item in kept["dev"]] == ["d1"]
    assert kept["train"] == []
    assert stats == {
        "cross_split_text_dropped": 2,
        "cross_split_source_group_dropped": 1,
    }


def test_manifest_is_valid_yaml_shape() -> None:
    # A small guard against accidentally editing the checked-in manifest into JSON text.
    manifest = Path("configs/external_skill_gold_sources.yaml").read_text(encoding="utf-8")
    assert "chinese_skillspan:" in manifest
    assert "skillspan_en:" in manifest
    assert not manifest.lstrip().startswith("{")
