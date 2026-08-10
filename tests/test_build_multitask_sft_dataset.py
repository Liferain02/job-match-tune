from __future__ import annotations

import json
from pathlib import Path

from jobmatch_tune.dataset.build_multitask_sft_dataset import build_multitask_dataset


def _sample(row_id: str, task_type: str, user: str, source_group: str | None = None) -> dict:
    row = {
        "id": row_id,
        "task_type": task_type,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps({"ok": row_id}, ensure_ascii=False)},
        ],
    }
    if source_group:
        row["source_group"] = source_group
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_multitask_dataset_samples_by_task_and_tags_meta(tmp_path: Path) -> None:
    jd_train = tmp_path / "jd_train.jsonl"
    jd_valid = tmp_path / "jd_valid.jsonl"
    resume_train = tmp_path / "resume_train.jsonl"
    resume_valid = tmp_path / "resume_valid.jsonl"
    _write_jsonl(jd_train, [_sample("jd_1", "jd_parse", "jd1"), _sample("jd_2", "jd_parse", "jd2")])
    _write_jsonl(jd_valid, [_sample("jd_v", "jd_parse", "jdv")])
    _write_jsonl(
        resume_train,
        [
            _sample("resume_1", "resume_parse", "resume1"),
            _sample("resume_2", "resume_parse", "resume2"),
            _sample("resume_3", "resume_parse", "resume3"),
        ],
    )
    _write_jsonl(resume_valid, [_sample("resume_v", "resume_parse", "resumev")])

    registry = {
        "multitask_sft": {
            "train_out": str(tmp_path / "train.jsonl"),
            "valid_out": str(tmp_path / "valid.jsonl"),
            "seed": 42,
            "tasks": {
                "jd": {
                    "train_file": str(jd_train),
                    "valid_file": str(jd_valid),
                    "train_samples": 1,
                    "valid_samples": 1,
                },
                "resume": {
                    "train_file": str(resume_train),
                    "valid_file": str(resume_valid),
                    "train_samples": 2,
                    "valid_samples": 1,
                },
            },
        }
    }

    result = build_multitask_dataset(registry, "multitask_sft")
    train_rows = [json.loads(line) for line in Path(result["train_out"]).read_text(encoding="utf-8").splitlines()]

    assert result["train_total"] == 3
    assert result["valid_total"] == 2
    assert result["train_stats"] == {"jd": 1, "resume": 2}
    assert {row["meta"]["dataset_task"] for row in train_rows} == {"jd", "resume"}


def test_build_multitask_dataset_prefers_unique_source_groups(tmp_path: Path) -> None:
    train = tmp_path / "train_source.jsonl"
    valid = tmp_path / "valid_source.jsonl"
    _write_jsonl(
        train,
        [
            _sample("a_1", "resume_parse", "a variant 1", "a"),
            _sample("a_2", "resume_parse", "a variant 2", "a"),
            _sample("a_3", "resume_parse", "a variant 3", "a"),
            _sample("b_1", "resume_parse", "b variant 1", "b"),
            _sample("c_1", "resume_parse", "c variant 1", "c"),
        ],
    )
    _write_jsonl(valid, [_sample("v_1", "resume_parse", "valid", "v")])
    registry = {
        "multitask_sft": {
            "train_out": str(tmp_path / "train.jsonl"),
            "valid_out": str(tmp_path / "valid.jsonl"),
            "seed": 42,
            "tasks": {
                "resume": {
                    "train_file": str(train),
                    "valid_file": str(valid),
                    "train_samples": 3,
                    "valid_samples": 1,
                }
            },
        }
    }

    result = build_multitask_dataset(registry, "multitask_sft")
    selected = [json.loads(line) for line in Path(result["train_out"]).read_text(encoding="utf-8").splitlines()]

    assert {row["source_group"] for row in selected} == {"a", "b", "c"}
    assert result["train_diversity"]["resume"]["source_group_ratio"] == 1.0
