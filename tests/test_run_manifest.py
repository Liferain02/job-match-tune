import json
from pathlib import Path

from jobmatch_tune.train.run_manifest import build_run_manifest, file_sha256, write_run_manifest
from jobmatch_tune.utils.io import write_jsonl


def test_build_run_manifest_profiles_config_dataset_and_readiness(tmp_path: Path):
    config = tmp_path / "config.yaml"
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    readiness = tmp_path / "readiness.json"
    config.write_text("learning_rate: 1.0e-4\n", encoding="utf-8")
    write_jsonl(
        train,
        [
            {"id": "t1", "task_type": "jd_parse", "source_group": "source_1"},
            {"id": "t2", "task_type": "jd_parse", "source_group": "source_2"},
        ],
    )
    write_jsonl(valid, [{"id": "v1"}])
    readiness.write_text(
        json.dumps({"summary": {"all_ready_for_training": True}}, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = build_run_manifest(
        stage="sft",
        config_path=str(config),
        output_dir=str(tmp_path / "out"),
        train_file=str(train),
        valid_file=str(valid),
        readiness_report=str(readiness),
        cli_args={"config": str(config)},
    )

    assert manifest["stage"] == "sft"
    assert manifest["config"]["sha256"] == file_sha256(config)
    assert manifest["datasets"]["train"]["rows"] == 2
    assert manifest["datasets"]["valid"]["rows"] == 1
    assert manifest["datasets"]["train"]["task_counts"] == {"jd_parse": 2}
    assert manifest["datasets"]["train"]["unique_source_groups"] == 2
    assert manifest["readiness"]["summary"]["all_ready_for_training"] is True
    assert "git" in manifest


def test_write_run_manifest_creates_output_dir(tmp_path: Path):
    config = tmp_path / "config.yaml"
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    config.write_text("seed: 42\n", encoding="utf-8")
    write_jsonl(train, [{"id": "t1"}])
    write_jsonl(valid, [{"id": "v1"}])
    out_dir = tmp_path / "checkpoint"

    write_run_manifest(
        stage="dpo",
        config_path=str(config),
        output_dir=str(out_dir),
        train_file=str(train),
        valid_file=str(valid),
        readiness_report=str(tmp_path / "missing.json"),
    )

    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "dpo"
    assert manifest["readiness"]["exists"] is False
