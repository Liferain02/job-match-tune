import json
from pathlib import Path
from unittest.mock import patch

import pytest

from jobmatch_tune.eval.report_data_readiness import (
    audit_sft_files,
    build_multitask_report,
    build_report,
    build_task_report,
    count_holdout_overlap,
    count_resume_evaluation_overlap,
    profile_match_education_consistency,
    profile_match_education_distribution,
    profile_match_experience_distribution,
    profile_match_training_sources,
    profile_match_training_privacy,
    read_match_evaluation_readiness,
)
from jobmatch_tune.dataset.templates import resume_parse_prompt
from jobmatch_tune.resume.privacy import sanitize_resume_text_for_training


@pytest.fixture(autouse=True)
def _fresh_pipeline_for_readiness_unit_tests():
    with patch(
        "jobmatch_tune.eval.report_data_readiness.build_pipeline_freshness_report",
        return_value={"fresh": True, "normalization": {}, "dependencies": []},
    ), patch(
        "jobmatch_tune.eval.report_data_readiness.profile_match_training_sources",
        return_value={
            "supports_real_pair_quality_claim": True,
            "source_concentration_ready": True,
        },
    ), patch(
        "jobmatch_tune.eval.report_data_readiness.profile_match_training_privacy",
        return_value={"total_rows": 1, "privacy_ready": True},
    ), patch(
        "jobmatch_tune.eval.report_data_readiness.profile_match_education_consistency",
        return_value={"total_rows": 1, "education_consistency_ready": True},
    ), patch(
        "jobmatch_tune.eval.report_data_readiness.profile_match_experience_distribution",
        return_value={"experience_distribution_ready": True},
    ), patch(
        "jobmatch_tune.eval.report_data_readiness.read_match_evaluation_readiness",
        return_value={
            "regression_ready": True,
            "blind_ready": False,
            "decision_ready": False,
            "ranking_ready": False,
        },
    ):
        yield


def test_build_task_report_not_ready_when_pool_missing():
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [5000, 500, 500, 0]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            report = build_task_report("jd", "a", "b", "c", "d")
    assert report["counts"]["combined_pool"] == 0
    assert report["ready_for_sft"] is False


@patch(
    "jobmatch_tune.eval.report_data_readiness.build_multitask_report",
    return_value={
        "task": "multitask",
        "ready_for_sft": True,
        "format_ready": True,
        "has_required_mix": True,
        "counts": {"train": 9540, "valid": 1180},
    },
)
def test_build_report_summarizes_not_ready_tasks(_multitask):
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [
            4000, 500, 500, 8000,  # jd
            2000, 200, 200, 3000,  # resume
            97, 12, 19, 0,  # match
            9540, 1180,  # multitask
        ]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            with patch("jobmatch_tune.eval.report_data_readiness.read_json_file") as read_json_file:
                read_json_file.side_effect = [
                    {},
                    {},
                    {},
                ]
                report = build_report()
    assert report["summary"]["all_ready_for_training"] is False
    assert "match" in report["summary"]["not_ready_tasks"]
    assert report["summary"]["match_regression_evaluation_ready"] is True
    assert report["summary"]["match_blind_evaluation_ready"] is False
    assert report["summary"]["match_level_decision_ready"] is False
    assert report["summary"]["match_ranking_evaluation_ready"] is False


@patch(
    "jobmatch_tune.eval.report_data_readiness.build_multitask_report",
    return_value={
        "task": "multitask",
        "ready_for_sft": True,
        "format_ready": True,
        "has_required_mix": True,
        "counts": {"train": 9540, "valid": 1180},
    },
)
def test_build_report_requires_resume_privacy_and_preference(_multitask):
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [
            4240, 530, 530, 8000,  # jd
            10000, 1000, 1000, 3000,  # resume
            3000, 400, 400, 4500,  # match
            9540, 1180,  # multitask
        ]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            with patch("jobmatch_tune.eval.report_data_readiness.count_holdout_overlap", return_value=0):
                with patch("jobmatch_tune.eval.report_data_readiness.read_json_file") as read_json_file:
                    read_json_file.side_effect = [
                        {"profile_ready": True},
                        {"ready_for_resume_training": True},
                        {"ready_for_dpo": True, "ready_for_dpo_smoke": True},
                    ]
                    report = build_report()
    assert report["summary"]["all_ready_for_training"] is True
    assert report["summary"]["ready_for_sft_experiment"] is True
    assert report["summary"]["dpo_paused_by_quality_goal"] is True
    assert report["summary"]["dpo_execution_ready"] is False
    assert report["tasks"]["resume"]["privacy_ready"] is True
    assert report["tasks"]["resume"]["readiness_dimensions"] == {
        "format_ready": True,
        "privacy_ready": True,
        "license_ready": True,
        "provenance_ready": True,
        "source_diversity_ready": True,
        "condition_distribution_ready": None,
        "real_pair_quality_evidence_ready": None,
        "split_leakage_ready": True,
        "evaluation_overlap_ready": True,
    }


@patch(
    "jobmatch_tune.eval.report_data_readiness.build_multitask_report",
    return_value={
        "task": "multitask",
        "ready_for_sft": True,
        "format_ready": True,
        "has_required_mix": True,
        "counts": {"train": 9540, "valid": 1180},
    },
)
def test_build_report_blocks_training_when_resume_privacy_fails(_multitask):
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [
            4240, 530, 530, 8000,
            10000, 1000, 1000, 3000,
            3000, 400, 400, 4500,
            9540, 1180,
        ]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            with patch("jobmatch_tune.eval.report_data_readiness.count_holdout_overlap", return_value=0):
                with patch("jobmatch_tune.eval.report_data_readiness.read_json_file") as read_json_file:
                    read_json_file.side_effect = [
                        {"profile_ready": True},
                        {"ready_for_resume_training": False},
                        {"ready_for_dpo": True, "ready_for_dpo_smoke": True},
                    ]
                    report = build_report()
    assert report["summary"]["all_ready_for_training"] is False
    assert report["summary"]["not_ready_tasks"] == ["resume"]
    assert report["tasks"]["resume"]["privacy_ready"] is False


@patch(
    "jobmatch_tune.eval.report_data_readiness.build_multitask_report",
    return_value={
        "task": "multitask",
        "ready_for_sft": True,
        "format_ready": True,
        "has_required_mix": True,
        "counts": {"train": 9540, "valid": 1180},
    },
)
def test_build_report_blocks_match_without_real_pair_quality_evidence(_multitask):
    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [
            4240, 530, 530, 8000,
            10000, 1000, 1000, 3000,
            3000, 400, 400, 4500,
            9540, 1180,
        ]
        with patch("jobmatch_tune.eval.report_data_readiness.audit_sft_files") as audit:
            audit.return_value = {
                "invalid_json": 0,
                "duplicate_ids": 0,
                "cross_split_duplicate_hashes": 0,
                "field_quality_ok": True,
            }
            with patch(
                "jobmatch_tune.eval.report_data_readiness.profile_match_training_sources",
                return_value={
                    "supports_real_pair_quality_claim": False,
                    "source_concentration_ready": True,
                },
            ), patch(
                "jobmatch_tune.eval.report_data_readiness.count_holdout_overlap",
                return_value=0,
            ), patch(
                "jobmatch_tune.eval.report_data_readiness.read_json_file"
            ) as read_json_file:
                read_json_file.side_effect = [
                    {"profile_ready": True},
                    {"ready_for_resume_training": True},
                    {"ready_for_dpo": True, "ready_for_dpo_smoke": True},
                ]
                report = build_report()

    assert report["tasks"]["match"]["real_pair_quality_evidence_ready"] is False
    assert report["tasks"]["match"]["ready_for_sft"] is False
    assert report["tasks"]["match"]["readiness_dimensions"][
        "real_pair_quality_evidence_ready"
    ] is False
    assert "match" in report["summary"]["not_ready_tasks"]


def test_match_evaluation_readiness_includes_expert_ranking_regression() -> None:
    with patch("jobmatch_tune.eval.report_data_readiness.read_json_file") as read_json_file:
        read_json_file.side_effect = [
            {"gold_ready": True, "ranking_ready": False},
            {
                "expert_regression_ready": True,
                "formal_evaluation_ready": False,
                "readiness_blockers": ["全部相关性标签必须人工复核"],
                "data_profile": {"num_queries": 20},
                "metrics": {"ndcg@5": 0.8},
            },
        ]
        report = read_match_evaluation_readiness()

    assert report["ranking_ready"] is True
    assert report["ranking_formal_ready"] is False
    assert report["ranking_expert_regression_ready"] is True


def _sample(row_id: str) -> dict:
    return {
        "id": row_id,
        "task_type": "jd_parse",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "岗位名称：后端开发工程师"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "岗位方向": "后端开发",
                        "核心职责": ["负责服务开发"],
                        "必备技能": ["Java"],
                        "学历要求": "本科",
                        "经验要求": "3年",
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }


def test_audit_sft_files_detects_cross_split_content_overlap(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    row = _sample("row_1")
    train.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    row["id"] = "row_2"
    valid.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = audit_sft_files("jd", [str(train), str(valid)])

    assert audit["cross_split_duplicate_hashes"] == 1


def test_audit_sft_files_detects_cross_split_normalized_input_overlap(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train_row = _sample("row_1")
    valid_row = _sample("row_2")
    train_row["messages"][1]["content"] = "岗位名称：后端 开发工程师！"
    valid_row["messages"][1]["content"] = "岗位名称:后端开发工程师"
    valid_row["messages"][-1]["content"] = json.dumps(
        {
            "岗位方向": "后端开发",
            "核心职责": ["负责接口治理"],
            "必备技能": ["Python"],
            "学历要求": "本科",
            "经验要求": "3年",
        },
        ensure_ascii=False,
    )
    train.write_text(json.dumps(train_row, ensure_ascii=False) + "\n", encoding="utf-8")
    valid.write_text(json.dumps(valid_row, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = audit_sft_files("jd", [str(train), str(valid)])

    assert audit["cross_split_duplicate_hashes"] == 0
    assert audit["cross_split_normalized_input_hashes"] == 1


def test_audit_sft_files_detects_shared_match_entity_across_splits(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train_row = _sample("match_train")
    train_row["linked_source_groups"] = ["match_jd:shared", "match_resume:train"]
    valid_row = _sample("match_valid")
    valid_row["messages"][1]["content"] = "不同的匹配输入"
    valid_row["linked_source_groups"] = ["match_jd:shared", "match_resume:valid"]
    train.write_text(json.dumps(train_row, ensure_ascii=False) + "\n", encoding="utf-8")
    valid.write_text(json.dumps(valid_row, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = audit_sft_files("jd", [str(train), str(valid)])

    assert audit["cross_split_linked_source_groups"] == 1
    assert audit["cross_split_linked_source_group_types"] == {"match_jd": 1}


def test_audit_sft_files_detects_placeholder_match_recommendation(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    row = _sample("match_placeholder")
    row["task_type"] = "match"
    assistant = json.loads(row["messages"][-1]["content"])
    assistant["推荐投递岗位方向"] = ["技能相近岗位"]
    row["messages"][-1]["content"] = json.dumps(assistant, ensure_ascii=False)
    train.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = audit_sft_files("jd", [str(train)])

    assert audit["placeholder_match_recommendation_rows"] == 1


def test_audit_sft_files_detects_match_prompt_without_project_evidence_field(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    row = _sample("match_without_projects")
    row["task_type"] = "match"
    train.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    audit = audit_sft_files("jd", [str(train)])

    assert audit["match_rows_missing_project_evidence_field"] == 1


def test_build_multitask_report_requires_all_tasks(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train_rows = []
    valid_rows = []
    for index, task in enumerate(["jd", "resume", "match"], start=1):
        row = _sample(f"train_{index}")
        row["messages"][1]["content"] = f"岗位名称：{task} train"
        row["meta"] = {"dataset_task": task}
        train_rows.append(row)
        valid_row = _sample(f"valid_{index}")
        valid_row["messages"][1]["content"] = f"岗位名称：{task} valid"
        valid_row["meta"] = {"dataset_task": task}
        valid_rows.append(valid_row)
    train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows), encoding="utf-8")
    valid.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid_rows), encoding="utf-8")

    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [9540, 1180]
        report = build_multitask_report(str(train), str(valid))

    assert report["has_required_mix"] is True
    assert report["source_diversity_ready"] is True
    assert report["cross_split_source_groups"] == 0
    assert report["ready_for_sft"] is True


def test_build_multitask_report_rejects_template_dominated_selection(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train_rows = []
    valid_rows = []
    for split, target in (("train", train_rows), ("valid", valid_rows)):
        for task in ("jd", "match"):
            row = _sample(f"{split}_{task}")
            row["messages"][1]["content"] = f"{split} {task}"
            row["meta"] = {"dataset_task": task}
            row["source_group"] = f"{split}_{task}"
            target.append(row)
        for index in range(10):
            row = _sample(f"{split}_resume_{index}")
            row["messages"][1]["content"] = f"{split} resume {index}"
            row["meta"] = {"dataset_task": "resume"}
            row["source_group"] = f"{split}_resume_same"
            target.append(row)
    train.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    valid.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid_rows),
        encoding="utf-8",
    )

    with patch("jobmatch_tune.eval.report_data_readiness.count_jsonl") as mocked:
        mocked.side_effect = [9540, 1180]
        report = build_multitask_report(str(train), str(valid))

    assert report["source_diversity"]["train"]["resume"]["source_group_ratio"] == 0.1
    assert report["source_diversity_ready"] is False
    assert report["ready_for_sft"] is False


def test_count_holdout_overlap_normalizes_jd_parse_suffix(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    train.write_text(json.dumps({"id": "jd_1_jd_parse"}) + "\n", encoding="utf-8")
    holdout.write_text(json.dumps({"source_id": "jd_1"}) + "\n", encoding="utf-8")

    assert count_holdout_overlap([str(train)], str(holdout)) == 1


def test_count_resume_evaluation_overlap_compares_prompt_content(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    resume_text = "姓名：张三\n目标岗位：后端开发\n核心技能：Java、MySQL"
    prompt = resume_parse_prompt(sanitize_resume_text_for_training(resume_text))
    train.write_text(
        json.dumps({"messages": [{"role": "system"}, {"content": prompt}]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    evaluation.write_text(
        json.dumps({"text": resume_text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert count_resume_evaluation_overlap([str(train)], str(evaluation)) == 1


def test_profile_match_training_sources_exposes_synthetic_only_pool(tmp_path: Path):
    pool = tmp_path / "match.jsonl"
    rows = [
        {
            "id": "synthetic_match_hf_job_educational_a",
            "source_type": "synthetic_text",
            "meta": {"generator": "pairing_v1"},
        },
        {"id": "b", "source_type": "synthetic_text", "meta": {"generator": "pairing_v1"}},
        {"id": "c", "source_type": "synthetic_text", "meta": {"generator": "pairing_v1"}},
    ]
    pool.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    profile = profile_match_training_sources(str(pool))

    assert profile["training_origin"] == "synthetic_only"
    assert profile["synthetic_rate"] == 1.0
    assert profile["human_reviewed_pair_ratio"] == 0.0
    assert profile["real_pair_ratio"] == 0.0
    assert profile["supports_real_pair_quality_claim"] is False
    assert profile["educational_source_rate"] == 0.3333
    assert profile["source_concentration_ready"] is True
    assert profile["pair_type_counts"] == {
        "synthetic_rule_pair": 3,
        "human_reviewed_pair": 0,
        "real_observed_pair": 0,
        "unknown_non_synthetic_pair": 0,
    }


def test_profile_match_training_sources_does_not_present_curated_fictional_as_real(
    tmp_path: Path,
):
    pool = tmp_path / "match.jsonl"
    row = {
        "id": "curated_1",
        "source_type": "curated_fictional_pair",
        "meta": {
            "generator": "ai_assisted_individual_case_authoring",
            "annotation_status": "repository_curated_unverified",
        },
    }
    pool.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    profile = profile_match_training_sources(str(pool))

    assert profile["training_origin"] == "curated_fictional_only"
    assert profile["curated_fictional_rows"] == 1
    assert profile["curated_fictional_pair_ratio"] == 1.0
    assert profile["non_synthetic_rows"] == 0
    assert profile["supports_real_pair_quality_claim"] is False


def test_profile_match_training_sources_rejects_educational_source_dominance(tmp_path: Path):
    pool = tmp_path / "match.jsonl"
    rows = [
        {"id": f"synthetic_match_hf_job_educational_{index}", "source_type": "synthetic_text"}
        for index in range(3)
    ] + [{"id": "synthetic_match_official_1", "source_type": "synthetic_text"}]
    pool.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    profile = profile_match_training_sources(str(pool))

    assert profile["educational_source_rate"] == 0.75
    assert profile["source_concentration_ready"] is False


def test_profile_match_training_privacy_detects_sensitive_resume_fields(tmp_path: Path):
    pool = tmp_path / "match.jsonl"
    rows = [
        {"id": "a", "resume_text": "婚姻状况：已婚\n核心技能：Python"},
        {"id": "b", "resume_text": "核心技能：Java"},
    ]
    pool.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    profile = profile_match_training_privacy(str(pool))

    assert profile["rows_with_privacy_findings"] == 1
    assert profile["finding_counts"]["marital_status"] == 1
    assert profile["privacy_ready"] is False


def test_profile_match_education_consistency_detects_stale_label(tmp_path: Path):
    pool = tmp_path / "match.jsonl"
    rows = [
        {
            "id": "wrong",
            "jd_text": "任职要求：硕士研究生及以上学历",
            "resume_text": "教育背景：本科",
            "label": {"学历匹配": True},
        },
        {
            "id": "preferred",
            "jd_text": "任职要求：硕士优先",
            "resume_text": "教育背景：本科",
            "label": {"学历匹配": True},
        },
    ]
    pool.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    profile = profile_match_education_consistency(str(pool))

    assert profile["disagreement_rows"] == 1
    assert profile["examples"][0]["id"] == "wrong"
    assert profile["education_consistency_ready"] is False


def test_profile_match_experience_distribution_requires_both_labels_per_split(tmp_path: Path):
    pool = tmp_path / "match.jsonl"
    rows = [
        {
            "id": "matched",
            "jd_text": "任职要求：两年以上开发经验",
            "label": {"经验匹配": True},
            "meta": {"entity_split": "train"},
        },
        {
            "id": "unmatched",
            "jd_text": "任职要求：三年以上开发经验",
            "label": {"经验匹配": False},
            "meta": {"entity_split": "train"},
        },
    ]
    pool.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    profile = profile_match_experience_distribution(str(pool))

    assert profile["explicit_experience_requirement_count"] == 2
    assert profile["matched_rows"] == 1
    assert profile["unmatched_rows"] == 1
    assert profile["positive_rate"] == 0.5
    assert profile["experience_distribution_ready"] is True
    assert profile["threshold_counts"] == {"2年": 1, "3年": 1}


def test_profile_match_education_distribution_reports_levels_and_splits(tmp_path: Path):
    pool = tmp_path / "match.jsonl"
    rows = [
        {
            "id": "bachelor",
            "jd_text": "任职要求：本科及以上学历",
            "label": {"学历匹配": True},
            "meta": {"entity_split": "train"},
        },
        {
            "id": "master",
            "jd_text": "任职要求：硕士研究生及以上学历",
            "label": {"学历匹配": False},
            "meta": {"entity_split": "valid"},
        },
    ]
    pool.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    profile = profile_match_education_distribution(str(pool))

    assert profile["explicit_education_requirement_count"] == 2
    assert profile["matched_rows"] == 1
    assert profile["unmatched_rows"] == 1
    assert profile["requirement_level_counts"] == {"本科": 1, "硕士/研究生": 1}
    assert profile["split_counts"]["train"]["matched"] == 1


def test_profile_match_experience_distribution_rejects_single_class_split(tmp_path: Path):
    pool = tmp_path / "match.jsonl"
    pool.write_text(
        json.dumps(
            {
                "id": "only_negative",
                "jd_text": "任职要求：三年以上开发经验",
                "label": {"经验匹配": False},
                "meta": {"entity_split": "valid"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    profile = profile_match_experience_distribution(str(pool))

    assert profile["experience_distribution_ready"] is False
