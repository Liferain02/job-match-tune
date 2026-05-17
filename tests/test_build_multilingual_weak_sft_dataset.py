from __future__ import annotations

from jobmatch_tune.dataset.build_multilingual_weak_sft_dataset import (
    build_multilingual_weak_sample,
    extract_en_education_requirement,
    extract_en_experience_requirement,
    infer_high_confidence_en_direction,
)


def test_infer_high_confidence_en_direction() -> None:
    direction, ok = infer_high_confidence_en_direction(
        "Senior Data Engineer",
        "Build Spark pipelines and data warehouse systems.",
    )
    assert ok is True
    assert direction == "数据开发"


def test_extract_en_requirements() -> None:
    text = "Bachelor's degree in CS. Minimum 5 years of experience in backend systems."
    assert extract_en_education_requirement(text) == "本科"
    assert "5 years" in extract_en_experience_requirement(text)


def test_build_multilingual_weak_sample_for_en() -> None:
    row = {
        "id": "demo_en_1",
        "job_title": "Senior Data Engineer",
        "language": "en",
        "clean_text": (
            "Position: Senior Data Engineer\n"
            "Responsibilities:\n"
            "- Build Spark data pipelines\n"
            "- Maintain ETL workflows\n"
            "Requirements:\n"
            "Bachelor's degree\n"
            "5 years of experience\n"
        ),
        "sections": {"responsibilities": "- Build Spark data pipelines\n- Maintain ETL workflows"},
        "labels": {},
    }
    schema = {"skill_alias": {"Spark": ["spark"], "ETL": ["etl"]}}
    sample = build_multilingual_weak_sample(row, schema)
    assert sample is not None
    assistant = sample["messages"][-1]["content"]
    assert "数据开发" in assistant
    assert "Spark" in assistant
