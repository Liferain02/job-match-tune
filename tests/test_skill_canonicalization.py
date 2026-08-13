from __future__ import annotations

import pytest

from jobmatch_tune.preprocess.skill_canonicalization import (
    canonicalize_skill_list,
    contains_skill_candidate,
    extract_known_skills,
)


SCHEMA = {
    "skill_alias": {
        "Python": ["python"],
        "PyTorch": ["pytorch"],
        "Pytest": ["pytest"],
        "MySQL": ["mysql", "My SOL"],
        "PostgreSQL": ["postgresql", "postgres"],
        "C++": ["cpp"],
        "C#": ["csharp"],
        "Node.js": ["nodejs"],
        "Vue 3": ["vue3"],
        "Spring Boot": ["springboot"],
        "FastAPI": ["fastapi"],
        "Kubernetes": ["kubernetes"],
    }
}


@pytest.mark.parametrize(
    ("ocr_text", "canonical"),
    [
        ("Py test", "Pytest"),
        ("Py  test", "Pytest"),
        ("Py-test", "Pytest"),
        ("Py thon", "Python"),
        ("Kubernet es", "Kubernetes"),
        ("My SOL", "MySQL"),
        ("C + +", "C++"),
        ("Node . js", "Node.js"),
        ("Vue 3", "Vue 3"),
        ("Spring  Boot", "Spring Boot"),
    ],
)
def test_known_vocabulary_recovers_bounded_ocr_splits(ocr_text: str, canonical: str) -> None:
    assert canonical in extract_known_skills(f"技能：{ocr_text}", SCHEMA)
    assert canonicalize_skill_list([ocr_text], SCHEMA) == [canonical]


@pytest.mark.parametrize(
    "text",
    [
        "Postman collection maintenance",
        "pytest-like ordinary text",
        "This is an ordinary English phrase",
        "irrelevant OCR t e x t",
        "prototype testing",
    ],
)
def test_bounded_ocr_matching_rejects_unrelated_text(text: str) -> None:
    assert "Pytest" not in extract_known_skills(text, SCHEMA)


def test_short_skill_names_do_not_enable_character_spaced_fuzzy_matching() -> None:
    assert not contains_skill_candidate("g o service", "Go")
    assert not contains_skill_candidate("g i t workflow", "Git")


def test_compact_key_collision_is_not_canonicalized() -> None:
    schema = {"skill_alias": {"AB": ["A-B"], "A B": []}}
    assert canonicalize_skill_list(["A  B"], schema, keep_unknown=True) == ["A B"]


def test_standalone_c_does_not_match_compound_language_names() -> None:
    assert not contains_skill_candidate("掌握 C 语言和 C++，也维护 C# 服务", "C")
    assert contains_skill_candidate("掌握 C、Python", "C")
