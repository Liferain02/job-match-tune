from jobmatch_tune.eval.build_real_ranking_dataset import (
    bm25_scores,
    build_dataset,
    redact_public_profile,
)


def test_redact_public_profile_masks_direct_contact_fields():
    text, counts = redact_public_profile(
        "Email person@example.com, phone +380 67 123 45 67, https://example.com/me"
    )

    assert "person@example.com" not in text
    assert "+380 67 123 45 67" not in text
    assert "https://example.com/me" not in text
    assert counts == {"email": 1, "phone": 1, "url": 1}


def test_bm25_prefers_document_with_required_stack():
    scores = bm25_scores(
        "Python FastAPI PostgreSQL",
        ["Python FastAPI backend with PostgreSQL", "Java Spring backend"],
    )

    assert scores[0] > scores[1]


def test_build_dataset_keeps_expert_status_and_detects_training_overlap():
    jobs = [
        {
            "id": "j1",
            "Position": "Python developer",
            "Long Description": "Python FastAPI",
            "Exp Years": "2y",
            "English Level": "upper",
            "Primary Keyword": "Python",
            "Published": "2023-01-01",
        }
    ]
    candidates = [
        {
            "id": f"c{index}",
            "Position": "Candidate",
            "CV": "Python FastAPI" if index == 0 else f"Other skill {index}",
            "Experience Years": index,
            "English Level": "upper",
            "Primary Keyword": "Python",
        }
        for index in range(5)
    ]
    annotations = {
        "reviewed_at": "2026-08-31",
        "rubric_version": "v1",
        "queries": [
            {
                "job_id": "j1",
                "candidates": [
                    {
                        "candidate_id": f"c{index}",
                        "relevance": 3 if index == 0 else 0,
                        "rationale": "explicit evidence" if index == 0 else "no evidence",
                    }
                    for index in range(5)
                ],
            }
        ],
    }

    result = build_dataset(
        jobs,
        candidates,
        annotations,
        jd_training_rows=[{"raw_text": "Python FastAPI"}],
    )

    assert result["audit"]["num_pairs"] == 5
    assert result["audit"]["exact_training_overlap_free"] is False
    assert result["labels"][0]["meta"]["annotation_status"] == "expert_reviewed_by_codex"
    assert result["labels"][0]["meta"]["training_eligible"] is False
