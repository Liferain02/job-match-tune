from jobmatch_tune.eval.generate_match_ranking_predictions import generate_predictions


def test_generate_predictions_parses_unique_records_once() -> None:
    labels = [
        {"query_id": "j1", "candidate_id": "c1"},
        {"query_id": "j1", "candidate_id": "c2"},
    ]
    jobs = [{"query_id": "j1", "text": "Python backend"}]
    candidates = [
        {"candidate_id": "c1", "text": "Python engineer"},
        {"candidate_id": "c2", "text": "Java engineer"},
    ]
    calls = []

    def parse_text(task: str, text: str) -> dict:
        calls.append((task, text))
        if task == "jd_parse":
            return {
                "ok": True,
                "data": {
                    "岗位方向": "后端开发",
                    "必备技能": ["Python"],
                    "学历要求": "",
                    "经验要求": "",
                },
            }
        skill = "Python" if "Python" in text else "Java"
        return {
            "ok": True,
            "data": {
                "目标岗位": "后端开发",
                "核心技能": [skill],
                "教育背景": [],
                "实习经历": [],
                "项目经历": [],
            },
        }

    predictions = generate_predictions(labels, jobs, candidates, parse_text)

    assert len(calls) == 3
    assert all(row["predicted_score"] >= 0 for row in predictions)
    assert all(row["meta"]["parse_ok"] for row in predictions)


def test_generate_predictions_assigns_negative_score_on_parse_failure() -> None:
    predictions = generate_predictions(
        [{"query_id": "j1", "candidate_id": "c1"}],
        [{"query_id": "j1", "text": "job"}],
        [{"candidate_id": "c1", "text": "resume"}],
        lambda _task, _text: {"ok": False},
    )

    assert predictions[0]["predicted_score"] == -1.0
    assert predictions[0]["meta"]["parse_ok"] is False
