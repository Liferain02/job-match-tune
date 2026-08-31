from jobmatch_tune.eval.run_match_ranking_eval import evaluate_ranking


def test_evaluate_ranking_reports_standard_ir_metrics_and_readiness():
    labels = [
        {
            "query_id": "jd-1",
            "candidate_id": "r-1",
            "relevance": 3,
            "meta": {"annotation_status": "human_verified"},
        },
        {
            "query_id": "jd-1",
            "candidate_id": "r-2",
            "relevance": 0,
            "meta": {"annotation_status": "human_verified"},
        },
        {
            "query_id": "jd-2",
            "candidate_id": "r-3",
            "relevance": 2,
            "meta": {"annotation_status": "human_verified"},
        },
        {
            "query_id": "jd-2",
            "candidate_id": "r-4",
            "relevance": 0,
            "meta": {"annotation_status": "human_verified"},
        },
    ]
    predictions = [
        {"query_id": "jd-1", "candidate_id": "r-1", "predicted_score": 90},
        {"query_id": "jd-1", "candidate_id": "r-2", "predicted_score": 10},
        {"query_id": "jd-2", "candidate_id": "r-3", "predicted_score": 80},
        {"query_id": "jd-2", "candidate_id": "r-4", "predicted_score": 20},
    ]

    report = evaluate_ranking(labels, predictions)

    assert report["metrics"]["ndcg@10"] == 1.0
    assert report["metrics"]["recall@1"] == 1.0
    assert report["metrics"]["mrr"] == 1.0
    assert report["worst_queries"][0]["query_id"] == "jd-1"
    assert report["worst_queries"][0]["ranking"][0]["candidate_id"] == "r-1"
    assert report["formal_evaluation_ready"] is False
    assert "至少需要 20 个独立查询岗位" in report["readiness_blockers"]


def test_evaluate_ranking_rejects_missing_prediction_pair():
    labels = [{"query_id": "jd-1", "candidate_id": "r-1", "relevance": 1}]

    try:
        evaluate_ranking(labels, [])
    except ValueError as error:
        assert "prediction keys differ" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_equal_scores_do_not_use_relevance_as_tie_breaker():
    labels = [
        {"query_id": "jd-1", "candidate_id": "r-1", "relevance": 0},
        {"query_id": "jd-1", "candidate_id": "r-2", "relevance": 3},
    ]
    predictions = [
        {"query_id": "jd-1", "candidate_id": "r-1", "predicted_score": 1},
        {"query_id": "jd-1", "candidate_id": "r-2", "predicted_score": 1},
    ]

    report = evaluate_ranking(labels, predictions)

    assert report["metrics"]["ndcg@1"] == 0.0


def test_expert_reviewed_set_is_regression_ready_but_not_formal_human_holdout():
    labels = []
    predictions = []
    for query_index in range(20):
        for candidate_index, relevance in enumerate((3, 2, 1, 0, 0)):
            labels.append(
                {
                    "query_id": f"jd-{query_index}",
                    "candidate_id": f"r-{query_index}-{candidate_index}",
                    "relevance": relevance,
                    "meta": {"annotation_status": "expert_reviewed_by_codex"},
                }
            )
            predictions.append(
                {
                    "query_id": f"jd-{query_index}",
                    "candidate_id": f"r-{query_index}-{candidate_index}",
                    "predicted_score": float(relevance),
                }
            )

    report = evaluate_ranking(labels, predictions)

    assert report["expert_regression_ready"] is True
    assert report["formal_evaluation_ready"] is False
    assert "全部相关性标签必须人工复核" in report["readiness_blockers"]
