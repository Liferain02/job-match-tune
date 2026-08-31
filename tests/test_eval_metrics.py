from jobmatch_tune.eval.metrics import (
    aggregate_set_metrics,
    bootstrap_confidence_interval,
    classification_metrics,
    precision_recall_f1,
    text_exact_match,
)


def test_precision_recall_f1():
    score = precision_recall_f1(["Python", "RAG"], ["python", "Java"])
    assert score["precision"] == 0.5
    assert score["recall"] == 0.5
    assert score["f1"] == 0.5


def test_text_exact_match_ignores_case_and_spaces():
    assert text_exact_match(" 本科及以上 ", "本科及以上") == 1.0
    assert text_exact_match("三年以上 工作经验", "三年以上工作经验") == 0.0


def test_aggregate_set_metrics_exposes_macro_and_micro_difference():
    score = aggregate_set_metrics([([], []), (["Python"], ["Python", "RAG"])])

    assert round(float(score["f1"]), 4) == 0.8333
    assert round(float(score["micro_f1"]), 4) == 0.6667
    assert score["false_negative"] == 1


def test_classification_metrics_reports_imbalance_and_invalid_predictions():
    score = classification_metrics(
        ["高匹配", "高匹配", "bad"],
        ["高匹配", "低匹配", "低匹配"],
        labels=["低匹配", "高匹配"],
        ordinal=True,
    )

    assert score["accuracy"] == 1 / 3
    assert score["balanced_accuracy"] == 0.5
    assert score["confusion_matrix"]["低匹配"]["__invalid__"] == 1
    assert score["unknown_prediction_counts"] == {"bad": 1}


def test_bootstrap_confidence_interval_is_deterministic():
    first = bootstrap_confidence_interval([0.0, 1.0], lambda values: sum(values) / len(values))
    second = bootstrap_confidence_interval([0.0, 1.0], lambda values: sum(values) / len(values))

    assert first == second
    assert first["lower"] == 0.0
    assert first["upper"] == 1.0
