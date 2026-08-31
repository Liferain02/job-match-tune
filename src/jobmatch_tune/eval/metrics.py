from __future__ import annotations

from collections.abc import Iterable
from collections import Counter
import math
import random
import re
from typing import Any, Callable, Sequence


def normalize_items(items: Iterable[str]) -> set[str]:
    return {str(item).strip().lower() for item in items if str(item).strip()}


def precision_recall_f1(pred: Iterable[str], gold: Iterable[str]) -> dict[str, float]:
    pred_set = normalize_items(pred)
    gold_set = normalize_items(gold)
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_set_metrics(
    pairs: Iterable[tuple[Iterable[str], Iterable[str]]],
) -> dict[str, float | int]:
    """Report both row-macro and corpus-micro metrics for set-valued fields."""
    pair_list = list(pairs)
    row_scores = [precision_recall_f1(pred, gold) for pred, gold in pair_list]
    true_positive = false_positive = false_negative = 0
    exact_matches = 0
    non_empty_gold_rows = 0
    for pred, gold in pair_list:
        pred_set = normalize_items(pred)
        gold_set = normalize_items(gold)
        true_positive += len(pred_set & gold_set)
        false_positive += len(pred_set - gold_set)
        false_negative += len(gold_set - pred_set)
        exact_matches += pred_set == gold_set
        non_empty_gold_rows += bool(gold_set)

    micro_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0 if not pair_list else 0.0
    )
    micro_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0 if not pair_list else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    count = len(pair_list)
    return {
        "precision": sum(score["precision"] for score in row_scores) / count if count else 0.0,
        "recall": sum(score["recall"] for score in row_scores) / count if count else 0.0,
        "f1": sum(score["f1"] for score in row_scores) / count if count else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "row_exact_match": exact_matches / count if count else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "non_empty_gold_rows": non_empty_gold_rows,
        "num_rows": count,
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def text_exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def classification_metrics(
    predictions: Sequence[str],
    golds: Sequence[str],
    *,
    labels: Sequence[str],
    ordinal: bool = False,
) -> dict[str, Any]:
    """Dependency-free multiclass metrics; unknown predictions count as errors."""
    if len(predictions) != len(golds):
        raise ValueError("predictions and golds must have the same length")
    label_list = list(labels)
    label_set = set(label_list)
    unknown_predictions = Counter(pred for pred in predictions if pred not in label_set)
    normalized_predictions = [
        pred if pred in label_set else "__invalid__" for pred in predictions
    ]
    confusion = {
        gold: {
            pred: sum(
                1
                for actual, guessed in zip(golds, normalized_predictions, strict=True)
                if actual == gold and guessed == pred
            )
            for pred in [*label_list, "__invalid__"]
        }
        for gold in label_list
    }
    per_class: dict[str, dict[str, float | int]] = {}
    for label in label_list:
        true_positive = sum(
            1 for actual, guessed in zip(golds, predictions, strict=True) if actual == label and guessed == label
        )
        false_positive = sum(
            1 for actual, guessed in zip(golds, predictions, strict=True) if actual != label and guessed == label
        )
        support = sum(actual == label for actual in golds)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = len(golds)
    present = [values for values in per_class.values() if values["support"]]
    accuracy = (
        sum(actual == guessed for actual, guessed in zip(golds, predictions, strict=True)) / total
        if total
        else 0.0
    )
    result: dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": (
            sum(float(values["recall"]) for values in present) / len(present) if present else 0.0
        ),
        "macro_f1": (
            sum(float(values["f1"]) for values in present) / len(present) if present else 0.0
        ),
        "weighted_f1": (
            sum(float(values["f1"]) * int(values["support"]) for values in present) / total
            if total
            else 0.0
        ),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "unknown_prediction_counts": dict(unknown_predictions),
        "num_samples": total,
    }
    if ordinal:
        ranks = {label: index for index, label in enumerate(label_list)}
        max_error = max(len(label_list) - 1, 1)
        errors = [
            abs(ranks[actual] - ranks[guessed]) if actual in ranks and guessed in ranks else max_error
            for actual, guessed in zip(golds, predictions, strict=True)
        ]
        result["ordinal_mae"] = sum(errors) / total if total else 0.0
    return result


def bootstrap_confidence_interval(
    items: Sequence[Any],
    statistic: Callable[[list[Any]], float],
    *,
    samples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
    strata: Callable[[Any], Any] | None = None,
) -> dict[str, float | int | str]:
    """Deterministic percentile bootstrap interval for a scalar metric."""
    values = list(items)
    point = statistic(values) if values else 0.0
    if not values or samples <= 0:
        return {
            "value": point,
            "lower": point,
            "upper": point,
            "confidence": confidence,
            "bootstrap_samples": 0,
            "method": "percentile_bootstrap",
        }
    rng = random.Random(seed)
    if strata is None:
        estimates = sorted(
            statistic([values[rng.randrange(len(values))] for _ in values])
            for _ in range(samples)
        )
        method = "percentile_bootstrap"
    else:
        buckets: dict[Any, list[Any]] = {}
        for item in values:
            buckets.setdefault(strata(item), []).append(item)
        estimates = sorted(
            statistic(
                [
                    bucket[rng.randrange(len(bucket))]
                    for bucket in buckets.values()
                    for _ in bucket
                ]
            )
            for _ in range(samples)
        )
        method = "stratified_percentile_bootstrap"
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, min(samples - 1, math.floor(tail * samples)))
    upper_index = max(0, min(samples - 1, math.ceil((1.0 - tail) * samples) - 1))
    return {
        "value": point,
        "lower": estimates[lower_index],
        "upper": estimates[upper_index],
        "confidence": confidence,
        "bootstrap_samples": samples,
        "method": method,
    }
