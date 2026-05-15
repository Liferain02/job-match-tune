from __future__ import annotations

from collections.abc import Iterable
import re


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def text_exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0
