from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from typing import Any, Iterable

from jobmatch_tune.eval.metrics import bootstrap_confidence_interval
from jobmatch_tune.utils.io import read_jsonl, write_text


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("query_id") or ""), str(row.get("candidate_id") or "")


def _dcg(relevances: list[int], k: int) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances[:k])
    )


def _query_metrics(
    items: list[tuple[float, int, str]],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    ranked = sorted(items, key=lambda item: (-item[0], item[2]))
    ranked_relevance = [relevance for _, relevance, _ in ranked]
    ideal_relevance = sorted((relevance for _, relevance, _ in items), reverse=True)
    relevant_total = sum(relevance > 0 for relevance in ideal_relevance)
    result: dict[str, float] = {}
    for k in k_values:
        ideal_dcg = _dcg(ideal_relevance, k)
        result[f"ndcg@{k}"] = _dcg(ranked_relevance, k) / ideal_dcg if ideal_dcg else 0.0
        result[f"recall@{k}"] = (
            sum(relevance > 0 for relevance in ranked_relevance[:k]) / relevant_total
            if relevant_total
            else 0.0
        )
    first_relevant = next(
        (rank for rank, relevance in enumerate(ranked_relevance, start=1) if relevance > 0),
        None,
    )
    result["mrr"] = 1.0 / first_relevant if first_relevant else 0.0
    result["ranking"] = [
        {"candidate_id": candidate_id, "predicted_score": score, "relevance": relevance}
        for score, relevance, candidate_id in ranked
    ]
    return result


def evaluate_ranking(
    label_rows: Iterable[dict[str, Any]],
    prediction_rows: Iterable[dict[str, Any]],
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    labels = list(label_rows)
    predictions = list(prediction_rows)
    label_keys = [_key(row) for row in labels]
    prediction_keys = [_key(row) for row in predictions]
    if any(not query_id or not candidate_id for query_id, candidate_id in label_keys + prediction_keys):
        raise ValueError("query_id and candidate_id are required")
    if len(set(label_keys)) != len(label_keys) or len(set(prediction_keys)) != len(prediction_keys):
        raise ValueError("query_id + candidate_id must be unique")
    if set(label_keys) != set(prediction_keys):
        missing = sorted(set(label_keys) - set(prediction_keys))
        extra = sorted(set(prediction_keys) - set(label_keys))
        raise ValueError(f"prediction keys differ from labels: missing={missing[:5]} extra={extra[:5]}")

    score_by_key: dict[tuple[str, str], float] = {}
    for row in predictions:
        score = float(row["predicted_score"])
        if not math.isfinite(score):
            raise ValueError(f"predicted_score must be finite: {_key(row)}")
        score_by_key[_key(row)] = score

    grouped: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    annotation_statuses: Counter[str] = Counter()
    for row in labels:
        relevance = row.get("relevance")
        if not isinstance(relevance, int) or not 0 <= relevance <= 3:
            raise ValueError(f"relevance must be an integer in 0..3: {_key(row)}")
        grouped[_key(row)[0]].append((score_by_key[_key(row)], relevance, _key(row)[1]))
        annotation_statuses[str((row.get("meta") or {}).get("annotation_status") or "missing")] += 1

    query_reports = [
        {"query_id": query_id, **_query_metrics(items, k_values)}
        for query_id, items in grouped.items()
    ]
    metric_names = [*(f"ndcg@{k}" for k in k_values), *(f"recall@{k}" for k in k_values), "mrr"]
    metrics = {
        metric: sum(report[metric] for report in query_reports) / len(query_reports)
        if query_reports
        else 0.0
        for metric in metric_names
    }
    candidates_per_query = [len(items) for items in grouped.values()]
    query_has_positive = [
        any(relevance > 0 for _, relevance, _ in items) for items in grouped.values()
    ]
    query_has_multiple_grades = [
        len({relevance for _, relevance, _ in items}) >= 2 for items in grouped.values()
    ]
    human_verified = annotation_statuses == {"human_verified": len(labels)}
    expert_reviewed = annotation_statuses == {"expert_reviewed_by_codex": len(labels)}
    formal_ready = bool(
        len(grouped) >= 20
        and candidates_per_query
        and min(candidates_per_query) >= 5
        and all(query_has_positive)
        and all(query_has_multiple_grades)
        and human_verified
    )
    largest_meaningful_k = max(
        (k for k in k_values if k < min(candidates_per_query, default=0)),
        default=min(k_values),
    )
    confidence_metrics = [
        name
        for name in (f"ndcg@{largest_meaningful_k}", f"recall@{largest_meaningful_k}", "mrr")
        if name in metrics
    ]
    return {
        "task": "match_ranking",
        "metrics": metrics,
        "confidence_intervals": {
            metric: bootstrap_confidence_interval(
                query_reports,
                lambda samples, name=metric: (
                    sum(report[name] for report in samples) / len(samples) if samples else 0.0
                ),
            )
            for metric in confidence_metrics
        },
        "data_profile": {
            "num_queries": len(grouped),
            "num_pairs": len(labels),
            "min_candidates_per_query": min(candidates_per_query, default=0),
            "max_candidates_per_query": max(candidates_per_query, default=0),
            "queries_without_positive": sum(not value for value in query_has_positive),
            "queries_without_multiple_relevance_grades": sum(
                not value for value in query_has_multiple_grades
            ),
            "annotation_status_counts": dict(annotation_statuses),
        },
        "worst_queries": sorted(
            query_reports,
            key=lambda report: (report.get("ndcg@3", 0.0), report["query_id"]),
        )[:5],
        "formal_evaluation_ready": formal_ready,
        "expert_regression_ready": bool(
            len(grouped) >= 20
            and candidates_per_query
            and min(candidates_per_query) >= 5
            and all(query_has_positive)
            and all(query_has_multiple_grades)
            and (human_verified or expert_reviewed)
        ),
        "readiness_blockers": [
            message
            for condition, message in (
                (len(grouped) < 20, "至少需要 20 个独立查询岗位"),
                (min(candidates_per_query, default=0) < 5, "每个岗位至少需要 5 份候选简历"),
                (not all(query_has_positive), "每个岗位至少需要 1 份相关简历"),
                (not all(query_has_multiple_grades), "每个岗位至少需要 2 档人工相关性标签"),
                (not human_verified, "全部相关性标签必须人工复核"),
            )
            if condition
        ],
        "interpretation_warnings": [
            f"候选池最多只有 {max(candidates_per_query, default=0)} 份，"
            f"K>={max(candidates_per_query, default=0)} 的 Recall 为平凡值，不应作为主指标"
        ]
        if any(k >= max(candidates_per_query, default=0) for k in k_values)
        else [],
        "metric_semantics": {
            "ndcg": "衡量高相关候选是否排在前面，并支持 0..3 分级相关性",
            "recall": "衡量前 K 名召回了多少人工标为相关的候选",
            "mrr": "衡量第一份相关候选出现的位置",
            "confidence_intervals": "以岗位查询为重采样单位的 95% percentile bootstrap",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate resume ranking for each job query")
    parser.add_argument("--labels", default="data/private/match_ranking_gold.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="outputs/eval_reports/match_ranking_report.json")
    args = parser.parse_args()
    report = evaluate_ranking(read_jsonl(args.labels), read_jsonl(args.predictions))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    write_text(args.out, rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
