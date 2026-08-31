from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from jobmatch_tune.dataset.grouped_split import normalized_input_hash
from jobmatch_tune.utils.io import read_jsonl, write_jsonl, write_text


JOB_SOURCE = {
    "name": "djinni_job_descriptions_english",
    "url": "https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english",
    "revision": "b56a6054c10f1266a141e37c8df66c79ff2863af",
    "license": "MIT",
}
CANDIDATE_SOURCE = {
    "name": "djinni_candidate_profiles_english",
    "url": "https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english",
    "revision": "86255174c6c378a2b52cd7e81d79002c64d899a6",
    "license": "MIT",
}

_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
# Public source authors already removed residual PII. Keep this second pass deliberately
# conservative: an optional leading plus caused employment date ranges to be mistaken for phones.
_PHONE_RE = re.compile(r"(?<!\w)\+\d[\d ()-]{7,}\d(?!\w)")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-/]*", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "to",
    "we",
    "will",
    "with",
    "you",
    "your",
}


def redact_public_profile(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    value = str(text or "")
    for name, pattern, replacement in (
        ("email", _EMAIL_RE, "[EMAIL]"),
        ("url", _URL_RE, "[URL]"),
        ("phone", _PHONE_RE, "[PHONE]"),
    ):
        value, count = pattern.subn(replacement, value)
        counts[name] += count
    return value.strip(), counts


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def bm25_scores(query: str, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized_documents = [_tokenize(document) for document in documents]
    query_terms = set(_tokenize(query))
    if not documents:
        return []
    average_length = sum(map(len, tokenized_documents)) / len(tokenized_documents) or 1.0
    document_frequency = Counter(
        term for document in tokenized_documents for term in set(document) if term in query_terms
    )
    scores: list[float] = []
    for document in tokenized_documents:
        term_frequency = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * len(document) / average_length)
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_hashes(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> set[str]:
    hashes = set()
    for row in rows:
        for field in fields:
            value = str(row.get(field) or "").strip()
            if value:
                hashes.add(normalized_input_hash(value))
    return hashes


def build_dataset(
    jobs: Iterable[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
    annotations: dict[str, Any],
    *,
    training_rows: Iterable[dict[str, Any]] = (),
    jd_training_rows: Iterable[dict[str, Any]] = (),
    resume_training_rows: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    jobs_by_id = {str(row["id"]): row for row in jobs}
    candidates_by_id = {str(row["id"]): row for row in candidates}
    output_jobs: list[dict[str, Any]] = []
    output_candidates_by_id: dict[str, dict[str, Any]] = {}
    labels: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    redaction_counts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()

    for query in annotations.get("queries") or []:
        source_job_id = str(query["job_id"])
        if source_job_id not in jobs_by_id:
            raise ValueError(f"job_id not found: {source_job_id}")
        job = jobs_by_id[source_job_id]
        query_id = f"djinni-job-{source_job_id}"
        job_text, counts = redact_public_profile(str(job.get("Long Description") or ""))
        redaction_counts.update(counts)
        output_jobs.append(
            {
                "query_id": query_id,
                "source_id": source_job_id,
                "position": str(job.get("Position") or ""),
                "experience": str(job.get("Exp Years") or ""),
                "english_level": str(job.get("English Level") or ""),
                "primary_keyword": str(job.get("Primary Keyword") or ""),
                "published": str(job.get("Published") or ""),
                "text": job_text,
            }
        )
        query_candidates = query.get("candidates") or []
        if len(query_candidates) < 5:
            raise ValueError(f"query requires at least five candidates: {source_job_id}")
        documents: list[str] = []
        candidate_ids: list[str] = []
        for annotation in query_candidates:
            source_candidate_id = str(annotation["candidate_id"])
            pair = source_job_id, source_candidate_id
            if pair in seen_pairs:
                raise ValueError(f"duplicate job/candidate pair: {pair}")
            seen_pairs.add(pair)
            if source_candidate_id not in candidates_by_id:
                raise ValueError(f"candidate_id not found: {source_candidate_id}")
            candidate = candidates_by_id[source_candidate_id]
            candidate_id = f"djinni-candidate-{source_candidate_id}"
            candidate_text, counts = redact_public_profile(str(candidate.get("CV") or ""))
            redaction_counts.update(counts)
            output_candidates_by_id.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "source_id": source_candidate_id,
                    "position": str(candidate.get("Position") or ""),
                    "experience_years": candidate.get("Experience Years"),
                    "english_level": str(candidate.get("English Level") or ""),
                    "primary_keyword": str(candidate.get("Primary Keyword") or ""),
                    "text": candidate_text,
                },
            )
            relevance = annotation.get("relevance")
            if not isinstance(relevance, int) or not 0 <= relevance <= 3:
                raise ValueError(f"invalid relevance for {pair}: {relevance}")
            rationale = str(annotation.get("rationale") or "").strip()
            if not rationale:
                raise ValueError(f"missing rationale for {pair}")
            labels.append(
                {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "relevance": relevance,
                    "evidence": {"rationale": rationale},
                    "meta": {
                        "annotation_status": "expert_reviewed_by_codex",
                        "annotation_provenance": "manual_evidence_review_of_real_public_records",
                        "reviewer_identity": "codex_ai_not_human",
                        "evaluation_role": "external_real_ranking_regression",
                        "inspection_status": "inspected",
                        "training_eligible": False,
                        "reviewed_at": str(annotations.get("reviewed_at") or ""),
                        "rubric_version": str(annotations.get("rubric_version") or ""),
                    },
                }
            )
            candidate_ids.append(candidate_id)
            documents.append(f"{candidate.get('Position') or ''}\n{candidate_text}")
        scores = bm25_scores(f"{job.get('Position') or ''}\n{job_text}", documents)
        predictions.extend(
            {
                "query_id": query_id,
                "candidate_id": candidate_id,
                "predicted_score": score,
                "baseline": "bm25_per_query_pool",
            }
            for candidate_id, score in zip(candidate_ids, scores, strict=True)
        )

    if len({row["query_id"] for row in output_jobs}) != len(output_jobs):
        raise ValueError("job_id must be unique across queries")

    pair_training = list(training_rows)
    jd_hashes = _text_hashes(pair_training, ("jd_text",)) | _text_hashes(
        jd_training_rows, ("raw_text", "clean_text", "text")
    )
    resume_hashes = _text_hashes(pair_training, ("resume_text",)) | _text_hashes(
        resume_training_rows, ("raw_text", "clean_text", "text")
    )
    selected_jd_overlap = [
        row["query_id"] for row in output_jobs if normalized_input_hash(row["text"]) in jd_hashes
    ]
    selected_resume_overlap = [
        row["candidate_id"]
        for row in output_candidates_by_id.values()
        if normalized_input_hash(row["text"]) in resume_hashes
    ]
    relevance_counts = Counter(row["relevance"] for row in labels)
    query_counts = Counter(row["query_id"] for row in labels)
    audit = {
        "num_queries": len(output_jobs),
        "num_unique_candidates": len(output_candidates_by_id),
        "num_pairs": len(labels),
        "min_candidates_per_query": min(query_counts.values(), default=0),
        "max_candidates_per_query": max(query_counts.values(), default=0),
        "relevance_counts": {str(level): relevance_counts[level] for level in range(4)},
        "annotation_status": "expert_reviewed_by_codex",
        "formal_human_holdout": False,
        "jd_training_overlap_ids": selected_jd_overlap,
        "resume_training_overlap_ids": selected_resume_overlap,
        "exact_training_overlap_free": not selected_jd_overlap and not selected_resume_overlap,
        "redaction_counts": dict(redaction_counts),
    }
    return {
        "jobs": output_jobs,
        "candidates": list(output_candidates_by_id.values()),
        "labels": labels,
        "predictions": predictions,
        "audit": audit,
    }


def _optional_rows(path: str) -> list[dict[str, Any]]:
    return list(read_jsonl(path)) if path and Path(path).exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private real-record candidate-ranking set")
    parser.add_argument("--jobs-parquet", required=True)
    parser.add_argument("--candidates-parquet", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out-dir", default="data/private/djinni_real_ranking_v1")
    parser.add_argument("--training", default="data/eval/match_train_pool_combined.jsonl")
    parser.add_argument("--jd-training", default="data/eval/jd_train_pool_combined.jsonl")
    parser.add_argument("--resume-training", default="data/eval/resume_train_pool_combined.jsonl")
    args = parser.parse_args()

    import pandas as pd

    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    job_ids = {str(query["job_id"]) for query in annotations.get("queries") or []}
    candidate_ids = {
        str(candidate["candidate_id"])
        for query in annotations.get("queries") or []
        for candidate in query.get("candidates") or []
    }
    jobs = pd.read_parquet(args.jobs_parquet)
    candidates = pd.read_parquet(args.candidates_parquet)
    result = build_dataset(
        jobs[jobs["id"].isin(job_ids)].to_dict("records"),
        candidates[candidates["id"].isin(candidate_ids)].to_dict("records"),
        annotations,
        training_rows=_optional_rows(args.training),
        jd_training_rows=_optional_rows(args.jd_training),
        resume_training_rows=_optional_rows(args.resume_training),
    )
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "jobs.jsonl", result["jobs"])
    write_jsonl(out_dir / "candidates.jsonl", result["candidates"])
    write_jsonl(out_dir / "labels.jsonl", result["labels"])
    write_jsonl(out_dir / "bm25_predictions.jsonl", result["predictions"])
    manifest = {
        "dataset_name": "djinni_real_ranking_v1",
        "intended_use": "private_external_ranking_regression_only",
        "redistribution": "do_not_redistribute_local_snapshot",
        "sources": [
            {**JOB_SOURCE, "sha256": _sha256_file(args.jobs_parquet)},
            {**CANDIDATE_SOURCE, "sha256": _sha256_file(args.candidates_parquet)},
        ],
        "annotation": {
            "status": "expert_reviewed_by_codex",
            "is_human_verified": False,
            "rubric": annotations.get("rubric"),
            "limitations": annotations.get("limitations") or [],
        },
        "audit": result["audit"],
    }
    write_text(out_dir / "source_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
