from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.utils.io import write_jsonl, write_text


CHINESE_LABEL_KEYS = {
    "L": "语言能力",
    "K": "知识技能",
    "S": "专业技能",
    "T": "通用能力",
}
MARKUP_PATTERN = re.compile(r"@@(.*?)##([LKST])")
SPLIT_PRIORITY = ("test", "dev", "train")
MAX_INVALID_SOURCE_RATE = 0.02


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, output_path: str | Path, expected_sha256: str) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and file_sha256(output) == expected_sha256:
        return {"path": str(output), "sha256": expected_sha256, "downloaded": False}

    partial = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "job-match-tune/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as target:
        shutil.copyfileobj(response, target)
    actual_sha256 = file_sha256(partial)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"sha256 mismatch for {output}: expected={expected_sha256}, actual={actual_sha256}"
        )
    partial.replace(output)
    return {"path": str(output), "sha256": actual_sha256, "downloaded": True}


def parse_chinese_tagged_sentence(text: str, tagged_text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    plain_parts: list[str] = []
    plain_length = 0
    cursor = 0
    for match in MARKUP_PATTERN.finditer(tagged_text):
        prefix = tagged_text[cursor : match.start()]
        if "@@" in prefix or "##" in prefix:
            raise ValueError("malformed LKST markup")
        plain_parts.append(prefix)
        plain_length += len(prefix)
        span_text, label = match.groups()
        if not span_text:
            raise ValueError("empty LKST span")
        start = plain_length
        plain_parts.append(span_text)
        plain_length += len(span_text)
        spans.append(
            {
                "start": start,
                "end": plain_length,
                "label": label,
                "text": span_text,
            }
        )
        cursor = match.end()
    suffix = tagged_text[cursor:]
    if "@@" in suffix or "##" in suffix:
        raise ValueError("malformed LKST markup")
    plain_parts.append(suffix)
    reconstructed = "".join(plain_parts)
    if reconstructed != text:
        raise ValueError("tagged output does not reconstruct the input sentence")
    return spans


def _label_from_spans(spans: list[dict[str, Any]]) -> dict[str, list[str]]:
    label = {value: [] for value in CHINESE_LABEL_KEYS.values()}
    for span in spans:
        key = CHINESE_LABEL_KEYS[span["label"]]
        if span["text"] not in label[key]:
            label[key].append(span["text"])
    label["全部技能"] = [
        value
        for key in CHINESE_LABEL_KEYS.values()
        for value in label[key]
    ]
    return label


def convert_chinese_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    source_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    converted: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        text = str(row.get("input") or "")
        tagged_text = str(row.get("output") or "")
        meta = row.get("meta") or {}
        try:
            if not text or not tagged_text:
                raise ValueError("missing input or output")
            spans = parse_chinese_tagged_sentence(text, tagged_text)
        except ValueError as exc:
            errors.append(f"{split}:{index}:{exc}")
            continue
        upstream_id = str(meta.get("id") or row.get("id") or index)
        job_id = str(meta.get("job_id") or meta.get("global_id") or upstream_id)
        converted.append(
            {
                "id": f"{dataset_name}_{split}_{upstream_id}",
                "task": "jd_skill_extract",
                "source_group": f"{dataset_name}:{job_id}",
                "text": text,
                "spans": spans,
                "label": _label_from_spans(spans),
                "meta": {
                    "dataset": dataset_name,
                    "split": split,
                    "language": source_cfg["language"],
                    "annotation": source_cfg["annotation"],
                    "license_status": source_cfg["license_status"],
                    "intended_usage": source_cfg["intended_usage"],
                    "upstream_id": upstream_id,
                    "job_id": job_id,
                    "source_domain": str(meta.get("source_domain") or meta.get("source_main") or ""),
                },
            }
        )
    return converted, errors


def _token_offsets(tokens: list[str]) -> tuple[str, list[int]]:
    starts: list[int] = []
    parts: list[str] = []
    length = 0
    for index, token in enumerate(tokens):
        if index:
            parts.append(" ")
            length += 1
        starts.append(length)
        parts.append(token)
        length += len(token)
    return "".join(parts), starts


def bio_spans(
    tokens: list[str], tags: list[str], label: str, starts: list[int]
) -> list[dict[str, Any]]:
    if len(tokens) != len(tags):
        raise ValueError("BIO tag count does not match token count")
    spans: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        tag = str(tags[index])
        if tag == "O":
            index += 1
            continue
        if tag != "B":
            raise ValueError(f"invalid BIO transition at token {index}: {tag}")
        end_index = index + 1
        while end_index < len(tokens) and str(tags[end_index]) == "I":
            end_index += 1
        start = starts[index]
        end = starts[end_index - 1] + len(tokens[end_index - 1])
        spans.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": " ".join(tokens[index:end_index]),
            }
        )
        index = end_index
    return spans


def convert_skillspan_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    source_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    converted: list[dict[str, Any]] = []
    errors: list[str] = []
    sentence_indices: Counter[tuple[str, str]] = Counter()
    for index, row in enumerate(rows):
        tokens = [str(token) for token in row.get("tokens") or []]
        source = str(row.get("source") or "unknown")
        posting_id = str(row.get("idx") or "unknown")
        sentence_key = (source, posting_id)
        sentence_index = sentence_indices[sentence_key]
        sentence_indices[sentence_key] += 1
        try:
            if not tokens:
                raise ValueError("missing tokens")
            text, starts = _token_offsets(tokens)
            spans = bio_spans(tokens, list(row.get("tags_skill") or []), "soft_skill", starts)
            spans.extend(
                bio_spans(tokens, list(row.get("tags_knowledge") or []), "knowledge", starts)
            )
            spans.sort(key=lambda item: (item["start"], item["end"], item["label"]))
        except ValueError as exc:
            errors.append(f"{split}:{index}:{exc}")
            continue
        converted.append(
            {
                "id": f"{dataset_name}_{split}_{source}_{posting_id}_{sentence_index}",
                "task": "jd_skill_extract",
                # Upstream documents are split before idx is restarted, so the split
                # is part of the only reliable document identity for this dataset.
                "source_group": f"{dataset_name}:{split}:{source}:{posting_id}",
                "text": text,
                "spans": spans,
                "label": {
                    "软技能": [span["text"] for span in spans if span["label"] == "soft_skill"],
                    "知识技能": [span["text"] for span in spans if span["label"] == "knowledge"],
                    "全部技能": [span["text"] for span in spans],
                },
                "meta": {
                    "dataset": dataset_name,
                    "split": split,
                    "language": source_cfg["language"],
                    "annotation": source_cfg["annotation"],
                    "license": source_cfg.get("license", ""),
                    "license_status": source_cfg["license_status"],
                    "intended_usage": source_cfg["intended_usage"],
                    "deidentified": bool(source_cfg.get("deidentified")),
                    "posting_id": posting_id,
                    "source_domain": source,
                },
            }
        )
    return converted, errors


def read_source_rows(path: str | Path, processor: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if processor == "chinese_lkst":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"expected JSON array: {file_path}")
        return [row for row in payload if isinstance(row, dict)]
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _text_hash(row: dict[str, Any]) -> str:
    normalized = " ".join(str(row.get("text") or "").split()).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def isolate_splits(
    rows_by_split: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    kept: dict[str, list[dict[str, Any]]] = {
        split: [] for split in dict.fromkeys((*SPLIT_PRIORITY, *rows_by_split))
    }
    seen_text: set[str] = set()
    seen_groups: set[str] = set()
    stats: Counter[str] = Counter()
    for split in SPLIT_PRIORITY:
        split_rows = rows_by_split.get(split, [])
        local_text: set[str] = set()
        local_groups = {str(row.get("source_group") or "") for row in split_rows}
        blocked_groups = local_groups & seen_groups
        for row in split_rows:
            text_hash = _text_hash(row)
            source_group = str(row.get("source_group") or "")
            if source_group in blocked_groups:
                stats["cross_split_source_group_dropped"] += 1
                continue
            if text_hash in seen_text:
                stats["cross_split_text_dropped"] += 1
                continue
            if text_hash in local_text:
                stats["within_split_text_dropped"] += 1
                continue
            kept[split].append(row)
            local_text.add(text_hash)
        seen_text.update(local_text)
        seen_groups.update(str(row.get("source_group") or "") for row in kept[split])
    return kept, dict(stats)


def audit_dataset(
    source_name: str,
    source_cfg: dict[str, Any],
    raw_counts: dict[str, int],
    rows_by_split: dict[str, list[dict[str, Any]]],
    errors: list[str],
    isolation_stats: dict[str, int],
) -> dict[str, Any]:
    split_counts = {split: len(rows) for split, rows in rows_by_split.items()}
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    label_counts: Counter[str] = Counter()
    source_groups_by_split: dict[str, set[str]] = {}
    text_splits: defaultdict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        source_groups_by_split[split] = {str(row.get("source_group") or "") for row in rows}
        for row in rows:
            label_counts.update(str(span.get("label") or "") for span in row.get("spans") or [])
            text_splits[_text_hash(row)].add(split)
    source_group_overlap = 0
    splits = list(source_groups_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            source_group_overlap += len(
                source_groups_by_split[left] & source_groups_by_split[right]
            )
    cross_split_text = sum(1 for split_names in text_splits.values() if len(split_names) > 1)
    negative_rows = sum(1 for row in all_rows if not row.get("spans"))
    isolation_drops = sum(isolation_stats.values())
    source_row_count = len(all_rows) + isolation_drops + len(errors)
    invalid_source_rate = len(errors) / source_row_count if source_row_count else 0.0
    processed_format_ready = source_group_overlap == 0 and cross_split_text == 0
    source_quality_ready = invalid_source_rate <= MAX_INVALID_SOURCE_RATE
    return {
        "dataset": source_name,
        "display_name": source_cfg.get("display_name", source_name),
        "source_page": source_cfg.get("source_page", ""),
        "paper_url": source_cfg.get("paper_url", ""),
        "language": source_cfg.get("language", ""),
        "annotation": source_cfg.get("annotation", ""),
        "license": source_cfg.get("license", ""),
        "license_status": source_cfg.get("license_status", "unconfirmed"),
        "intended_usage": source_cfg.get("intended_usage", ""),
        "raw_counts": raw_counts,
        "processed_counts": split_counts,
        "total_processed": len(all_rows),
        "unique_source_groups": len(
            {str(row.get("source_group") or "") for row in all_rows}
        ),
        "span_count": sum(label_counts.values()),
        "label_counts": dict(label_counts),
        "negative_rows": negative_rows,
        "negative_row_rate": round(negative_rows / len(all_rows), 4) if all_rows else 0.0,
        "invalid_rows": len(errors),
        "invalid_source_rate": round(invalid_source_rate, 4),
        "max_invalid_source_rate": MAX_INVALID_SOURCE_RATE,
        "error_examples": errors[:20],
        "isolation": isolation_stats,
        "cross_split_source_groups": source_group_overlap,
        "cross_split_texts": cross_split_text,
        "processed_format_ready": processed_format_ready,
        "source_quality_ready": source_quality_ready,
        "ready_for_internal_evaluation": processed_format_ready
        and source_quality_ready
        and split_counts.get("test", 0) > 0,
        "ready_for_training": processed_format_ready
        and source_quality_ready
        and source_cfg.get("license_status") == "confirmed"
        and source_cfg.get("intended_usage") not in {"internal_evaluation_only", "external_evaluation"},
    }


def prepare_sources(
    manifest_path: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8")) or {}
    raw_root = Path(raw_dir)
    output_root = Path(output_dir)
    reports = {}
    downloads = {}
    for source_name, source_cfg in (manifest.get("sources") or {}).items():
        processor = str(source_cfg["processor"])
        rows_by_split: dict[str, list[dict[str, Any]]] = {}
        raw_counts = {}
        source_errors: list[str] = []
        downloads[source_name] = {}
        for split, file_cfg in source_cfg["files"].items():
            suffix = ".jsonl" if processor == "skillspan_bio" else ".json"
            raw_path = raw_root / source_name / f"{split}{suffix}"
            downloads[source_name][split] = download_verified(
                str(file_cfg["url"]), raw_path, str(file_cfg["sha256"])
            )
            raw_rows = read_source_rows(raw_path, processor)
            raw_counts[split] = len(raw_rows)
            if not file_cfg.get("annotated", True):
                continue
            if processor == "chinese_lkst":
                converted, errors = convert_chinese_rows(
                    raw_rows,
                    dataset_name=source_name,
                    split=split,
                    source_cfg=source_cfg,
                )
            elif processor == "skillspan_bio":
                converted, errors = convert_skillspan_rows(
                    raw_rows,
                    dataset_name=source_name,
                    split=split,
                    source_cfg=source_cfg,
                )
            else:
                raise ValueError(f"unsupported processor: {processor}")
            rows_by_split[split] = converted
            source_errors.extend(errors)
        isolated_rows, isolation_stats = isolate_splits(rows_by_split)
        for split, rows in isolated_rows.items():
            write_jsonl(output_root / source_name / f"{split}.jsonl", rows)
        reports[source_name] = audit_dataset(
            source_name,
            source_cfg,
            raw_counts,
            isolated_rows,
            source_errors,
            isolation_stats,
        )
    return {
        "manifest": str(manifest_path),
        "raw_dir": str(raw_root),
        "output_dir": str(output_root),
        "downloads": downloads,
        "datasets": reports,
        "summary": {
            "datasets": len(reports),
            "ready_for_internal_evaluation": [
                name for name, report in reports.items() if report["ready_for_internal_evaluation"]
            ],
            "ready_for_training": [
                name for name, report in reports.items() if report["ready_for_training"]
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/external_skill_gold_sources.yaml")
    parser.add_argument("--raw-dir", default="data/external/skill_gold_raw")
    parser.add_argument("--out-dir", default="data/external/skill_gold_processed")
    parser.add_argument("--report", default="outputs/eval_reports/external_skill_gold_report.json")
    args = parser.parse_args()

    report = prepare_sources(args.manifest, args.raw_dir, args.out_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    write_text(args.report, rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
