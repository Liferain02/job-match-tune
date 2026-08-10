from __future__ import annotations

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def load_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def content_hash(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    user = str(messages[1].get("content") or "") if len(messages) > 1 else ""
    assistant = str(messages[-1].get("content") or "") if messages else ""
    return hashlib.sha1(f"{user}\n---\n{assistant}".encode("utf-8")).hexdigest()


def source_group_key(row: dict[str, Any]) -> str:
    """Return the original-example identity used to keep template variants together."""
    return str(row.get("source_group") or row.get("source_id") or row.get("id") or content_hash(row))


def sample_task_rows(
    rows: list[dict[str, Any]],
    *,
    task_name: str,
    sample_count: int,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{task_name}:{split}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[source_group_key(row)].append(row)

    group_keys = list(groups)
    rng.shuffle(group_keys)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    # Take one row from every source before taking a second template variant.
    # This keeps large synthetic/template expansions from crowding out semantic
    # source diversity while preserving the configured task sample counts.
    selected: list[dict[str, Any]] = []
    depth = 0
    target = min(sample_count, len(rows))
    while len(selected) < target:
        added = False
        for group_key in group_keys:
            group_rows = groups[group_key]
            if depth < len(group_rows):
                selected.append(group_rows[depth])
                added = True
                if len(selected) >= target:
                    break
        if not added:
            break
        depth += 1
    output = []
    for row in selected:
        enriched = dict(row)
        meta = dict(enriched.get("meta") or {})
        meta["dataset_task"] = task_name
        meta["dataset_split"] = split
        enriched["meta"] = meta
        output.append(enriched)
    return output


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for row in rows:
        key = content_hash(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_split(
    task_cfg: dict[str, Any],
    *,
    split: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, float | int]]]:
    path_key = f"{split}_file"
    count_key = f"{split}_samples"
    built = []
    stats = {}
    diversity = {}
    for task_name, cfg in task_cfg.items():
        rows = list(read_jsonl(cfg[path_key]))
        requested = int(cfg[count_key])
        sampled = sample_task_rows(
            rows,
            task_name=task_name,
            sample_count=requested,
            seed=seed,
            split=split,
        )
        built.extend(sampled)
        stats[task_name] = len(sampled)
        unique_groups = len({source_group_key(row) for row in sampled})
        diversity[task_name] = {
            "rows": len(sampled),
            "unique_source_groups": unique_groups,
            "source_group_ratio": round(unique_groups / len(sampled), 4) if sampled else 0.0,
        }
    built = deduplicate_rows(built)
    rng = random.Random(f"{seed}:shuffle:{split}")
    rng.shuffle(built)
    return built, stats, diversity


def build_multitask_dataset(registry: dict[str, Any], section: str) -> dict[str, Any]:
    cfg = registry[section]
    seed = int(cfg.get("seed", 42))
    task_cfg = cfg["tasks"]
    train_rows, train_stats, train_diversity = build_split(task_cfg, split="train", seed=seed)
    valid_rows, valid_stats, valid_diversity = build_split(task_cfg, split="valid", seed=seed)
    write_jsonl(cfg["train_out"], train_rows)
    write_jsonl(cfg["valid_out"], valid_rows)
    return {
        "train_out": cfg["train_out"],
        "valid_out": cfg["valid_out"],
        "train_total": len(train_rows),
        "valid_total": len(valid_rows),
        "train_stats": train_stats,
        "valid_stats": valid_stats,
        "train_diversity": train_diversity,
        "valid_diversity": valid_diversity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/dataset_registry.yaml")
    parser.add_argument("--section", default="multitask_sft")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    result = build_multitask_dataset(registry, args.section)
    print(result)


if __name__ == "__main__":
    main()
