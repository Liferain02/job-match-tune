from __future__ import annotations

import hashlib
import random
import unicodedata
from collections import defaultdict
from typing import Any


def normalized_input_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    compact = "".join(
        character
        for character in normalized
        if unicodedata.category(character)[0] in {"L", "N"}
    )
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()


def split_linked_samples(
    samples: list[dict[str, Any]], train_ratio: float, valid_ratio: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    sample_keys: list[str] = []
    for index, sample in enumerate(samples):
        group = str(sample.get("source_group") or sample.get("id") or index)
        messages = sample.get("messages") or []
        user_text = str(messages[1].get("content") or "") if len(messages) > 1 else ""
        group_key = f"group:{group}"
        find(group_key)
        for linked_group in sample.get("linked_source_groups") or []:
            union(group_key, f"linked:{linked_group}")
        if user_text:
            input_key = f"input:{normalized_input_hash(user_text)}"
            union(group_key, input_key)
        sample_keys.append(group_key)

    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample, key in zip(samples, sample_keys, strict=True):
        components[find(key)].append(sample)

    component_keys = list(components)
    random.Random(seed).shuffle(component_keys)
    count = len(component_keys)
    if count < 3:
        all_rows = [row for key in component_keys for row in components[key]]
        return {"train": all_rows, "valid": all_rows[:1], "test": all_rows[:1]}

    valid_count = max(1, int(count * valid_ratio))
    test_count = max(1, count - int(count * train_ratio) - valid_count)
    train_count = max(1, count - valid_count - test_count)
    train_keys = component_keys[:train_count]
    valid_keys = component_keys[train_count : train_count + valid_count]
    test_keys = component_keys[train_count + valid_count :]
    return {
        "train": [row for key in train_keys for row in components[key]],
        "valid": [row for key in valid_keys for row in components[key]],
        "test": [row for key in test_keys for row in components[key]],
    }
