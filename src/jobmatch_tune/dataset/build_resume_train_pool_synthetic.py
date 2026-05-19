from __future__ import annotations

import argparse

from jobmatch_tune.dataset.build_resume_sft_dataset import VARIANT_BUILDERS
from jobmatch_tune.utils.io import read_jsonl, write_jsonl


def build_rows(rows: list[dict]) -> list[dict]:
    built = []
    for row in rows:
        base_id = str(row.get("id") or "")
        label = row.get("label") or {}
        for variant_name, builder in VARIANT_BUILDERS:
            rendered = builder(row).strip()
            if not rendered:
                continue
            source_type = "synthetic_ocr_like" if variant_name == "ocr_like" else "synthetic_text"
            built.append(
                {
                    "id": f"{base_id}_{variant_name}",
                    "task": "resume_parse",
                    "source_type": source_type,
                    "text": rendered,
                    "label": label,
                    "meta": {"language": "zh", "generator": "resume_variant_render_v1"},
                }
            )
    return built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/resume_manual_train_pool.jsonl")
    parser.add_argument("--out", default="data/eval/resume_train_pool_synthetic.jsonl")
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    built = build_rows(rows)
    write_jsonl(args.out, built)
    print(f"synthetic={len(built)}")


if __name__ == "__main__":
    main()
