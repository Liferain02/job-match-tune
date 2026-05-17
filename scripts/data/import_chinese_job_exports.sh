#!/usr/bin/env bash
set -euo pipefail

source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo

mkdir -p data/external/public_job_exports

curl --compressed -L \
  "https://huggingface.co/datasets/wangzihaogithub/job-educational-parser-dataset-08-0-0805/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet" \
  -o data/external/public_job_exports/job_educational_train_2026-05-17.parquet

curl --compressed -L \
  "https://huggingface.co/datasets/wangzihaogithub/job-educational-parser-dataset-08-0-0805/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet" \
  -o data/external/public_job_exports/job_educational_validation_2026-05-17.parquet

curl --compressed -L \
  "https://huggingface.co/datasets/wangzihaogithub/job-educational-parser-dataset-08-0-0805/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet" \
  -o data/external/public_job_exports/job_educational_test_2026-05-17.parquet

PYTHONPATH=src python -m jobmatch_tune.crawler.import_public_job_data \
  --sources configs/public_job_sources_zh_large.yaml \
  --out data/raw/public_job_datasets_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
