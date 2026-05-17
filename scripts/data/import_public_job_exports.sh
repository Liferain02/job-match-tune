#!/usr/bin/env bash
set -euo pipefail

source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo

mkdir -p data/external/public_job_exports

# Public export seeds:
# - GitHub jhcoco/bosszp CSV
# - GitHub WorkAggregation CSV
# - Hugging Face open-apply-jobs Greenhouse parquet shard
# Extend configs/public_job_sources.yaml to add more sources without changing
# the downstream import pipeline.
curl --compressed -L \
  "https://raw.githubusercontent.com/jhcoco/bosszp/master/%E5%85%A8%E5%9B%BD-%E7%83%AD%E9%97%A8%E5%9F%8E%E5%B8%82%E5%B2%97%E4%BD%8D%E6%95%B0%E6%8D%AE.csv" \
  -o data/external/public_job_exports/jhcoco_bosszp.csv

curl --compressed -L \
  "https://raw.githubusercontent.com/xming521/WorkAggregation/master/data/test.csv" \
  -o data/external/public_job_exports/workaggregation_test.csv

curl --compressed -L \
  "https://huggingface.co/datasets/edwarddgao/open-apply-jobs/resolve/main/data/date=2026-05-15/source=greenhouse/52db4a0a5e67446bb34dceaf6c9f6c43-0.parquet" \
  -o data/external/public_job_exports/open_apply_greenhouse_2026-05-15.parquet

curl --compressed -L \
  "https://huggingface.co/datasets/edwarddgao/open-apply-jobs/resolve/main/data/date=2026-05-15/source=ashby/52db4a0a5e67446bb34dceaf6c9f6c43-0.parquet" \
  -o data/external/public_job_exports/open_apply_ashby_2026-05-15.parquet

curl --compressed -L \
  "https://huggingface.co/datasets/edwarddgao/open-apply-jobs/resolve/main/data/date=2026-05-15/source=lever/52db4a0a5e67446bb34dceaf6c9f6c43-0.parquet" \
  -o data/external/public_job_exports/open_apply_lever_2026-05-15.parquet

PYTHONPATH=src python -m jobmatch_tune.crawler.import_public_job_data \
  --sources configs/public_job_sources.yaml \
  --out data/raw/public_job_datasets_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
