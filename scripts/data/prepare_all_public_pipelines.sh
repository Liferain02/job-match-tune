#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

bash scripts/data/prepare_public_jd_pipeline.sh

if [ -f data/external/public_resume_exports/faircv_resume_parse.jsonl ] || [ -f data/external/public_resume_exports/resume_ner_train.jsonl ]; then
  bash scripts/data/prepare_public_resume_pipeline.sh
else
  echo "skip resume public pipeline: no public resume export files found"
fi

if [ -f data/external/public_match_exports/resume_job_fit_merged_v1.jsonl ] || [ -f data/external/public_match_exports/resume_job_description_fit.jsonl ]; then
  bash scripts/data/prepare_public_match_pipeline.sh
else
  echo "skip match public pipeline: no public match export files found"
fi

bash scripts/data/report_external_data_status.sh
bash scripts/data/report_data_readiness.sh
