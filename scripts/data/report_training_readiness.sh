#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

bash scripts/data/report_resume_sft_profile.sh \
  --out outputs/eval_reports/resume_sft_profile.json

bash scripts/data/report_resume_privacy_readiness.sh \
  --inputs data/sft_resume/train.jsonl data/sft_resume/valid.jsonl data/sft_resume/test.jsonl \
  --out outputs/eval_reports/resume_privacy_readiness_report.json

bash scripts/data/report_preference_readiness.sh \
  --train data/preference/train.jsonl \
  --valid data/preference/valid.jsonl \
  --holdout data/eval/jd_manual_eval_50.jsonl \
  --out outputs/eval_reports/preference_readiness_report.json

bash scripts/data/report_preference_readiness.sh \
  --train data/preference_product_bootstrap/train.jsonl \
  --valid data/preference_product_bootstrap/valid.jsonl \
  --holdout data/eval/jd_manual_eval_50.jsonl \
  --out outputs/eval_reports/preference_product_bootstrap_readiness_report.json

bash scripts/data/report_data_readiness.sh \
  --out outputs/eval_reports/data_readiness_report.json
