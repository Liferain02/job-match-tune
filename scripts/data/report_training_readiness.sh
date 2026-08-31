#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHONPATH=src python -m jobmatch_tune.eval.report_resume_sft_profile \
  --out outputs/eval_reports/resume_sft_profile.json

PYTHONPATH=src python -m jobmatch_tune.eval.report_resume_privacy_readiness \
  --inputs data/sft_resume/train.jsonl data/sft_resume/valid.jsonl data/sft_resume/test.jsonl \
  --out outputs/eval_reports/resume_privacy_readiness_report.json

PYTHONPATH=src python -m jobmatch_tune.eval.report_preference_readiness \
  --train data/preference/train.jsonl \
  --valid data/preference/valid.jsonl \
  --holdout data/eval/jd_manual_eval_50.jsonl \
  --out outputs/eval_reports/preference_readiness_report.json

PYTHONPATH=src python -m jobmatch_tune.eval.audit_match_gold \
  --out outputs/eval_reports/match_gold_audit.json

if [[ -f data/private/djinni_real_ranking_v1/labels.jsonl && \
      -f data/private/djinni_real_ranking_v1/bm25_predictions.jsonl ]]; then
  PYTHONPATH=src python -m jobmatch_tune.eval.run_match_ranking_eval \
    --labels data/private/djinni_real_ranking_v1/labels.jsonl \
    --predictions data/private/djinni_real_ranking_v1/bm25_predictions.jsonl \
    --out outputs/eval_reports/djinni_real_ranking_bm25.json
fi

PYTHONPATH=src python -m jobmatch_tune.eval.report_data_readiness \
  --out outputs/eval_reports/data_readiness_report.json
