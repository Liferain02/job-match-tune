#!/usr/bin/env bash
set -euo pipefail

source /share/home/lifr/miniconda3/etc/profile.d/conda.sh
conda activate tune-demo

cd /share/home/lifr/workspace/code/job-match-tune
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-models/Qwen3-14B}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/checkpoints/qwen3-14b-jobmatch-product-dpo}"
TAG="${TAG:-product_dpo}"
LOAD_4BIT_FLAG="${LOAD_4BIT_FLAG:---load-4bit}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_REGRESSION="${MAX_REGRESSION:-0.005}"

PYTHONPATH=src python -m jobmatch_tune.eval.run_manual_eval \
  --dataset data/eval/jd_manual_eval_50.jsonl \
  --model "$MODEL_PATH" \
  --adapter "$ADAPTER_PATH" \
  --out "outputs/eval_reports/manual_eval_50_${TAG}_report.json" \
  --predictions-out "outputs/eval_reports/manual_eval_50_${TAG}_predictions.jsonl" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  $LOAD_4BIT_FLAG

PYTHONPATH=src python -m jobmatch_tune.eval.run_resume_pipeline_eval \
  --dataset data/eval/resume_manual_eval_text_seed.jsonl \
  --model "$MODEL_PATH" \
  --adapter "$ADAPTER_PATH" \
  --out "outputs/eval_reports/resume_pipeline_eval_${TAG}_report.json" \
  --predictions-out "outputs/eval_reports/resume_pipeline_eval_${TAG}_predictions.jsonl" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  $LOAD_4BIT_FLAG

PYTHONPATH=src python -m jobmatch_tune.eval.run_match_eval \
  --dataset data/eval/match_manual_eval_seed.jsonl \
  --model "$MODEL_PATH" \
  --adapter "$ADAPTER_PATH" \
  --out "outputs/eval_reports/match_eval_${TAG}_report.json" \
  --predictions-out "outputs/eval_reports/match_eval_${TAG}_predictions.jsonl" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  $LOAD_4BIT_FLAG

PYTHONPATH=src python -m jobmatch_tune.eval.report_product_readiness \
  --jd-report "outputs/eval_reports/manual_eval_50_${TAG}_report.json" \
  --resume-report "outputs/eval_reports/resume_pipeline_eval_${TAG}_report.json" \
  --match-report "outputs/eval_reports/match_eval_${TAG}_report.json" \
  --out "outputs/eval_reports/product_readiness_${TAG}_report.json"

if [[ -n "${BASELINE_TAG:-}" || -n "${BASELINE_JD_REPORT:-}" ]]; then
  BASELINE_JD_REPORT="${BASELINE_JD_REPORT:-outputs/eval_reports/manual_eval_50_${BASELINE_TAG}_report.json}"
  BASELINE_RESUME_REPORT="${BASELINE_RESUME_REPORT:-outputs/eval_reports/resume_pipeline_eval_${BASELINE_TAG}_report.json}"
  BASELINE_MATCH_REPORT="${BASELINE_MATCH_REPORT:-outputs/eval_reports/match_eval_${BASELINE_TAG}_report.json}"

  PYTHONPATH=src python -m jobmatch_tune.eval.compare_product_reports \
    --candidate-jd-report "outputs/eval_reports/manual_eval_50_${TAG}_report.json" \
    --candidate-resume-report "outputs/eval_reports/resume_pipeline_eval_${TAG}_report.json" \
    --candidate-match-report "outputs/eval_reports/match_eval_${TAG}_report.json" \
    --baseline-jd-report "$BASELINE_JD_REPORT" \
    --baseline-resume-report "$BASELINE_RESUME_REPORT" \
    --baseline-match-report "$BASELINE_MATCH_REPORT" \
    --max-regression "$MAX_REGRESSION" \
    --out "outputs/eval_reports/product_regression_${TAG}_vs_${BASELINE_TAG:-baseline}_report.json"
fi
