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
