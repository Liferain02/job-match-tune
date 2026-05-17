#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/checkpoints

run_exp() {
  local gpu_id="$1"
  local name="$2"
  local lr="$3"
  local lora_r="$4"
  local lora_alpha="$5"

  CUDA_VISIBLE_DEVICES="${gpu_id}" python -m jobmatch_tune.train.train_lora \
    --config configs/train_qlora.yaml \
    --model_name_or_path models/Qwen3-1.7B \
    --output_dir "outputs/checkpoints/${name}" \
    --learning_rate "${lr}" \
    --lora_r "${lora_r}" \
    --lora_alpha "${lora_alpha}" \
    > "outputs/logs/${name}.log" 2>&1
}

run_exp 0 qwen3-1.7b-jobmatch-lr1e-4-r8 1.0e-4 8 16 &
pid0=$!

run_exp 1 qwen3-1.7b-jobmatch-lr2e-4-r8 2.0e-4 8 16 &
pid1=$!

run_exp 2 qwen3-1.7b-jobmatch-lr1e-4-r16 1.0e-4 16 32 &
pid2=$!

wait "${pid0}" "${pid1}" "${pid2}"
