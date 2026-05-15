#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/logs outputs/checkpoints

run_exp() {
  local gpu_id="$1"
  local name="$2"
  shift 2

  CUDA_VISIBLE_DEVICES="${gpu_id}" python -m jobmatch_tune.train.train_lora \
    --config configs/train_qlora.yaml \
    --model_name_or_path models/Qwen3-1.7B \
    --output_dir "outputs/checkpoints/${name}" \
    "$@" \
    > "outputs/logs/${name}.log" 2>&1
}

run_exp 0 qwen3-1.7b-sdpa-lr1e-4 \
  --learning_rate 1.0e-4 &
pid0=$!

run_exp 1 qwen3-1.7b-dft-lr1e-4 \
  --learning_rate 1.0e-4 \
  --loss_type dft &
pid1=$!

run_exp 2 qwen3-1.7b-liger-lr1e-4 \
  --learning_rate 1.0e-4 \
  --use_liger_kernel &
pid2=$!

wait "${pid0}" "${pid1}" "${pid2}"

echo "All modern experiments finished. Logs:"
echo "  outputs/logs/qwen3-1.7b-sdpa-lr1e-4.log"
echo "  outputs/logs/qwen3-1.7b-dft-lr1e-4.log"
echo "  outputs/logs/qwen3-1.7b-liger-lr1e-4.log"
