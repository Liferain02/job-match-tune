# Incremental SFT Result (2026-05-13)

## Setup

- Base adapter: `outputs/checkpoints/qwen3-1.7b-dft-lr1e-4`
- Incremental adapter: `outputs/checkpoints/qwen3-1.7b-direction-incremental`
- Train dataset: `data/sft/train_incremental.jsonl` (`716 + 135 = 851`)
- Valid dataset: `data/sft/valid_incremental.jsonl` (`89 + 5 = 94`)
- Training mode:
  - continue training from existing adapter
  - `learning_rate=5e-5`
  - `num_train_epochs=1`

## Training summary

- `train_loss`: `0.6519`
- `eval_loss`: `0.3025`

## Holdout result

Holdout dataset: `data/eval/jd_manual_eval_50.jsonl`

### Before incremental SFT

- `json_valid_rate`: `0.98`
- `岗位方向 exact_match`: `0.918`
- `必备技能 F1`: `1.00`

### After incremental SFT

- `json_valid_rate`: `0.98`
- `岗位方向 exact_match`: `0.878`
- `必备技能 F1`: `0.939`

## Conclusion

This incremental SFT is **not** an improvement.

Main regression pattern:

1. Some `后端开发` titles with strong `推理/算法` wording were pulled toward `算法工程`
2. Some `测试开发` titles with `AI评测` wording were also pulled toward `算法工程`
3. Skills started to over-generate from domain phrases that were not part of the gold schema

## Decision

Do **not** replace the current serving adapter with `qwen3-1.7b-direction-incremental`.

Keep serving:

- `outputs/checkpoints/qwen3-1.7b-dft-lr1e-4`

Next useful step:

1. Expand hard-case labels again, especially:
   - `AI应用后台开发`
   - `应用算法工程师`
   - `AI评测 / 测试`
2. Tighten the skill schema so domain phrases are not treated as canonical skills
3. Rebuild a cleaner incremental dataset and retry with a smaller hard-case repeat factor
