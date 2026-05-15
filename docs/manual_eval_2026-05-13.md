# Manual Eval Notes (2026-05-13)

## Dataset

- Manual gold dataset: `data/eval/jd_manual_eval_50.jsonl`
- Scope: 50 JD samples
- Focus: `前端开发 / 后端开发 / 测试开发 / 算法工程 / AI应用开发` boundary cases

## What Was Fixed

1. Added a reproducible builder for the 50-sample manual eval set.
2. Rebuilt `data/sft/` so `经验要求` / `学历要求` are no longer empty.
3. Upgraded postprocessing so it can:
   - split `经验要求 / 学历要求 / 任职要求 / 加分项` out of `核心职责`
   - infer skills from context
   - canonicalize `岗位方向` with rule-based normalization
4. Added context-aware direction normalization to use original JD text when available.

## Current Quality

### Best stable result on the 50-sample set

- `json_valid_rate`: `0.98`
- `核心职责 F1`: `1.00`
- `必备技能 F1`: `1.00`
- `加分项 F1`: `1.00`
- `经验要求 exact_match`: `1.00`
- `学历要求 exact_match`: `1.00`
- `岗位方向 exact_match`: `0.936`

### Remaining direction errors

The remaining errors are not random. They cluster around:

- `AI应用开发` vs `算法工程`
- `AI应用后台开发工程师` vs `算法工程`
- `应用算法工程师 / 算法应用` style titles
- `Agentic Engineer` vs `AI应用开发`

This means the current bottleneck is taxonomy boundary, not model format following.

## Decision

Do **not** start full retraining now.

Recommended next step:

1. Freeze a clearer `岗位方向` labeling policy for boundary titles:
   - `应用算法工程师`
   - `算法应用`
   - `AI应用后台开发`
   - `Agent 应用开发`
2. Add another 20-30 targeted direction samples only for the remaining boundary cases.
3. Keep the current serving adapter and prefer rule upgrades over more incremental SFT for now.

## Why

`职责 / 技能 / 经验 / 学历` are already stable enough.  
The latest rule-only upgrade beat both incremental SFT runs on `岗位方向`, so current effort should stay on labeling policy and rule refinement rather than more training.
