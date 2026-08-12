#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

TARGET="${1:-data/private/match_gold.jsonl}"
if [[ -e "$TARGET" ]]; then
  echo "保留已有人工审核文件，不覆盖：$TARGET"
  exit 0
fi

bash scripts/data/build_match_eval_dataset.sh
mkdir -p "$(dirname "$TARGET")"
cp data/eval/match_gold_review_candidates.jsonl "$TARGET"
echo "已创建人工审核文件：$TARGET"
