#!/usr/bin/env bash

set -euo pipefail

if [[ "${JOBMATCH_ALLOW_DPO:-0}" != "1" ]]; then
  echo "DPO 已暂停：当前先完成独立 Match Gold、错误分析和三任务回归。" >&2
  echo "仅在有新的人工 preference 且明确决定恢复实验时，设置 JOBMATCH_ALLOW_DPO=1。" >&2
  exit 2
fi
