#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.crawler.xiaomi_careers \
  --list-path 8-0-2 \
  --search-keyword 开发 \
  --search-keyword 算法 \
  --search-keyword 前端 \
  --search-keyword 后端 \
  --search-keyword 客户端 \
  --search-keyword 测试 \
  --search-keyword 数据 \
  --search-keyword Java \
  --out data/raw/xiaomi_jd_raw.jsonl \
  --db data/jobmatch_tune.sqlite3
