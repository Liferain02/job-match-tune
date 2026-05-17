#!/usr/bin/env bash
set -euo pipefail

CRAWL_MODE="${1:-auto}"

run_rebuild() {
  PYTHONPATH=src bash scripts/data/rebuild_data_pipeline.sh
}

run_crawl() {
  PYTHONPATH=src python -m jobmatch_tune.crawler.tencent_careers \
    --keywords-file configs/tencent_keywords.txt \
    --limit 5000 \
    --page-size 50 \
    --max-pages 40 \
    --interval-seconds 0.4 \
    --retries 3 \
    --category 技术 \
    --out data/raw/tencent_jd_raw.jsonl \
    --db data/jobmatch_tune.sqlite3
}

case "${CRAWL_MODE}" in
  auto)
    if run_crawl; then
      echo "crawl finished, rebuilding downstream datasets"
      run_rebuild
    else
      echo "crawl skipped due to network/API failure, rebuilding from existing raw data"
      run_rebuild
    fi
    ;;
  crawl)
    run_crawl
    run_rebuild
    ;;
  rebuild)
    run_rebuild
    ;;
  *)
    echo "usage: bash scripts/data/refresh_tencent_data.sh [auto|crawl|rebuild]" >&2
    exit 1
    ;;
esac
