#!/usr/bin/env bash
set -euo pipefail
python -m jobmatch_tune.crawler.crawl_jd --config configs/crawl.yaml "$@"
