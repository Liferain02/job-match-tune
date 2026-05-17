#!/usr/bin/env bash
set -euo pipefail

bash scripts/data/refresh_tencent_data.sh auto
bash scripts/data/refresh_baidu_data.sh
bash scripts/data/refresh_jd_data.sh
bash scripts/data/refresh_moka_data.sh
bash scripts/data/rebuild_data_pipeline.sh
