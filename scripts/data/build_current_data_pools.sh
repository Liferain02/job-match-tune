#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

bash scripts/data/build_public_jd_candidate_pool.sh
bash scripts/data/build_jd_train_pool_supplemental.sh
bash scripts/data/build_jd_train_pool_weak_structured.sh
bash scripts/data/build_jd_train_pool_combined.sh
bash scripts/data/build_match_train_pool_synthetic.sh
bash scripts/data/build_resume_train_pool_synthetic.sh
bash scripts/data/build_resume_train_pool_from_sft.sh
bash scripts/data/build_resume_train_pool_bootstrap.sh
bash scripts/data/build_resume_train_pool_combined.sh
bash scripts/data/build_match_train_pool_combined.sh
bash scripts/data/report_data_readiness.sh
