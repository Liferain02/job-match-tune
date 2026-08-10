#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

bash scripts/data/build_public_jd_candidate_pool.sh
bash scripts/data/build_jd_train_pool_supplemental.sh
bash scripts/data/build_jd_train_pool_weak_structured.sh
bash scripts/data/build_jd_train_pool_combined.sh
bash scripts/data/build_resume_eval_dataset.sh
bash scripts/data/build_resume_train_pool_synthetic.sh
bash scripts/data/build_resume_train_pool_bootstrap.sh
bash scripts/data/build_resume_train_pool_combined.sh
bash scripts/data/build_match_eval_dataset.sh
bash scripts/data/build_match_train_pool_synthetic.sh
bash scripts/data/build_match_train_pool_combined.sh
bash scripts/data/build_jd_quality_sft_dataset.sh
bash scripts/data/build_resume_sft_dataset.sh
bash scripts/data/build_match_sft_dataset.sh
bash scripts/data/build_multitask_sft_dataset.sh
bash scripts/data/build_preference_bootstrap_dataset.sh
bash scripts/data/build_product_preference_bootstrap_dataset.sh
bash scripts/data/report_training_readiness.sh
