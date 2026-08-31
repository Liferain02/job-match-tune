#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHONPATH=src python -m jobmatch_tune.dataset.build_public_jd_candidate_pool
PYTHONPATH=src python -m jobmatch_tune.dataset.build_jd_train_pool_combined
PYTHONPATH=src python -m jobmatch_tune.eval.build_resume_eval_dataset
PYTHONPATH=src python -m jobmatch_tune.dataset.build_resume_train_pool_combined
PYTHONPATH=src python -m jobmatch_tune.eval.build_match_eval_dataset
PYTHONPATH=src python -m jobmatch_tune.dataset.build_match_train_pool_combined
PYTHONPATH=src python -m jobmatch_tune.dataset.build_jd_strict_plus_sft_dataset
PYTHONPATH=src python -m jobmatch_tune.dataset.build_resume_sft_dataset
PYTHONPATH=src python -m jobmatch_tune.dataset.build_match_sft_dataset
PYTHONPATH=src python -m jobmatch_tune.dataset.build_multitask_sft_dataset
bash scripts/data/report_training_readiness.sh
