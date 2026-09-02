#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHONPATH=src python -m jobmatch_tune.dataset.build_public_jd_candidate_pool
if [[ -f data/external/faircv/data/resumes_template.json ]]; then
  PYTHONPATH=src python -m jobmatch_tune.dataset.import_public_training_data
else
  echo "中文技术简历模板源未下载，保留现有最小数据池" >&2
fi

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
