#!/usr/bin/env bash

run_training_readiness_gate() {
  if [[ "${SKIP_TRAINING_READINESS_GATE:-0}" == "1" ]]; then
    echo "SKIP_TRAINING_READINESS_GATE=1, skip training readiness gate"
    return 0
  fi
  bash scripts/data/report_training_readiness.sh >/tmp/jobmatch_training_readiness.log
  bash scripts/data/assert_training_readiness.sh \
    --report outputs/eval_reports/data_readiness_report.json
}
