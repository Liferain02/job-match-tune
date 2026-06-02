#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m jobmatch_tune.eval.report_resume_privacy_readiness "$@"
