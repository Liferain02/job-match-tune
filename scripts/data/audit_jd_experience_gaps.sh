#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHONPATH=src python -m jobmatch_tune.eval.audit_jd_experience_gaps "$@"
