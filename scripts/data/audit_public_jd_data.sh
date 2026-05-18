#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHONPATH=src python -m jobmatch_tune.eval.audit_public_jd_data "$@"
