#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHONPATH=src python -m jobmatch_tune.dataset.build_public_jd_candidate_pool "$@"
