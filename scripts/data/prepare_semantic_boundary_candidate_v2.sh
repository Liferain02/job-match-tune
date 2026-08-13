#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHONPATH=src python -m jobmatch_tune.eval.build_semantic_boundary_candidate_v2 "$@"
