#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHONPATH=src python -m jobmatch_tune.dataset.prepare_external_skill_gold "$@"
