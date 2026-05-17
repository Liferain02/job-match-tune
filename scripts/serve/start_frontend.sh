#!/usr/bin/env bash
set -euo pipefail

python -m http.server "${JOBMATCH_FRONTEND_PORT:-5173}" --directory frontend
