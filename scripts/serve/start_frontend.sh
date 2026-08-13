#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PORT="${JOBMATCH_FRONTEND_PORT:-5173}"
if [[ ! -d frontend/node_modules ]]; then
  echo "frontend dependencies missing; run: npm ci --prefix frontend"
  exit 1
fi
npm run build --prefix frontend >/dev/null
echo "frontend serving on http://127.0.0.1:${PORT}"
python -m http.server "${PORT}" \
  --bind "${JOBMATCH_FRONTEND_HOST:-127.0.0.1}" \
  --directory frontend/dist
