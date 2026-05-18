#!/usr/bin/env bash
set -euo pipefail

pick_port() {
  local start_port="$1"
  python - "$start_port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
for candidate in range(port, port + 20):
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", candidate))
    except OSError:
        sock.close()
        continue
    sock.close()
    print(candidate)
    raise SystemExit(0)
raise SystemExit(1)
PY
}

PORT="$(pick_port "${JOBMATCH_FRONTEND_PORT:-5173}")"
echo "frontend serving on http://127.0.0.1:${PORT}"
python -m http.server "${PORT}" --directory frontend
