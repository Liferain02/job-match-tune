#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_DIR="${JOBMATCH_RUNTIME_DIR:-${PROJECT_ROOT}/outputs/runtime}"
LOG_DIR="${JOBMATCH_LOG_DIR:-${PROJECT_ROOT}/outputs/logs}"
CONDA_ENV_NAME="${JOBMATCH_CONDA_ENV:-tune-demo}"
API_PORT="${JOBMATCH_API_PORT:-8000}"
FRONTEND_PORT="${JOBMATCH_FRONTEND_PORT:-5174}"
INFERENCE_BACKEND="${JOBMATCH_INFERENCE_BACKEND:-transformers}"
VLLM_PORT="${JOBMATCH_VLLM_PORT:-8010}"

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"
cd "${PROJECT_ROOT}"

is_running() {
  local pid_file="$1"
  local pid=""
  [[ -f "${pid_file}" ]] || return 1
  read -r pid < "${pid_file}" || return 1
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

port_is_free() {
  local port="$1"
  python - "${port}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}

wait_for_url() {
  local url="$1"
  local attempts="${2:-40}"
  local index
  for ((index = 0; index < attempts; index += 1)); do
    if curl --silent --fail --max-time 1 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

API_PID_FILE="${RUNTIME_DIR}/api.pid"
FRONTEND_PID_FILE="${RUNTIME_DIR}/frontend.pid"
VLLM_PID_FILE="${RUNTIME_DIR}/vllm.pid"

if is_running "${API_PID_FILE}" || is_running "${FRONTEND_PID_FILE}" || is_running "${VLLM_PID_FILE}"; then
  echo "项目已有进程在运行。请先执行：bash scripts/serve/stop_project.sh"
  exit 1
fi
rm -f "${API_PID_FILE}" "${FRONTEND_PID_FILE}" "${VLLM_PID_FILE}"

if ! port_is_free "${API_PORT}"; then
  echo "API 端口 ${API_PORT} 已被占用，请停止对应进程或设置 JOBMATCH_API_PORT。"
  exit 1
fi
if ! port_is_free "${FRONTEND_PORT}"; then
  echo "前端端口 ${FRONTEND_PORT} 已被占用，请停止对应进程或设置 JOBMATCH_FRONTEND_PORT。"
  exit 1
fi
if [[ "${INFERENCE_BACKEND}" == "vllm" ]] && ! port_is_free "${VLLM_PORT}"; then
  echo "vLLM 端口 ${VLLM_PORT} 已被占用，请停止对应进程或设置 JOBMATCH_VLLM_PORT。"
  exit 1
fi
if [[ "${INFERENCE_BACKEND}" != "transformers" && "${INFERENCE_BACKEND}" != "vllm" ]]; then
  echo "不支持的 JOBMATCH_INFERENCE_BACKEND：${INFERENCE_BACKEND}"
  exit 1
fi

RUNNER=()
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    RUNNER=("${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV_NAME}")
  elif command -v conda >/dev/null 2>&1; then
    RUNNER=(conda run --no-capture-output -n "${CONDA_ENV_NAME}")
  else
    echo "未找到 conda。请先激活 ${CONDA_ENV_NAME} 环境后重新执行。"
    exit 1
  fi
fi

if ! "${RUNNER[@]}" python -c "import fastapi, torch, transformers, uvicorn" >/dev/null 2>&1; then
  echo "环境 ${CONDA_ENV_NAME} 缺少服务依赖，请先执行：pip install -r requirements.txt"
  exit 1
fi
if [[ ! -d frontend/node_modules ]]; then
  echo "前端依赖未安装，请先执行：npm ci --prefix frontend"
  exit 1
fi
if [[ "${INFERENCE_BACKEND}" == "vllm" ]] && \
  ! "${RUNNER[@]}" python -c "import openai, vllm" >/dev/null 2>&1; then
  echo "环境 ${CONDA_ENV_NAME} 缺少 vLLM 服务依赖，请先安装 vllm 和 openai。"
  exit 1
fi

API_LOG="${LOG_DIR}/api.log"
FRONTEND_LOG="${LOG_DIR}/frontend.log"
VLLM_LOG="${LOG_DIR}/vllm.log"

export JOBMATCH_INFERENCE_BACKEND="${INFERENCE_BACKEND}"
export JOBMATCH_API_HOST="${JOBMATCH_API_HOST:-127.0.0.1}"
export JOBMATCH_API_PORT="${API_PORT}"
export JOBMATCH_FRONTEND_HOST="${JOBMATCH_FRONTEND_HOST:-127.0.0.1}"
export JOBMATCH_FRONTEND_PORT="${FRONTEND_PORT}"

if [[ "${INFERENCE_BACKEND}" == "vllm" ]]; then
  export JOBMATCH_VLLM_HOST="${JOBMATCH_VLLM_HOST:-127.0.0.1}"
  export JOBMATCH_VLLM_PORT="${VLLM_PORT}"
  export JOBMATCH_VLLM_BASE_URL="${JOBMATCH_VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
  nohup setsid "${RUNNER[@]}" bash "${SCRIPT_DIR}/start_vllm_server.sh" \
    >"${VLLM_LOG}" 2>&1 </dev/null &
  VLLM_PID=$!
  echo "${VLLM_PID}" > "${VLLM_PID_FILE}"

  if ! wait_for_url "${JOBMATCH_VLLM_BASE_URL}/models" 360; then
    echo "vLLM 启动失败，最近日志："
    tail -n 30 "${VLLM_LOG}" || true
    bash "${SCRIPT_DIR}/stop_project.sh" >/dev/null 2>&1 || true
    exit 1
  fi
fi

nohup setsid "${RUNNER[@]}" bash "${SCRIPT_DIR}/start_api.sh" \
  >"${API_LOG}" 2>&1 </dev/null &
API_PID=$!
echo "${API_PID}" > "${API_PID_FILE}"

nohup setsid "${RUNNER[@]}" bash "${SCRIPT_DIR}/start_frontend.sh" \
  >"${FRONTEND_LOG}" 2>&1 </dev/null &
FRONTEND_PID=$!
echo "${FRONTEND_PID}" > "${FRONTEND_PID_FILE}"

if ! wait_for_url "http://127.0.0.1:${API_PORT}/health"; then
  echo "API 启动失败，最近日志："
  tail -n 30 "${API_LOG}" || true
  bash "${SCRIPT_DIR}/stop_project.sh" >/dev/null 2>&1 || true
  exit 1
fi
if ! wait_for_url "http://127.0.0.1:${FRONTEND_PORT}/"; then
  echo "前端启动失败，最近日志："
  tail -n 30 "${FRONTEND_LOG}" || true
  bash "${SCRIPT_DIR}/stop_project.sh" >/dev/null 2>&1 || true
  exit 1
fi

echo "JobMatchTune 已启动："
echo "  前端：http://127.0.0.1:${FRONTEND_PORT}"
echo "  API： http://127.0.0.1:${API_PORT}"
echo "  后端：${INFERENCE_BACKEND}"
if [[ "${INFERENCE_BACKEND}" == "vllm" ]]; then
  echo "  vLLM：http://127.0.0.1:${VLLM_PORT}/v1"
fi
echo "  日志：${LOG_DIR}"
echo "停止：bash scripts/serve/stop_project.sh"
