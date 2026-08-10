#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_DIR="${JOBMATCH_RUNTIME_DIR:-${PROJECT_ROOT}/outputs/runtime}"

stop_process_group() {
  local name="$1"
  local pid_file="$2"
  local pid=""
  local command=""
  local index

  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}：未运行"
    return
  fi
  read -r pid < "${pid_file}" || true
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "${name}：进程已结束，清理旧 PID"
    rm -f "${pid_file}"
    return
  fi

  command="$(ps -o args= -p "${pid}" 2>/dev/null || true)"
  if [[ "${command}" != *"${PROJECT_ROOT}"* && "${command}" != *"start_${name}.sh"* ]]; then
    echo "${name}：PID ${pid} 与项目命令不匹配，为避免误杀已跳过。"
    return 1
  fi

  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  for ((index = 0; index < 20; index += 1)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${pid_file}"
      echo "${name}：已停止"
      return
    fi
    sleep 0.25
  done

  kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  rm -f "${pid_file}"
  echo "${name}：已强制停止"
}

stop_process_group "api" "${RUNTIME_DIR}/api.pid"
stop_process_group "frontend" "${RUNTIME_DIR}/frontend.pid"
stop_process_group "vllm_server" "${RUNTIME_DIR}/vllm.pid"
