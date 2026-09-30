#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../02-研发/核心代码" && pwd)"
ENV_NAME="rag-kf"
MODE="production"
WORKERS="${RAG_WORKERS:-2}"

usage() {
  cat <<'EOF'
用法: run.sh [--mode development|production] [--env-name NAME] [--workers N]
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --env-name)
      ENV_NAME="${2:-}"
      shift 2
      ;;
    --workers)
      WORKERS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '未知参数: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "development" && "$MODE" != "production" ]]; then
  printf '%s\n' '--mode 必须是 development 或 production' >&2
  exit 2
fi

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  printf '%s\n' '--workers 必须是正整数' >&2
  exit 2
fi

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  printf '%s\n' '未找到 .env，请先执行 install.sh 并完成配置。' >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    CONDA_EXE="${HOME}/miniconda3/bin/conda"
  else
    printf '%s\n' '未找到 Conda，请先执行 install.sh。' >&2
    exit 1
  fi
else
  CONDA_EXE="$(command -v conda)"
fi

CONDA_BASE="$($CONDA_EXE info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/data/pids"
PID_FILE="${PROJECT_ROOT}/data/pids/rag-api.pid"
LOG_FILE="${PROJECT_ROOT}/logs/rag-api.log"

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(cat "$PID_FILE")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    printf '服务已经运行，PID=%s\n' "$EXISTING_PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

eval "$(python - "${PROJECT_ROOT}/.env" <<'PY'
import os
import shlex
import sys

from dotenv import dotenv_values

for key, value in dotenv_values(sys.argv[1]).items():
    if value is not None:
        print(f"export {key}={shlex.quote(value)}")
PY
)"

HOST="${API_HOST:-127.0.0.1}"
PORT="${API_PORT:-8000}"

if [[ "$MODE" == "production" ]]; then
  export APP_ENV="production"
  if [[ -z "${API_KEY:-}" && -z "${API_KEY_IDENTITIES:-}" ]]; then
    printf '%s\n' '生产模式必须在 .env 中配置 API_KEY 或 API_KEY_IDENTITIES。' >&2
    exit 1
  fi
  COMMAND=(python -m uvicorn app.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --proxy-headers)
else
  export APP_ENV="development"
  COMMAND=(python -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload)
fi

cd "$PROJECT_ROOT"
nohup "${COMMAND[@]}" >>"$LOG_FILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"

for _ in {1..30}; do
  if ! kill -0 "$PID" 2>/dev/null; then
    printf '服务启动失败，请查看 %s\n' "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  if curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    printf '服务启动成功: http://%s:%s\n' "$HOST" "$PORT"
    printf 'PID=%s 日志=%s\n' "$PID" "$LOG_FILE"
    exit 0
  fi
  sleep 1
done

printf '进程已启动但健康检查超时，请查看 %s\n' "$LOG_FILE" >&2
exit 1
