#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../02-研发/核心代码" && pwd)"
PID_FILE="${PROJECT_ROOT}/data/pids/rag-api.pid"
TIMEOUT_SECONDS="${RAG_SHUTDOWN_TIMEOUT:-20}"

if [[ ! -f "$PID_FILE" ]]; then
  printf '%s\n' '未找到 PID 文件，服务可能未由交付脚本启动。'
  exit 0
fi

PID="$(cat "$PID_FILE")"
if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
  printf 'PID 文件内容无效: %s\n' "$PID_FILE" >&2
  exit 1
fi

if ! kill -0 "$PID" 2>/dev/null; then
  printf '进程 %s 已不存在，清理 PID 文件。\n' "$PID"
  rm -f "$PID_FILE"
  exit 0
fi

PROCESS_COMMAND="$(ps -p "$PID" -o args= 2>/dev/null || true)"
if [[ "$PROCESS_COMMAND" != *"uvicorn app.main:app"* ]]; then
  printf '拒绝停止 PID %s：进程命令不属于本项目 Uvicorn。\n' "$PID" >&2
  printf '实际命令: %s\n' "$PROCESS_COMMAND" >&2
  exit 1
fi

printf '正在停止 RAG API，PID=%s\n' "$PID"
kill -TERM "$PID"

for ((elapsed = 0; elapsed < TIMEOUT_SECONDS; elapsed++)); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    printf '%s\n' '服务已优雅停止。'
    exit 0
  fi
  sleep 1
done

printf '等待 %s 秒后进程仍未退出，发送 SIGKILL。\n' "$TIMEOUT_SECONDS" >&2
kill -KILL "$PID"
rm -f "$PID_FILE"
printf '%s\n' '服务已强制停止。'
