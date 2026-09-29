#!/usr/bin/env bash
# ============================================================
# RAG 角色扮演系统 · 停止脚本
# 用法：bash deploy/shutdown.sh
# ============================================================
set -uo pipefail

cd "$(dirname "$0")/.."
PID_FILE="$(pwd)/.server.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "[shutdown] 没有运行中的服务（无 .server.pid）"
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 10); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$PID" 2>/dev/null && echo "[shutdown] 已强制结束" || echo "[shutdown] 服务已停止（PID $PID）"
else
  echo "[shutdown] 进程已不存在"
fi
rm -f "$PID_FILE"
