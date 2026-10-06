#!/usr/bin/env bash
# 法律 RAG 服务停止脚本
#
# 先按 PID 文件优雅停止（SIGTERM），超时未退出再 SIGKILL。
# 只停本项目拉起的服务，不动系统里的 Redis / MySQL。
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$APP_DIR/run/api.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "未找到 PID 文件（$PID_FILE），服务可能未启动。"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "PID $PID 已不存在，清理 PID 文件。"
  rm -f "$PID_FILE"
  exit 0
fi

echo "==> 停止服务 PID $PID"
kill "$PID" 2>/dev/null || true

for _ in $(seq 1 20); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "    已优雅退出"
    rm -f "$PID_FILE"
    exit 0
  fi
  sleep 0.5
done

echo "    未在 10s 内退出，强制结束"
kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "==> 已停止"
