#!/usr/bin/env bash
# 停止服务：按 pidfile 停掉 uvicorn，不碰 Redis/MySQL/Milvus（系统服务）。
set -euo pipefail
cd "$(dirname "$0")/.."
PIDFILE=".uvicorn.pid"
if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "已停止服务（PID $PID）"
  else
    echo "进程 $PID 已不在运行"
  fi
  rm -f "$PIDFILE"
else
  if pgrep -f "uvicorn api:app" >/dev/null 2>&1; then
    pkill -f "uvicorn api:app"
    echo "已按进程名停止 uvicorn（无 pidfile）"
  else
    echo "未发现运行中的 uvicorn 服务"
  fi
fi
