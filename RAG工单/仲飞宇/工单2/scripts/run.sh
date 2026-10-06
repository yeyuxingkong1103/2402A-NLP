#!/usr/bin/env bash
# 启动脚本：后台启动 uvicorn，写入 pid 文件
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/lib.sh

[[ -d ".venv" ]] && source .venv/bin/activate

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PIDFILE="logs/uvicorn.pid"

if is_running "$PIDFILE"; then
    echo "服务已在运行（pid $(pid_of "$PIDFILE")）"
    exit 0
fi

mkdir -p logs
echo "启动服务: http://$HOST:$PORT"
nohup uvicorn app.main:app --host "$HOST" --port "$PORT" > logs/uvicorn.out 2>&1 &
echo $! > "$PIDFILE"
echo "已启动，pid=$(cat "$PIDFILE")，日志: logs/uvicorn.out"
