#!/bin/bash
# 工单十八 · 启动质检 API 服务
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/document_quality_assessment"
PORT="${1:-8028}"
mkdir -p data/logs
PID="data/logs/quality-api.pid"; LOG="data/logs/quality-api.log"
[ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null && { echo "已运行 PID=$(cat $PID)"; exit 0; }
nohup python -m uvicorn api.quality_api:app --host 0.0.0.0 --port "$PORT" \
  --log-level info >"$LOG" 2>&1 &
echo $! > "$PID"; sleep 3
kill -0 "$(cat "$PID")" 2>/dev/null && echo "启动成功 PID=$(cat $PID) 端口=$PORT" || { echo "启动失败，见 $LOG"; exit 1; }
