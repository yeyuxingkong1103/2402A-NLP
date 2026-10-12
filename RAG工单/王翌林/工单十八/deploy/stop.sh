#!/bin/bash
# 工单十八 · 停止质检 API 服务
cd "$(dirname "$0")/.."
PID_FILE="data/logs/quality-api.pid"
PIDS="$(cat "$PID_FILE" 2>/dev/null) $(pgrep -f 'uvicorn api.quality_api:app' 2>/dev/null)"
for pid in $PIDS; do kill -0 "$pid" 2>/dev/null && { kill "$pid" 2>/dev/null; sleep 1; kill -9 "$pid" 2>/dev/null; }; done
rm -f "$PID_FILE"; echo "已停止。"
