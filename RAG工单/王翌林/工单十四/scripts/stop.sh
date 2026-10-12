#!/bin/bash
# 工单十四：停止脚本
cd "$(dirname "$0")/.."

for svc in api streamlit; do
    pidfile="data/pids/${svc}.pid"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "已停止 $svc (pid=$pid)"
        fi
        rm -f "$pidfile"
    fi
done

# 兜底清理
pkill -f "uvicorn src.api" 2>/dev/null || true
pkill -f "streamlit run app/streamlit_app" 2>/dev/null || true
echo "所有服务已停止"
