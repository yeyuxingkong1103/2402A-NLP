#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# scripts/stop_v5.sh —— 工单五 停止 FastAPI v5 + Streamlit v5
set -e
cd "$(dirname "$0")/.."

for name in api_v5 streamlit_v5; do
    pidfile="data/pids/${name}.pid"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        kill "$pid" 2>/dev/null && echo "[v5] 已停止 $name (pid=$pid)" || true
        rm -f "$pidfile"
    fi
done

# 工单五：兜底按端口杀（防止 pid 文件丢失）
for port in 8005 8505; do
    pids=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | sort -u)
    for pid in $pids; do
        kill "$pid" 2>/dev/null && echo "[v5] 已停止端口 $port 进程 (pid=$pid)" || true
    done
done
echo "[v5] 停止完成"
