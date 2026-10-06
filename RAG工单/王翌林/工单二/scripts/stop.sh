#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统
# scripts/stop.sh — 停止服务（FastAPI + Streamlit）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_DIR="$PROJECT_ROOT/data/pids"

echo "============================================================"
echo "🛑 停止服务 — 工单：人工智能NLP-RAG-基于PDF文档的问答系统"
echo "============================================================"

stopped=0

# ========== 方式 1: PID 文件 ==========
for svc in api streamlit; do
    pid_file="$PID_DIR/$svc.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "📌 停止 $svc: PID=$pid"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            kill -KILL "$pid" 2>/dev/null || true
            rm -f "$pid_file"
            stopped=$((stopped + 1))
        else
            echo "📌 $svc PID=$pid 已不存在"
            rm -f "$pid_file"
        fi
    fi
done

# ========== 方式 2: 端口扫描（兜底） ==========
API_PORT="${APP_PORT:-8001}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

for port in "$API_PORT" "$STREAMLIT_PORT"; do
    if ss -tlnp 2>/dev/null | grep -q ":$port "; then
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | head -1 | grep -oP 'pid=\K[0-9]+')
        if [ -n "$pid" ]; then
            echo "🔍 端口 $port 被 PID=$pid 占用，强制停止"
            kill -9 "$pid" 2>/dev/null || true
            stopped=$((stopped + 1))
        fi
    fi
done

# ========== 方式 3: 进程名扫描（兜底）==========
for pattern in "uvicorn src.api:app" "streamlit run"; do
    pids=$(pgrep -f "$pattern" || true)
    if [ -n "$pids" ]; then
        echo "🔍 进程 '$pattern': $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        stopped=$((stopped + 1))
    fi
done

sleep 1

# ========== 验证 ==========
echo ""
echo "📋 残留进程检查："
remain=$(pgrep -af "uvicorn src.api:app|streamlit run" || true)
if [ -z "$remain" ]; then
    echo "  ✅ 全部清理完成"
else
    echo "  ⚠️  仍有残留：$remain"
fi

echo ""
echo "✅ 停止完成（清理了 $stopped 个进程）"
