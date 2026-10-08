#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================================
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# scripts/stop_optimized.sh —— 停止工单二优化版服务（FastAPI 8000 + Streamlit 8502）
#
# 清理顺序（三层兜底，保证停干净且不误伤其他项目）：
#   1) data/pids/{api_optimized,streamlit_optimized}.pid
#   2) 端口 8000 / 8502 监听进程
#   3) 命令行特征匹配（仅匹配优化版前端文件与 --port 8000 的 uvicorn）
#
# 中文路径：PID 文件路径经 BASH_SOURCE 解析并全程加引号，兼容中文目录。
# ============================================================================
set -euo pipefail
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

WORKORDER="人工智能NLP-RAG-基于PDF文档的问答系统优化"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_DIR="$PROJECT_ROOT/data/pids"
API_PORT="${APP_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8502}"

echo "============================================================"
echo "🛑 停止优化版服务（工单编号：$WORKORDER）"
echo "   FastAPI=$API_PORT | Streamlit=$STREAMLIT_PORT"
echo "============================================================"

stopped=0

kill_pid() {
    local pid="$1" tag="$2"
    if kill -0 "$pid" 2>/dev/null; then
        echo "  📌 停止 $tag：PID=$pid（TERM → 等 2s → KILL）"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        kill -KILL "$pid" 2>/dev/null || true
        stopped=$((stopped + 1))
    fi
}

# ---------- 1) PID 文件 ----------
echo "[1/3] 按 PID 文件停止 ..."
for svc in api_optimized streamlit_optimized; do
    pid_file="$PID_DIR/$svc.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file" 2>/dev/null || echo "")
        if [ -n "$pid" ]; then kill_pid "$pid" "$svc"; fi
        rm -f "$pid_file"
    else
        echo "  ↳ 无 $svc.pid（可能已停止）"
    fi
done

# ---------- 2) 端口扫描兜底 ----------
echo ""
echo "[2/3] 按端口扫描兜底 ..."
for port in "$API_PORT" "$STREAMLIT_PORT"; do
    if ss -tln 2>/dev/null | grep -q ":$port "; then
        pids=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | sort -u || true)
        for pid in $pids; do
            echo "  🔍 端口 $port 的残留进程 PID=$pid，强制停止"
            kill -KILL "$pid" 2>/dev/null || true
            stopped=$((stopped + 1))
        done
    else
        echo "  ✅ 端口 $port 无监听"
    fi
done

# ---------- 3) 命令行特征兜底（仅优化版，不动基线 8001/8002） ----------
echo ""
echo "[3/3] 按进程特征扫描兜底 ..."
# 3a) 优化版 Streamlit 前端（文件名唯一，不会误伤基线）
st_pids=$(pgrep -f "streamlit run app/streamlit_app_optimized.py" 2>/dev/null || true)
if [ -n "$st_pids" ]; then
    for pid in $st_pids; do
        echo "  🔍 残留优化版 Streamlit PID=$pid，强制停止"
        kill -KILL "$pid" 2>/dev/null || true
        stopped=$((stopped + 1))
    done
fi
# 3b) 仅停止命令行带 --port 8000 的 uvicorn，避免误杀 8001/8002 上的其他实例
uv_pids=$(pgrep -af "uvicorn src.api:app" 2>/dev/null \
          | grep -- "--port $API_PORT" | awk '{print $1}' || true)
if [ -n "$uv_pids" ]; then
    for pid in $uv_pids; do
        echo "  🔍 残留优化版 API（--port $API_PORT）PID=$pid，强制停止"
        kill -KILL "$pid" 2>/dev/null || true
        stopped=$((stopped + 1))
    done
fi

sleep 1

# ---------- 结果确认 ----------
echo ""
echo "============================================================"
remain=0
ss -tln 2>/dev/null | grep -q ":$API_PORT " && { echo "  ⚠️  端口 $API_PORT 仍被占用"; remain=1; }
ss -tln 2>/dev/null | grep -q ":$STREAMLIT_PORT " && { echo "  ⚠️  端口 $STREAMLIT_PORT 仍被占用"; remain=1; }
if [ "$remain" -eq 0 ]; then
    echo "✅ 优化版服务已全部停止（共清理 $stopped 个进程），端口 $API_PORT/$STREAMLIT_PORT 已释放"
fi
echo "============================================================"
