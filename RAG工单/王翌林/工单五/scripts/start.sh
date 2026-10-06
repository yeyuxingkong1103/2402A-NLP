#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统
# scripts/start.sh — 启动服务（FastAPI + Streamlit）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ========== 工单：人工智能NLP-RAG-基于PDF文档的问答系统 ==========
# 配置 —— 用 8001 端口（8000 被其他项目占用）
API_PORT="${APP_PORT:-8001}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
LOG_DIR="$PROJECT_ROOT/data/logs"
PID_DIR="$PROJECT_ROOT/data/pids"

mkdir -p "$LOG_DIR" "$PID_DIR"

# ========== Python 解释器 ==========
# 优先用当前 venv，回退 conda，再回退系统 python
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -f "/home/dabaie/code/my_project/.venv/bin/python" ]; then
        PYTHON_BIN="/home/dabaie/code/my_project/.venv/bin/python"
    elif command -v conda &>/dev/null && conda env list | grep -q rag_pdf_qa; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate rag_pdf_qa 2>/dev/null || true
        PYTHON_BIN="$(which python)"
    else
        PYTHON_BIN="$(which python3)"
    fi
fi

echo "============================================================"
echo "🚀 启动服务 — 工单：人工智能NLP-RAG-基于PDF文档的问答系统"
echo "   API port=$API_PORT | Streamlit port=$STREAMLIT_PORT"
echo "   Python: $PYTHON_BIN"
echo "============================================================"

# ========== 1. 前置检查 ==========
echo ""
echo "[1/3] 检查依赖服务..."
# MySQL
if command -v mysql &>/dev/null; then
    MYSQL_OK=false
    for _ in $(seq 1 3); do
        if mysql -h localhost -P "${MYSQL_PORT:-3307}" -u "${MYSQL_USER:-root}" -p"${MYSQL_PASSWORD:-355359}" -e "SELECT 1" &>/dev/null; then
            MYSQL_OK=true; break
        fi
        sleep 2
    done
    $MYSQL_OK && echo "  ✅ MySQL OK" || echo "  ⚠️  MySQL 未启动（api.log 会记录）"
else
    echo "  ⚠️  mysql client 未安装，跳过 MySQL 检查"
fi

# Milvus
MILVUS_HOST="${MILVUS_HOST:-localhost}"
MILVUS_PORT="${MILVUS_PORT:-19530}"
if command -v curl &>/dev/null; then
    curl -s -m 3 "http://$MILVUS_HOST:$MILVUS_PORT" &>/dev/null && echo "  ✅ Milvus OK" || echo "  ⚠️  Milvus 未启动（自动回退 Lite）"
fi

# ========== 2. 检查端口占用 ==========
echo ""
echo "[2/3] 检查端口占用..."
for port in "$API_PORT" "$STREAMLIT_PORT"; do
    if ss -tlnp 2>/dev/null | grep -q ":$port "; then
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | head -1 | grep -oP 'pid=\K[0-9]+')
        if [ -n "$pid" ]; then
            echo "  ⚠️  端口 $port 已被 PID=$pid 占用，先停止..."
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
    fi
    echo "  ✅ 端口 $port 空闲"
done

# ========== 3. 启动服务 ==========
echo ""
echo "[3/3] 启动服务..."

# FastAPI
nohup "$PYTHON_BIN" -m uvicorn src.api:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    > "$LOG_DIR/api.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "$PID_DIR/api.pid"
echo "  ✅ FastAPI started: PID=$API_PID → http://127.0.0.1:$API_PORT"
sleep 3

# Streamlit
nohup "$PYTHON_BIN" -m streamlit run app/streamlit_app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.headless true \
    > "$LOG_DIR/streamlit.log" 2>&1 &
ST_PID=$!
echo "$ST_PID" > "$PID_DIR/streamlit.pid"
echo "  ✅ Streamlit started: PID=$ST_PID → http://127.0.0.1:$STREAMLIT_PORT"

sleep 2

# ========== 4. 验证 ==========
echo ""
echo "📋 服务状态："
if command -v curl &>/dev/null; then
    curl -s -m 3 "http://127.0.0.1:$API_PORT/api/health" | "$PYTHON_BIN" -c "import sys,json; d=json.load(sys.stdin); print(f'  API: {d[\"status\"]} | mysql={d[\"mysql\"]} | milvus={d[\"milvus\"]}')" 2>/dev/null || echo "  ⚠️  API 暂未就绪（查看 $LOG_DIR/api.log）"
fi
echo "  Streamlit: http://127.0.0.1:$STREAMLIT_PORT"

echo ""
echo "🎉 PID 文件：$PID_DIR/{api,streamlit}.pid"
echo "📄 日志文件：$LOG_DIR/{api,streamlit}.log"
echo "🛑 停止服务：bash scripts/stop.sh"
