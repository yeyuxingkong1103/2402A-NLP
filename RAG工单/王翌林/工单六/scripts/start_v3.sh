#!/bin/bash
# ================================================================
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# scripts/start_v3.sh —— 启动 FastAPI(8003) + Streamlit(8503)
# 用法：bash scripts/start_v3.sh
# ================================================================
set -e

echo "================================================"
echo "工单三：启动服务"
echo "工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化"
echo "================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="/home/dabaie/code/my_project/.venv/bin/python"

export RAG_EMBED_DEVICE=cuda
export RAG_EMBED_BATCH_SIZE=8
export RAG_EMBED_MAX_SEQ=512

# ================= 1. 启动 FastAPI =================
echo "[STEP 1] 启动 FastAPI (端口 8003)..."
# 检查端口是否被占用
if lsof -i :8003 2>/dev/null | grep LISTEN; then
    echo "[WARN] 端口 8003 已被占用，跳过"
else
    nohup $PYTHON_BIN -m uvicorn src.api_v3:app \
        --host 0.0.0.0 --port 8003 \
        > /tmp/api_v3.log 2>&1 &
    echo "[OK] FastAPI PID: $!"
    echo "     日志: /tmp/api_v3.log"
    sleep 3
    if curl -s http://127.0.0.1:8003/api/v3/health | grep -q "ok"; then
        echo "[OK] FastAPI 健康检查通过"
    else
        echo "[WARN] FastAPI 未就绪，请检查日志"
    fi
fi

# ================= 2. 启动 Streamlit =================
echo ""
echo "[STEP 2] 启动 Streamlit (端口 8503)..."
if lsof -i :8503 2>/dev/null | grep LISTEN; then
    echo "[WARN] 端口 8503 已被占用，跳过"
else
    nohup $PYTHON_BIN -m streamlit run app/streamlit_app_v3.py \
        --server.port 8503 \
        --server.headless true \
        --server.address 0.0.0.0 \
        > /tmp/streamlit_v3.log 2>&1 &
    echo "[OK] Streamlit PID: $!"
    echo "     日志: /tmp/streamlit_v3.log"
    sleep 3
    echo "[OK] Streamlit 启动中"
fi

echo ""
echo "================================================"
echo "[DONE] 服务启动完成！"
echo "  FastAPI:   http://127.0.0.1:8003/docs"
echo "  Streamlit: http://127.0.0.1:8503"
echo "  停止服务:  bash scripts/stop_v3.sh"
echo "================================================"
