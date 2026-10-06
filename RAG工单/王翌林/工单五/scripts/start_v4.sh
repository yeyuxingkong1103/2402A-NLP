#!/bin/bash
# ================================================================
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# scripts/start_v4.sh —— 启动 FastAPI(8004) + Streamlit(8504)
# 用法：bash scripts/start_v4.sh
# ================================================================
set -e

echo "================================================"
echo "工单四：启动服务"
echo "工单编号：人工智能NLP-RAG-图像内容解析及检索优化"
echo "================================================"

# 工单四：项目根目录（中文路径加引号处理）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="/home/dabaie/code/my_project/.venv/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

export RAG_EMBED_DEVICE=cuda
export RAG_EMBED_BATCH_SIZE=8
export RAG_EMBED_MAX_SEQ=512

# ================= 0. 前置检查 =================
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "[ERROR] 未找到 .env，请先执行 bash scripts/install_v4.sh"
    exit 1
fi

# ================= 1. 启动 FastAPI =================
echo ""
echo "[STEP 1] 启动 FastAPI (端口 8004)..."
if lsof -i :8004 2>/dev/null | grep LISTEN; then
    echo "[WARN] 端口 8004 已被占用，跳过"
else
    nohup $PYTHON_BIN -m uvicorn src.api_v4:app \
        --host 0.0.0.0 --port 8004 \
        > /tmp/api_v4.log 2>&1 &
    echo "[OK] FastAPI PID: $!"
    echo "     日志: /tmp/api_v4.log"
    # 工单四：等待就绪（模型懒加载最长 90s）
    echo -n "     健康检查"
    for i in $(seq 1 30); do
        if curl -s http://127.0.0.1:8004/api/v4/health 2>/dev/null | grep -q '"ok"'; then
            echo " -> [OK] FastAPI 就绪"
            break
        fi
        echo -n "."
        sleep 3
    done
    curl -s http://127.0.0.1:8004/api/v4/health 2>/dev/null | grep -q '"ok"' || \
        echo " -> [WARN] 未就绪，请查看 /tmp/api_v4.log"
fi

# ================= 2. 启动 Streamlit =================
echo ""
echo "[STEP 2] 启动 Streamlit (端口 8504)..."
if lsof -i :8504 2>/dev/null | grep LISTEN; then
    echo "[WARN] 端口 8504 已被占用，跳过"
else
    nohup $PYTHON_BIN -m streamlit run app/streamlit_app_v4.py \
        --server.port 8504 \
        --server.headless true \
        --server.address 0.0.0.0 \
        > /tmp/streamlit_v4.log 2>&1 &
    echo "[OK] Streamlit PID: $!"
    echo "     日志: /tmp/streamlit_v4.log"
    sleep 3
    echo "[OK] Streamlit 启动中（首次打开需加载引擎，约 20-40s）"
fi

echo ""
echo "================================================"
echo "[DONE] 服务启动完成！"
echo "  FastAPI:   http://127.0.0.1:8004/docs"
echo "  Streamlit: http://127.0.0.1:8504"
echo "  停止服务:  bash scripts/stop_v4.sh"
echo "================================================"
