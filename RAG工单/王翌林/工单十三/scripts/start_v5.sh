#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-Query 理解优化任务
# scripts/start_v5.sh —— 工单五 启动 FastAPI v5(8005) + Streamlit v5(8505)
set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/dabaie/code/my_project/.venv/bin/python}"
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

echo "[v5] 启动 FastAPI v5 (port 8005)..."
nohup "$PYTHON" -m uvicorn src.api_v5:app --host 0.0.0.0 --port 8005 \
    > "$LOG_DIR/api_v5.log" 2>&1 &
echo $! > data/pids/api_v5.pid

echo "[v5] 启动 Streamlit v5 (port 8505)..."
nohup "$PYTHON" -m streamlit run app/streamlit_app_v5.py \
    --server.port 8505 --server.headless true --server.address 0.0.0.0 \
    > "$LOG_DIR/streamlit_v5.log" 2>&1 &
echo $! > data/pids/streamlit_v5.pid

# 工单五：健康检查重试（最多 90s）
echo "[v5] 等待服务就绪..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8005/api/v5/health > /dev/null 2>&1; then
        echo "[v5] API v5 就绪 (${i}x3s)"
        break
    fi
    sleep 3
done

echo "[v5] 启动完成"
echo "  API v5:      http://localhost:8005/api/v5/health"
echo "  Streamlit:   http://localhost:8505"
