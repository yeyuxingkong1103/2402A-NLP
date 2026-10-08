#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-混合检索任务
# scripts/start_v6.sh —— 工单六 启动 FastAPI v6(8006) + Streamlit v6(8506)
set -e
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/dabaie/code/my_project/.venv/bin/python}"
LOG_DIR="data/logs"
mkdir -p "$LOG_DIR" data/pids

echo "[v6] 启动 FastAPI v6 (port 8006)..."
nohup "$PYTHON" -m uvicorn src.api_v6:app --host 0.0.0.0 --port 8006 \
    > "$LOG_DIR/api_v6.log" 2>&1 &
echo $! > data/pids/api_v6.pid

echo "[v6] 启动 Streamlit v6 (port 8506)..."
nohup "$PYTHON" -m streamlit run app/streamlit_app_v6.py \
    --server.port 8506 --server.headless true --server.address 0.0.0.0 \
    > "$LOG_DIR/streamlit_v6.log" 2>&1 &
echo $! > data/pids/streamlit_v6.pid

# 工单六：健康检查重试（首次需加载模型，最多 180s）
echo "[v6] 等待服务就绪..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8006/api/v6/health > /dev/null 2>&1; then
        echo "[v6] API v6 就绪 (${i}x3s)"
        break
    fi
    sleep 3
done

# 工单六：后台预热（加载 bge-m3/reranker/全文索引，保障首问 ≤3s，不阻塞脚本）
nohup curl -s -X POST http://localhost:8006/api/v6/ask \
    -H "Content-Type: application/json" \
    -d '{"question":"公司主营业务是什么","doc_id":"招股说明书1"}' \
    > /dev/null 2>&1 &

echo "[v6] 启动完成（模型后台预热中，约1-2分钟）"
echo "  API v6:      http://localhost:8006/api/v6/health"
echo "  Streamlit:   http://localhost:8506"
