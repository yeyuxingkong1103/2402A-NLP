#!/bin/bash
# 工单十四：启动脚本（FastAPI + Streamlit）
set -e
cd "$(dirname "$0")/.."

PYTHON=/home/dabaie/code/my_project/.venv/bin/python3
export PYTHONPATH=$(pwd)

mkdir -p data/logs data/pids

# 启动 FastAPI
echo "启动 FastAPI 服务 (port 8014)..."
$PYTHON -m uvicorn src.api:app --host 0.0.0.0 --port 8014 > data/logs/api.log 2>&1 &
echo $! > data/pids/api.pid

# 启动 Streamlit
echo "启动 Streamlit 界面 (port 8514)..."
$PYTHON -m streamlit run app/streamlit_app.py --server.port 8514 --server.address 0.0.0.0 > data/logs/streamlit.log 2>&1 &
echo $! > data/pids/streamlit.pid

sleep 3
echo "=== 服务已启动 ==="
echo "FastAPI:  http://localhost:8014/docs"
echo "Streamlit: http://localhost:8514"
