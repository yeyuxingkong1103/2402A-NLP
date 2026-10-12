#!/bin/bash
# 工单十五：启动脚本（FastAPI + Streamlit）
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/dabaie/code/my_project/.venv/bin/python3}

mkdir -p data/logs data/pids

echo "启动 FastAPI 服务 (port 8015)..."
setsid nohup $PYTHON_BIN -m uvicorn src.api:app --host 0.0.0.0 --port 8015 > data/logs/api.log 2>&1 < /dev/null &
echo $! > data/pids/api.pid

echo "启动 Streamlit 界面 (port 8515)..."
setsid nohup $PYTHON_BIN -m streamlit run app/streamlit_app.py --server.port 8515 --server.address 0.0.0.0 --server.headless true > data/logs/streamlit.log 2>&1 < /dev/null &
echo $! > data/pids/streamlit.pid

sleep 3
echo "=== 服务已启动 ==="
echo "FastAPI:  http://localhost:8015/docs"
echo "Streamlit: http://localhost:8515"
