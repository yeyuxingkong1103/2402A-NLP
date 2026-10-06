#!/usr/bin/env bash
# 启动脚本：API + Streamlit（后台）
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs run

# 激活环境（Conda 优先，回退 venv）
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -q '^roleplay$'; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate roleplay
elif [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

export APP_ENV="${APP_ENV:-production}"

echo "================ 启动 FastAPI ================"
nohup python -m uvicorn api:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" \
    > logs/api.log 2>&1 &
echo $! > run/api.pid
echo "API pid=$(cat run/api.pid)  日志：logs/api.log"

echo "================ 启动 Streamlit ================"
nohup python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 \
    > logs/streamlit.log 2>&1 &
echo $! > run/streamlit.pid
echo "Streamlit pid=$(cat run/streamlit.pid)  日志：logs/streamlit.log"

echo "启动完成。访问：http://<host>:8000/health  http://<host>:8501"
echo "停止：bash shutdown.sh"
