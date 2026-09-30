#!/usr/bin/env bash
set -euo pipefail
docker compose up -d --wait --wait-timeout 180
mkdir -p logs
nohup .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >logs/api.log 2>&1 &
echo $! > .api.pid
nohup .venv/bin/python -m streamlit run web_ui.py --server.port 8501 --server.address 0.0.0.0 >logs/ui.log 2>&1 &
echo $! > .ui.pid
echo "API: http://127.0.0.1:8000/docs"
echo "UI:  http://127.0.0.1:8501"
