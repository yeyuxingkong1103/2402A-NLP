#!/bin/bash
# 启动脚本：拉起基础服务 + 后端 + 前端
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== 启动 RAG 多角色扮演系统 ==="

# 基础服务
sudo systemctl start redis-server 2>/dev/null || sudo systemctl start redis 2>/dev/null || true
sudo systemctl start mysql 2>/dev/null || sudo systemctl start mysqld 2>/dev/null || true

# Milvus（若已用 docker compose 部署在 ~/milvus）
if [ -d "$HOME/milvus" ]; then
  (cd "$HOME/milvus" && sudo docker compose up -d) || true
fi

# shellcheck disable=SC1091
source venv/bin/activate

nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
nohup streamlit run frontend/app_ui.py --server.address 0.0.0.0 > frontend.log 2>&1 &

echo "后端接口: http://localhost:8000/docs"
echo "前端界面: http://localhost:8501"
echo "=== 启动完成 ==="
