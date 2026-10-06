#!/bin/bash
# 停止脚本：关闭应用与基础服务
echo "=== 停止 RAG 多角色扮演系统 ==="
pkill -f "uvicorn app.main:app" || true
pkill -f "streamlit run frontend/app_ui.py" || true

sudo systemctl stop redis-server 2>/dev/null || sudo systemctl stop redis 2>/dev/null || true
sudo systemctl stop mysql 2>/dev/null || sudo systemctl stop mysqld 2>/dev/null || true

if [ -d "$HOME/milvus" ]; then
  (cd "$HOME/milvus" && sudo docker compose down) || true
fi

echo "=== 已全部停止 ==="
