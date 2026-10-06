#!/usr/bin/env bash
# RAG2 在线服务启动脚本（WSL / Git Bash）
set -e

cd "$(dirname "$0")/.."
PY="D:/an/envs/rags_/python.exe"

echo "[1/3] 检查 Redis ..."
if "$PY" -c "import redis; redis.Redis(host='127.0.0.1', port=6379).ping(); print('    Redis OK')" 2>/dev/null; then
  :
else
  echo "    [警告] Redis 不可用，请先启动 Redis 服务。"
fi

echo "[2/3] 检查 Ollama ..."
if "$PY" -c "import httpx; httpx.get('http://localhost:11434/api/tags', timeout=3).raise_for_status(); print('    Ollama OK')" 2>/dev/null; then
  :
else
  echo "    [警告] Ollama 不可用，请先启动 Ollama。"
fi

echo "[3/3] 启动服务（http://127.0.0.1:8000）..."
exec "$PY" run.py
