#!/usr/bin/env bash
# Role RAG_try 启动脚本（WSL / Linux；模型与向量库仍在 Windows/WSL 本机）
#
# 用法：
#   bash start.sh                 # 启动服务
#   bash start.sh --ingest        # 先重建知识库
#   bash start.sh --check         # 只做环境自检
set -euo pipefail

PYTHON="${PYTHON:-/mnt/d/an/envs/rags_/python.exe}"
PORT="${PORT:-8020}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONIOENCODING=utf-8
export PYTHONPATH="$ROOT/src"
export ROLE_RAG_PORT="$PORT"

echo "==> 环境自检"
"$PYTHON" tools/check_env.py

for arg in "$@"; do
  case "$arg" in
    --check) exit 0 ;;
    --ingest)
      echo "==> 重建并写入知识库"
      "$PYTHON" tools/ingest_cli.py --all --recreate
      ;;
  esac
done

echo "==> 启动服务：http://127.0.0.1:${PORT}/"
exec "$PYTHON" -m uvicorn role_rag.api.app:app --host 127.0.0.1 --port "$PORT"
