#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${RAG_VENV:-/home/lenovo/.venvs/rag-roleplay}"
cd "$ROOT"

[[ "$(uname -s)" == Linux ]] || { echo "请在 Ubuntu/WSL 中运行。" >&2; exit 1; }
[[ -f .env.local ]] || cp .env.local.example .env.local
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements-local.txt
"$VENV/bin/python" -c 'import fastapi,mysql.connector,pymilvus,redis,sentence_transformers; print("PYTHON_ENV_OK")'
echo "SETUP_OK: Ubuntu Python 环境已创建在 $VENV（Ubuntu 虚拟磁盘位于 D 盘）"
