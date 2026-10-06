#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p data/uploads data/indexes logs
if [[ -f .env ]]; then
  echo "Using .env configuration"
else
  echo "Warning: .env not found; application defaults will be used"
fi

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
elif [[ -x zhuangao6/bin/python ]]; then
  PYTHON_BIN="zhuangao6/bin/python"
elif [[ -x zhuangao6/Scripts/python.exe ]]; then
  PYTHON_BIN="zhuangao6/Scripts/python.exe"
else
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" run.py
