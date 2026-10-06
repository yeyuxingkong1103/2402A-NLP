#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8000}"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

PYTHONPATH=backend exec "$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
