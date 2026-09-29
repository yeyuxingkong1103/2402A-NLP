#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v docker >/dev/null 2>&1 && [[ -f docker-compose.yml ]]; then
  docker compose down
  exit 0
fi

pkill -f "uvicorn.*app.main:app" || true
pkill -f "python.*run.py" || true
echo "Application processes stopped"
