#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p .runtime

"$ROOT_DIR/scripts/start/backend.sh" > .runtime/backend.log 2>&1 &
echo $! > .runtime/backend.pid

(
  cd "$ROOT_DIR/frontend"
  npm run dev -- --host 0.0.0.0 > "$ROOT_DIR/.runtime/frontend.log" 2>&1
) &
echo $! > .runtime/frontend.pid

echo "backend and frontend started; logs are in .runtime/"
