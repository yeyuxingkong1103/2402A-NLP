#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID_DIR="$ROOT_DIR/.runtime"

stop_pid() {
  local name="$1"
  local file="$PID_DIR/$2"
  if [ -f "$file" ]; then
    local pid
    pid="$(cat "$file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      echo "$name stopped: $pid"
    fi
    rm -f "$file"
  fi
}

stop_pid backend backend.pid
stop_pid frontend frontend.pid
echo "local services stopped"
