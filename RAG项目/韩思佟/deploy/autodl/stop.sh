#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

stop_pid_file() {
  local file="$1"
  local name="$2"
  if [ ! -f "$file" ]; then
    echo "$name pid file not found: $file"
    return
  fi

  local pid
  pid="$(cat "$file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "Stopped $name, pid=$pid"
  else
    echo "$name is not running, stale pid=$pid"
  fi
  rm -f "$file"
}

stop_pid_file logs/rag-api.pid "RAG API"
stop_pid_file logs/vllm.pid "vLLM"
