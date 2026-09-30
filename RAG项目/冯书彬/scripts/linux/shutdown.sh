#!/usr/bin/env bash
set -Eeuo pipefail

KEEP_DEPENDENCIES=0
if [[ "${1:-}" == "--keep-dependencies" ]]; then
  KEEP_DEPENDENCIES=1
elif [[ $# -gt 0 ]]; then
  echo "未知参数：$1" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID_DIR="$REPO_ROOT/runtime/pids"

stop_pid_file() {
  local name="$1" pid_file="$PID_DIR/$1.pid"
  [[ -f "$pid_file" ]] || return 0
  local pid
  pid="$(<"$pid_file")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "停止 $name（PID $pid）..."
    kill "$pid" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

stop_pid_file celery
stop_pid_file api

if (( ! KEEP_DEPENDENCIES )); then
  command -v docker >/dev/null || { echo "找不到命令：docker，请先安装 Docker。" >&2; exit 1; }
  cd "$REPO_ROOT"
  echo "停止 Docker 依赖服务（保留数据卷）..."
  docker compose down
fi

echo "停止完成。"
