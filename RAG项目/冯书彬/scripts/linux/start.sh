#!/usr/bin/env bash
set -Eeuo pipefail

SKIP_DEPENDENCIES=0
SKIP_MIGRATIONS=0
API_PORT="${API_PORT:-8010}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-dependencies) SKIP_DEPENDENCIES=1 ;;
    --skip-migrations) SKIP_MIGRATIONS=1 ;;
    --port) shift; API_PORT="${1:?--port 需要端口号}" ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/runtime"
LOG_DIR="$RUNTIME_DIR/logs"
PID_DIR="$RUNTIME_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"
cd "$REPO_ROOT"

command -v docker >/dev/null || { echo "找不到命令：docker，请先安装 Docker。" >&2; exit 1; }
command -v python3 >/dev/null || { echo "找不到命令：python3，请先安装 Python 3。" >&2; exit 1; }

wait_tcp_port() {
  local host="$1" port="$2" timeout="${3:-120}" start now
  start="$(date +%s)"
  while ! (echo >/dev/tcp/"$host"/"$port") >/dev/null 2>&1; do
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      echo "等待 $host:$port 超时。" >&2
      exit 1
    fi
    sleep 2
  done
}

stop_pid_file() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 0
  local pid
  pid="$(<"$pid_file")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

if (( ! SKIP_DEPENDENCIES )); then
  echo "启动 MySQL、Redis、etcd、MinIO 和 Milvus..."
  docker compose up -d mysql redis etcd minio milvus
  wait_tcp_port 127.0.0.1 3306
  wait_tcp_port 127.0.0.1 6380
  wait_tcp_port 127.0.0.1 19530
fi

if (( ! SKIP_MIGRATIONS )); then
  echo "执行数据库迁移..."
  python3 -m alembic upgrade head
fi

stop_pid_file "$PID_DIR/api.pid"
stop_pid_file "$PID_DIR/celery.pid"

echo "启动 API：http://127.0.0.1:$API_PORT"
nohup python3 -m uvicorn backend.app.main:app \
  --host 127.0.0.1 --port "$API_PORT" \
  >>"$LOG_DIR/api.log" 2>>"$LOG_DIR/api-error.log" &
echo $! > "$PID_DIR/api.pid"

echo "启动 Celery worker..."
nohup python3 -m celery -A backend.app.workers.celery_app.celery_app worker \
  --loglevel=info --pool=solo \
  >>"$LOG_DIR/celery.log" 2>>"$LOG_DIR/celery-error.log" &
echo $! > "$PID_DIR/celery.pid"

echo "启动完成。日志目录：$LOG_DIR"
