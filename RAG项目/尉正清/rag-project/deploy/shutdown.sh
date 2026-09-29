#!/usr/bin/env bash
# 结束脚本：停止 API，可选停止依赖服务
#
#   ./shutdown.sh            只停 API
#   ./shutdown.sh --all      连 Redis / MySQL / Milvus 一起停
set -uo pipefail

# 脚本位于 deploy/ 下，项目根目录是它的上一级
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/app.pid"

info() { printf '\033[32m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[33m[WARN]\033[0m  %s\n' "$*"; }

# ---------- 1. 停 API ----------
stopped=0

if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    info "停止 API (pid=$pid)"
    kill "$pid"
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    stopped=1
  fi
  rm -f "$PID_FILE"
fi

# 兜底：按进程名找 uvicorn（例如前台启动、没有 pid 文件）
if pgrep -f "uvicorn app.main:app" >/dev/null 2>&1; then
  info "停止残留的 uvicorn 进程"
  pkill -f "uvicorn app.main:app" || true
  stopped=1
fi

[ "$stopped" -eq 0 ] && warn "没有正在运行的 API 进程"

# ---------- 2. 可选：停依赖服务 ----------
# 与 run.sh 同一套解析：容器名未必等于 compose 服务名
find_container() {
  local service="$1" name
  name="$(docker ps -a --filter "label=com.docker.compose.service=$service" \
          --format '{{.Names}}' 2>/dev/null | head -1)"
  if [ -z "$name" ]; then
    docker ps -a --format '{{.Names}}' | grep -qx "$service" && name="$service"
  fi
  printf '%s' "$name"
}

if [ "${1:-}" = "--all" ]; then
  if command -v docker >/dev/null 2>&1; then
    # 先停 milvus 再停它的依赖 etcd / minio
    for svc in milvus etcd minio redis mysql; do
      name="$(find_container "$svc")"
      if [ -n "$name" ] && docker ps --format '{{.Names}}' | grep -qx "$name"; then
        info "停止容器 $name"
        docker stop "$name" >/dev/null
      fi
    done
  else
    warn "未检测到 docker，跳过"
  fi
fi

info "已结束"
