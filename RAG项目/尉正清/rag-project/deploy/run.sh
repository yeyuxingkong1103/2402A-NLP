#!/usr/bin/env bash
# 启动脚本：拉起依赖服务并启动 API
#
#   ./run.sh              前台启动
#   ./run.sh -d           后台启动（日志写入 logs/app.log）
#   PORT=9000 ./run.sh    指定端口
set -euo pipefail

# 脚本位于 deploy/ 下，项目根目录是它的上一级
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

info()  { printf '\033[32m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[31m[ERROR]\033[0m %s\n' "$*" >&2; }

# ---------- 1. 检查 Python 环境 ----------
if [ ! -x "$PYTHON" ]; then
  error "未找到虚拟环境: $PYTHON"
  echo "请先执行 ./install.sh 初始化环境"
  exit 1
fi

# ---------- 2. 拉起依赖服务 ----------
# 按 compose **服务名**解析容器名：两者未必同名。本机在用的那套 compose 就把
# redis 服务命名成 redis7、mysql 命名成 mysql8，写死容器名会找不到。
find_container() {
  local service="$1" name
  name="$(docker ps -a --filter "label=com.docker.compose.service=$service" \
          --format '{{.Names}}' 2>/dev/null | head -1)"
  if [ -z "$name" ]; then
    # 没有 compose 标签（docker run 手工起的），退回按服务名当容器名找
    docker ps -a --format '{{.Names}}' | grep -qx "$service" && name="$service"
  fi
  printf '%s' "$name"
}

if command -v docker >/dev/null 2>&1; then
  for svc in redis mysql milvus etcd minio; do
    name="$(find_container "$svc")"
    if [ -z "$name" ]; then
      warn "找不到 $svc 容器，请先执行 ./install.sh"
    elif ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      info "启动容器 $name"
      docker start "$name" >/dev/null
    fi
  done
else
  warn "未检测到 docker，跳过依赖服务启动"
fi

# ---------- 3. 等 Milvus 就绪 ----------
info "等待 Milvus 就绪..."
for i in $(seq 1 40); do
  if "$PYTHON" - <<'PY' >/dev/null 2>&1
import sys
from pymilvus import MilvusClient
try:
    MilvusClient(uri="http://localhost:19530").list_collections()
except Exception:
    sys.exit(1)
PY
  then
    info "Milvus 已就绪"
    break
  fi
  [ "$i" -eq 40 ] && warn "Milvus 等待超时，服务仍会启动"
  sleep 3
done

# ---------- 4. 启动服务 ----------
info "启动 API  http://127.0.0.1:${PORT}/docs"

if [ "${1:-}" = "-d" ]; then
  nohup "$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT" \
    >> "$LOG_DIR/app.log" 2>&1 &
  echo $! > "$LOG_DIR/app.pid"
  sleep 3
  if kill -0 "$(cat "$LOG_DIR/app.pid")" 2>/dev/null; then
    info "已在后台启动 (pid=$(cat "$LOG_DIR/app.pid"))，日志: logs/app.log"
  else
    error "启动失败，请查看 logs/app.log"
    exit 1
  fi
else
  exec "$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
fi
