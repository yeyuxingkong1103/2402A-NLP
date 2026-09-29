#!/usr/bin/env bash
# =============================================================================
# 法律 RAG 停止脚本（批次 33，配合 run.sh 使用）
#
# 顺序：先停前端、再停后端；Redis/MySQL/Milvus 数据服务【默认不停】，
# 交互询问确认后才停（避免误把数据库一起带走）。
# 用法：shutdown.sh [--stop-data]   （--stop-data 跳过询问直接连数据服务一起停）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$INSTALL_DIR/logs"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
DATA_CONTAINERS="${DATA_CONTAINERS:-rag-redis rag-mysql rag-milvus}"
STOP_DATA=false
DRY_RUN=false

if [ -t 1 ]; then
  C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_RED='\033[0;31m'; C_OFF='\033[0m'
else
  C_GREEN=''; C_YELLOW=''; C_RED=''; C_OFF=''
fi
log()  { printf "${C_GREEN}[shutdown]%s %s${C_OFF}\n" "" "$1"; }
warn() { printf "${C_YELLOW}[shutdown][警告]%s %s${C_OFF}\n" "" "$1"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --stop-data) STOP_DATA=true ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
  shift
done

# 停止一个 pid 文件管理的进程：TERM → 最多 10 秒 → KILL
stop_pidfile() {
  local name="$1" pidfile="$2" pid
  if $DRY_RUN; then log "[dry-run] 将停止 $name（pid 文件 $pidfile；存在则 TERM → 10 秒 → KILL）"; return 0; fi
  if [ ! -f "$pidfile" ]; then
    log "$name：无 pid 文件，可能未由 run.sh 启动，跳过"
    return 0
  fi
  pid="$(cat "$pidfile")"
  if ! kill -0 "$pid" 2>/dev/null; then
    log "$name：进程已不在（pid $pid），清理 pid 文件"
    rm -f "$pidfile"
    return 0
  fi
  log "$name：停止进程 pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    warn "$name：10 秒未退出，强制 KILL"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
  log "$name：已停止"
}

# 兜底：按端口找到残留进程（pid 文件丢失时）
kill_by_port() {
  local port="$1" name="$2" pids=""
  if $DRY_RUN; then log "[dry-run] 将按端口兜底清理：$name 端口 $port 上的残留进程"; return 0; fi
  pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    warn "$name：端口 $port 仍有进程（$pids，可能是 pid 文件丢失的残留），终止之"
    fuser -k -n tcp "$port" 2>/dev/null || true
  fi
}

stop_pidfile "前端" "$LOG_DIR/frontend.pid"
stop_pidfile "后端" "$LOG_DIR/backend.pid"
kill_by_port "$FRONTEND_PORT" "前端"
kill_by_port "$BACKEND_PORT" "后端"

# ---------------- 数据服务：默认不停，询问确认 ----------------
if $DRY_RUN; then
  log "[dry-run] 将询问是否停止数据服务（$DATA_CONTAINERS），确认 y 后逐个 docker stop；默认保留"
  log "[dry-run] 演练结束，未停止任何进程"
  exit 0
fi
if $STOP_DATA; then
  CONFIRM="y"
else
  echo
  printf "${C_YELLOW}是否同时停止数据服务（Redis/MySQL/Milvus：%s）？停了下次启动要重新拉。${C_OFF}\n" "$DATA_CONTAINERS"
  read -r -p "输入 y 确认停止，其它任意键保留数据服务 [n]: " CONFIRM
fi
if [ "${CONFIRM:-n}" = "y" ]; then
  if command -v docker >/dev/null 2>&1; then
    for c in $DATA_CONTAINERS; do
      if docker inspect "$c" >/dev/null 2>&1; then
        log "停止容器 $c"
        docker stop "$c" >/dev/null
      else
        log "容器 $c 不存在（可能用外部实例），跳过"
      fi
    done
  else
    warn "本机无 docker，无法停数据服务容器；若为外部实例请自行处理"
  fi
else
  log "数据服务保持运行（Redis/MySQL/Milvus 未动）"
fi

log "完成。日志保留在 $LOG_DIR/（backend.log / frontend.log）"
