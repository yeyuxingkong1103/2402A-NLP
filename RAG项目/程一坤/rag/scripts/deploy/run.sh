#!/usr/bin/env bash
# =============================================================================
# 法律 RAG 启动脚本（批次 33，配合 install.sh 使用）
#
# 流程：检查依赖服务 → 加载环境配置 → 起后端（uvicorn）→ 起前端（next start）
#       → 等健康检查通过 → 打印访问地址与日志路径
# 幂等：进程已在跑则不重复起（pid 文件 + kill -0 判重）。
# 失败：打印对应日志最后 20 行。
# 用法：run.sh [--env production|development|test] [--skip-deps-check]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$INSTALL_DIR/logs"
ENV_NAME="production"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
ENV_CONDA="${ENV_NAME_CONDA:-rag}"            # conda 环境名（与 install.sh 一致）
PY_BIN="$CONDA_BASE/envs/$ENV_CONDA/bin/python"
SKIP_DEPS_CHECK=false
DRY_RUN=false

if [ -t 1 ]; then
  C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_RED='\033[0;31m'; C_OFF='\033[0m'
else
  C_GREEN=''; C_YELLOW=''; C_RED=''; C_OFF=''
fi
log()  { printf "${C_GREEN}[run]%s %s${C_OFF}\n" "" "$1"; }
warn() { printf "${C_YELLOW}[run][警告]%s %s${C_OFF}\n" "" "$1"; }
die()  { printf "${C_RED}[run][失败]%s %s${C_OFF}\n" "" "$2" >&2; tail_log "$1" 2>/dev/null || true; exit 1; }
tail_log() { [ -f "$1" ] && { echo "---- $1 最后 20 行 ----"; tail -n 20 "$1"; } }

while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift ;;
    --skip-deps-check) SKIP_DEPS_CHECK=true ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$LOG_DIR"

# ---------------- 环境配置（按 ENVIRONMENT 选择文件，缺文件明确报错） ----------------
ENV_FILE="$INSTALL_DIR/.env.$ENV_NAME"
[ -f "$ENV_FILE" ] || { echo "缺少环境配置文件：$ENV_FILE（可用 --env development|test|production 切换）" >&2; exit 1; }
if ! grep -Eq "^ENVIRONMENT=$ENV_NAME\$" "$ENV_FILE"; then
  # 允许 test 文件里写 development（批次 33 裁决）；production/development 必须一致
  if [ "$ENV_NAME" != "test" ]; then
    echo "$ENV_FILE 内 ENVIRONMENT 取值与 --env $ENV_NAME 不一致，先核对文件" >&2; exit 1
  fi
fi
log "加载环境配置：$ENV_FILE"
# 与 backend/app/core/config.py 的最小解析语义保持一致：
# KEY=VALUE、跳过注释与空行。不用 source —— .env 值里的 <...> 占位符、
# 空格、$ 等字符会被 shell 当重定向/变量展开（实测 .env.production 第 20 行直接语法报错）
load_env_file() {
  local file="$1" line key value
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|\#*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"
    value="${line#*=}"
    export "$key=$value"
  done < "$file"
}
load_env_file "$ENV_FILE"
export ENVIRONMENT="$ENV_NAME"    # test 文件内是 development，进程级强制与所选文件一致
export BACKEND_ORIGIN="${BACKEND_ORIGIN:-http://127.0.0.1:$BACKEND_PORT}"

# ---------------- 端口占用检查 ----------------
# 进程判重工具需在端口检查前定义，避免已运行服务被重复判定为冲突。
is_running() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

port_owner() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :$port" 2>/dev/null | awk 'NR > 1 {print $NF; exit}'
  else
    return 1
  fi
}

check_port_available() {
  local port="$1" service="$2" owner owner_pid expected_pid_file
  owner="$(port_owner "$port" || true)"
  [ -n "$owner" ] || return 0
  if [[ "$owner" =~ ^[0-9]+$ ]]; then
    owner_pid="$owner"
  else
    owner_pid="$(printf '%s' "$owner" | sed -n 's/.*pid=\([^,]*\).*/\1/p' | tr -dc '0-9')"
  fi
  expected_pid_file="$LOG_DIR/$service.pid"
  if [ -f "$expected_pid_file" ] && [ "$(cat "$expected_pid_file")" = "$owner_pid" ] && is_running "$expected_pid_file"; then
    return 0
  fi
  die "$LOG_DIR/$service.log" "端口 $port 已被其他进程占用（占用者：$owner）。请换端口或修改前端 BACKEND_ORIGIN，禁止静默启动后导致 401。"
}

check_port_available "$BACKEND_PORT" backend
check_port_available "$FRONTEND_PORT" frontend

# ---------------- 依赖服务检查 ----------------
if $DRY_RUN; then
  log "[dry-run] 演练模式：只打印将执行的动作，不真正起进程；依赖检查/启动/健康检查全部跳过"
elif $SKIP_DEPS_CHECK; then
  warn "按 --skip-deps-check 跳过依赖服务检查"
else
  log "检查依赖服务（Redis/MySQL/Milvus）..."
  DEP_FAIL=""
  if ! "$PY_BIN" - <<'PYEOF' 2>/dev/null; then DEP_FAIL="$DEP_FAIL Redis"; fi
import os, redis
u = os.environ.get("REDIS_URL", "")
assert u, "REDIS_URL 未配置"
redis.Redis.from_url(u, socket_connect_timeout=3).ping()
PYEOF
  if ! "$PY_BIN" - <<'PYEOF' 2>/dev/null; then DEP_FAIL="$DEP_FAIL MySQL"; fi
import os, pymysql
pymysql.connect(
    host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
    port=int(os.environ.get("MYSQL_PORT", "3306")),
    user=os.environ.get("MYSQL_USER", ""),
    password=os.environ.get("MYSQL_PASSWORD", ""),
    database=os.environ.get("MYSQL_DATABASE", "legal_rag"),
    connect_timeout=3,
).close()
PYEOF
  if ! "$PY_BIN" - <<'PYEOF' 2>/dev/null; then DEP_FAIL="$DEP_FAIL Milvus"; fi
import os
from pymilvus import connections
connections.connect(
    host=os.environ.get("MILVUS_HOST", "127.0.0.1"),
    port=os.environ.get("MILVUS_PORT", "19530"),
)
connections.disconnect("default")
PYEOF
  [ -z "$DEP_FAIL" ] || die "$LOG_DIR/backend.log" "依赖服务不可用：$DEP_FAIL —— 先拉起对应服务（Docker：docker start rag-redis rag-mysql rag-milvus）"
  log "依赖服务全部可达"
fi

# ---------------- 进程启动工具 ----------------

start_backend() {
  if $DRY_RUN; then
    log "[dry-run] 将启动后端：cd backend && uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT（pid 文件 $LOG_DIR/backend.pid）"
    return 0
  fi
  if is_running "$LOG_DIR/backend.pid"; then
    log "后端已在运行（pid $(cat "$LOG_DIR/backend.pid")），跳过"
    return 0
  fi
  log "启动后端：uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT"
  (cd "$INSTALL_DIR/backend" && nohup "$CONDA_BASE/envs/$ENV_CONDA/bin/uvicorn" \
      app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" \
      >> "$LOG_DIR/backend.log" 2>&1 & echo $! > "$LOG_DIR/backend.pid")
  sleep 1
  is_running "$LOG_DIR/backend.pid" || die "$LOG_DIR/backend.log" "后端进程启动即退出，日志见下"
}

start_frontend() {
  if $DRY_RUN; then
    log "[dry-run] 将启动前端：cd frontend && npx next start --port $FRONTEND_PORT（BACKEND_ORIGIN=$BACKEND_ORIGIN，pid 文件 $LOG_DIR/frontend.pid；未构建时先 npm run build）"
    return 0
  fi
  if is_running "$LOG_DIR/frontend.pid"; then
    log "前端已在运行（pid $(cat "$LOG_DIR/frontend.pid")），跳过"
    return 0
  fi
  if [ ! -d "$INSTALL_DIR/frontend/node_modules" ]; then
    die "$LOG_DIR/frontend.log" "frontend/node_modules 不存在，先执行：cd frontend && npm install && npm run build"
  fi
  if [ ! -f "$INSTALL_DIR/frontend/.next/BUILD_ID" ]; then
    warn "未发现生产构建产物（frontend/.next），先执行构建（约 1~3 分钟）..."
    (cd "$INSTALL_DIR/frontend" && npm run build >> "$LOG_DIR/frontend.log" 2>&1)
  fi
  log "启动前端：next start --port $FRONTEND_PORT（BACKEND_ORIGIN=$BACKEND_ORIGIN）"
  (cd "$INSTALL_DIR/frontend" && nohup npx next start --port "$FRONTEND_PORT" \
      >> "$LOG_DIR/frontend.log" 2>&1 & echo $! > "$LOG_DIR/frontend.pid")
  sleep 1
  is_running "$LOG_DIR/frontend.pid" || die "$LOG_DIR/frontend.log" "前端进程启动即退出，日志见下"
}

wait_backend_healthy() {
  if $DRY_RUN; then log "[dry-run] 将等待 http://127.0.0.1:$BACKEND_PORT/health/live 与 /health/ready 通过（最多 60 秒）"; return 0; fi
  log "等待后端健康检查（最多 60 秒）..."
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$BACKEND_PORT/health/live" >/dev/null 2>&1 \
      && curl -fsS "http://127.0.0.1:$BACKEND_PORT/health/ready" >/dev/null 2>&1; then
      log "后端健康检查通过（/health/live + /health/ready）"
      return 0
    fi
    if ! is_running "$LOG_DIR/backend.pid"; then
      die "$LOG_DIR/backend.log" "后端在健康检查期间退出，日志见下"
    fi
    sleep 1
  done
  die "$LOG_DIR/backend.log" "后端 60 秒内未通过健康检查，日志见下"
}

wait_frontend_healthy() {
  if $DRY_RUN; then log "[dry-run] 将等待 http://127.0.0.1:$FRONTEND_PORT/ 就绪（最多 30 秒）"; return 0; fi
  log "等待前端就绪（最多 30 秒）..."
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$FRONTEND_PORT/" >/dev/null 2>&1; then
      log "前端就绪"
      return 0
    fi
    if ! is_running "$LOG_DIR/frontend.pid"; then
      die "$LOG_DIR/frontend.log" "前端在就绪检查期间退出，日志见下"
    fi
    sleep 1
  done
  die "$LOG_DIR/frontend.log" "前端 30 秒内未就绪，日志见下"
}

start_backend
start_frontend
wait_backend_healthy
wait_frontend_healthy

printf "${C_GREEN}[run] ===== 启动完成 =====${C_OFF}\n"
echo "  前端访问： http://<host>:$FRONTEND_PORT"
echo "  后端API：  http://<host>:$BACKEND_PORT（问答 SSE：/api/v1/chat/stream）"
echo "  健康检查： http://127.0.0.1:$BACKEND_PORT/health/live  /health/ready"
echo "  日志路径： $LOG_DIR/backend.log  $LOG_DIR/frontend.log"
echo "  停止服务： $SCRIPT_DIR/shutdown.sh"
