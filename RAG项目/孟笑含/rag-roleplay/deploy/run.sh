#!/usr/bin/env bash
# ============================================================
# RAG 角色扮演系统 · 启动脚本
# 用法：bash deploy/run.sh [--mock] [--port 8000]
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
PORT=8000
MOCK=false
for arg in "$@"; do
  case $arg in
    --mock) MOCK=true ;;
    --port=*) PORT="${arg#*=}" ;;
  esac
done

log() { echo -e "\033[1;32m[run]\033[0m $*"; }
die() { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }

VENV_PY="$PROJECT_DIR/venv/bin/python"
PID_FILE="$PROJECT_DIR/.server.pid"
LOG_FILE="$PROJECT_DIR/server.log"

[ -x "$VENV_PY" ] || die "venv 不存在，请先执行 bash deploy/install.sh"
[ -f .env ] || die ".env 不存在，请先执行 bash deploy/install.sh 并填写配置"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  die "服务已在运行（PID $(cat "$PID_FILE")），先执行 bash deploy/shutdown.sh"
fi

# 建库（幂等，MySQL 需可达）
if [ "${SKIP_DB_INIT:-0}" != "1" ]; then
  log "初始化数据库（幂等）"
  set -a; . ./.env; set +a   # 加载 .env 为环境变量（stdin 脚本内 python-dotenv 不可用）
  "$VENV_PY" - <<'EOF' || echo "[warn] 建库失败，请确认 MySQL 配置与可达性"
import pymysql, os
conn = pymysql.connect(
    host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
    port=int(os.environ.get("MYSQL_PORT", "3306")),
    user=os.environ.get("MYSQL_USER", "root"),
    password=os.environ.get("MYSQL_PASSWORD", ""),
    charset="utf8mb4",
)
with conn.cursor() as cur:
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {os.environ.get('MYSQL_DATABASE', 'rag_roleplay')} CHARACTER SET utf8mb4")
conn.commit(); conn.close()
print("database ready")
EOF
fi

log "启动服务（端口 $PORT，MOCK=$MOCK）"
if $MOCK; then
  LLM_MOCK=true nohup "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" >> "$LOG_FILE" 2>&1 &
else
  nohup "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" >> "$LOG_FILE" 2>&1 &
fi
echo $! > "$PID_FILE"

sleep 5
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  log "服务已启动：http://127.0.0.1:$PORT/docs （日志 $LOG_FILE）"
  curl -sf "http://127.0.0.1:$PORT/docs" >/dev/null && log "健康检查通过 ✓"
else
  die "启动失败，查看日志：tail -50 $LOG_FILE"
fi
