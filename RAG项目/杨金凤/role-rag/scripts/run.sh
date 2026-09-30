#!/usr/bin/env bash
# 启动服务：检查 venv 与外部依赖，后台运行 uvicorn 并写入 pidfile。
# 外部依赖不可用仅警告不中断（Redis 降级无记忆、Milvus 降级 Chroma）。
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="0.0.0.0"; PORT="8000"; FOREGROUND=false
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --foreground) FOREGROUND=true; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done
[ -x .venv/bin/uvicorn ] || { echo "错误：未找到 .venv，请先运行 bash scripts/install.sh"; exit 1; }
if command -v redis-cli >/dev/null 2>&1 && redis-cli ping >/dev/null 2>&1; then
  echo "[OK] Redis 已运行"
else
  echo "[警告] Redis 不可用，会话记忆将降级为无记忆"
fi
use_milvus=$(grep -E '^USE_MILVUS=' .env 2>/dev/null | cut -d= -f2- || true)
case "${use_milvus:-true}" in
  true|1|yes|TRUE|YES)
    milvus_uri=$(grep -E '^MILVUS_URI=' .env 2>/dev/null | cut -d= -f2- || true)
    milvus_uri="${milvus_uri:-http://127.0.0.1:19530}"
    if curl -sf "$milvus_uri/healthz" >/dev/null 2>&1; then
      echo "[OK] Milvus 已运行（$milvus_uri）"
    else
      echo "[警告] Milvus 不可用（$milvus_uri），未部署可设 USE_MILVUS=false 降级 Chroma"
    fi ;;
  *) echo "[信息] USE_MILVUS=false，使用 Chroma 本地向量库" ;;
esac
mkdir -p logs
PIDFILE=".uvicorn.pid"
if [ "$FOREGROUND" = true ]; then
  echo "前台启动 uvicorn（Ctrl-C 停止）：http://$HOST:$PORT"
  exec .venv/bin/uvicorn api:app --host "$HOST" --port "$PORT"
fi
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "错误：服务已在运行（PID $(cat "$PIDFILE")），先 bash scripts/shutdown.sh 停止。"; exit 1
fi
nohup .venv/bin/uvicorn api:app --host "$HOST" --port "$PORT" >> logs/uvicorn.log 2>&1 &
echo $! > "$PIDFILE"
echo "已后台启动（PID $(cat "$PIDFILE")），日志：logs/uvicorn.log"
echo "健康检查：curl http://$HOST:$PORT/health"
