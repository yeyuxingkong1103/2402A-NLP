#!/usr/bin/env bash
# 法律 RAG 服务启动脚本（Ubuntu / 算力云）
#
# 做四件事：
#   1. 检查基础环境（python / 项目结构）
#   2. 加载 .env（如果存在）
#   3. 尽力拉起 Redis / MySQL（装了才拉，没装就依赖代码里的降级路径）
#   4. 后台拉起 FastAPI，并把 PID 写到 run/api.pid
#
# 用法：
#   bash run.sh              # 正常启动
#   bash run.sh --offline    # 离线模式（无 key / 无 GPU 时先跑通）
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

RUN_DIR="$APP_DIR/run"
mkdir -p "$RUN_DIR"
PID_FILE="$RUN_DIR/api.pid"
LOG_FILE="$RUN_DIR/api.log"

OFFLINE_FLAG=""
if [[ "${1:-}" == "--offline" ]]; then
  OFFLINE_FLAG="--offline"
fi

echo "==> [1/4] 检查环境"
if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  echo "未找到 python，请先安装 Python 3.10+" >&2
  exit 1
fi
PYTHON_BIN="$(command -v python3 || command -v python)"

# 优先使用项目内虚拟环境
if [[ -x "$APP_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$APP_DIR/.venv/bin/python"
  echo "    使用虚拟环境: $PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "    使用 conda 环境: $CONDA_PREFIX"
else
  echo "    使用系统 Python: $PYTHON_BIN"
fi

echo "==> [2/4] 加载配置"
if [[ -f "$APP_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$APP_DIR/.env"
  set +a
  echo "    已加载 .env"
else
  echo "    未发现 .env（可复制 .env.example 后填写）"
fi
export API_PORT="${API_PORT:-8000}"
export API_HOST="${API_HOST:-0.0.0.0}"

echo "==> [3/4] 启动依赖服务（Redis / MySQL，装了才启动）"
if command -v redis-server >/dev/null 2>&1 && ! redis-cli ping >/dev/null 2>&1; then
  echo "    启动 Redis …"
  redis-server --daemonize yes || echo "    Redis 启动失败，代码会自动降级为内存记忆"
else
  echo "    Redis 已就绪或未安装（未安装时自动降级为内存记忆）"
fi

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q '^mysql\.service'; then
  echo "    确保 MySQL 运行 …"
  sudo systemctl start mysql 2>/dev/null || echo "    MySQL 启动失败，业务数据会自动降级为 SQLite"
else
  echo "    MySQL 未安装或非 systemd 管理（未配置时自动降级为 SQLite）"
fi

echo "==> [4/4] 启动服务"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "    服务已在运行 (PID $(cat "$PID_FILE"))，先执行 bash shutdown.sh 再启动" >&2
  exit 1
fi

nohup "$PYTHON_BIN" scripts/run_api.py --host "$API_HOST" --port "$API_PORT" $OFFLINE_FLAG \
  >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

sleep 2
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "    已启动，PID $(cat "$PID_FILE")"
  echo "    日志: $LOG_FILE"
  echo "    健康检查: curl http://127.0.0.1:${API_PORT}/health"
else
  echo "    启动失败，请查看日志: $LOG_FILE" >&2
  tail -n 30 "$LOG_FILE" >&2 || true
  exit 1
fi
