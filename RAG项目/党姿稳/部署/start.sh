#!/usr/bin/env bash
# ============================================================
# start.sh — 启动服务
#
#   bash scripts/start.sh            # 同时启动 FastAPI 与 Streamlit
#   bash scripts/start.sh api        # 只启动 FastAPI
#   bash scripts/start.sh web        # 只启动 Streamlit
#
# 环境变量：
#   PYTHON      指定解释器，默认自动探测（conda > 项目 venv > python3）
#   API_PORT    FastAPI 端口，默认 8080
#   WEB_PORT    Streamlit 端口，默认 8501
#   LOG_LEVEL   日志级别，默认 INFO
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
API_PORT="${API_PORT:-8080}"
WEB_PORT="${WEB_PORT:-8501}"
CONDA_ENV="${CONDA_ENV:-rag-chat}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

cd "$PROJECT_DIR"
mkdir -p logs

# 选择 Python 解释器：优先环境变量 PYTHON，其次 conda 环境，其次项目内 venv，最后系统 Python
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v conda >/dev/null 2>&1 && conda env list | grep -qE "^${CONDA_ENV}\s"; then
    PYTHON="conda run --no-capture-output -n ${CONDA_ENV} python"
  elif [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
  elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    fail "未找到可用的 Python 解释器"
    exit 1
  fi
fi
info "使用解释器：$PYTHON"

# 启动前自检
info "检查配置"
$PYTHON config.py || warn "配置检查有告警，服务仍会尝试启动"

if ! $PYTHON -c "import fastapi" 2>/dev/null; then
  fail "缺少依赖，请先执行：pip install -r requirements.txt"
  exit 1
fi

PIDS=()

cleanup() {
  echo
  info "正在停止服务..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  exit 0
}
trap cleanup INT TERM

start_api() {
  info "启动 FastAPI（端口 ${API_PORT}），日志：logs/api.log"
  $PYTHON -m uvicorn server:app --host 0.0.0.0 --port "$API_PORT" \
    >> logs/api.log 2>&1 &
  PIDS+=($!)
  ok "FastAPI 已启动：http://0.0.0.0:${API_PORT}  （接口文档：/docs）"
}

start_web() {
  info "启动 Streamlit（端口 ${WEB_PORT}），日志：logs/web.log"
  $PYTHON -m streamlit run streamlit_app.py \
    --server.port "$WEB_PORT" --server.address 0.0.0.0 --server.headless true \
    >> logs/web.log 2>&1 &
  PIDS+=($!)
  ok "Streamlit 已启动：http://0.0.0.0:${WEB_PORT}"
}

case "$TARGET" in
  api)  start_api ;;
  web)  start_web ;;
  all)  start_api; sleep 2; start_web ;;
  *)    fail "未知参数：$TARGET（可选 api / web / all）"; exit 1 ;;
esac

echo
ok "服务已就绪，按 Ctrl+C 停止"
wait
