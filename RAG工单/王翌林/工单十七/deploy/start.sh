#!/bin/bash
# ============================================================
# 工单十七 · 启动脚本（后台启动 RAG API 优化版服务）
# 用法： bash deploy/start.sh [port]  默认 8017
# ============================================================
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

PORT="${1:-8017}"
LOG_DIR="$(pwd)/data/logs"
mkdir -p "${LOG_DIR}"
PID_FILE="$(pwd)/data/logs/rag-api.pid"
LOG_FILE="${LOG_DIR}/rag-api.log"

# 激活 conda 环境（若存在）
CONDA_BASE="$(conda info --base 2>/dev/null || echo '')"
if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate rag17 2>/dev/null || true
fi

# 已运行则提示
if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "服务已在运行，PID=$(cat "${PID_FILE}")"
  exit 0
fi

echo "启动 RAG API (优化版) 端口=${PORT} ..."
export LLM_LATENCY="${LLM_LATENCY:-0.8}"
nohup python -m uvicorn src.api:app --host 0.0.0.0 --port "${PORT}" \
  --workers 1 --log-level info >"${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"
sleep 3
if kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "启动成功 PID=$(cat "${PID_FILE}") 日志=${LOG_FILE}"
  echo "健康检查: curl http://127.0.0.1:${PORT}/health"
else
  echo "启动失败，请查看 ${LOG_FILE}"
  exit 1
fi
