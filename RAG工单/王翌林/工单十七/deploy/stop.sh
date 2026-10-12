#!/bin/bash
# ============================================================
# 工单十七 · 结束脚本（停止 RAG API 服务）
# 用法： bash deploy/stop.sh
# ============================================================
cd "$(dirname "$0")/.."
PID_FILE="$(pwd)/data/logs/rag-api.pid"

if [ ! -f "${PID_FILE}" ]; then
  echo "未找到 PID 文件，尝试按端口查找进程..."
  PIDS=$(pgrep -f "uvicorn src.api:app" || true)
else
  PIDS="$(cat "${PID_FILE}")"
fi

if [ -z "${PIDS}" ]; then
  echo "未发现运行中的 RAG API 服务。"
  rm -f "${PID_FILE}"
  exit 0
fi

for pid in ${PIDS}; do
  if kill -0 "${pid}" 2>/dev/null; then
    echo "停止进程 ${pid} ..."
    kill "${pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${pid}" 2>/dev/null || true
  fi
done
rm -f "${PID_FILE}"
echo "已停止。"
