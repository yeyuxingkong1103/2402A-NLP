#!/bin/bash
# 工单十五：停止脚本
cd "$(dirname "$0")/.."
for pidf in data/pids/api.pid data/pids/streamlit.pid; do
  if [ -f "$pidf" ]; then
    pid=$(cat "$pidf")
    kill "$pid" 2>/dev/null && echo "已停止 $pidf (pid=$pid)" || echo "$pidf 进程不存在"
    rm -f "$pidf"
  fi
done
echo "=== 服务已停止 ==="
