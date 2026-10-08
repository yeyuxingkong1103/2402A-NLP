#!/bin/bash
# 文档质检 API 服务启动脚本
set -e

ENV_NAME="doc_quality"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${1:-8380}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR/研发/src"
echo "启动质检 API 服务，端口: $PORT"
echo "端点: POST http://0.0.0.0:$PORT/v1/document/quality-inspection"

nohup python api_server.py "$PORT" > "$PROJECT_DIR/部署/api_server.log" 2>&1 &
echo $! > "$SCRIPT_DIR/api_server.pid"
echo "服务已启动，PID: $(cat "$SCRIPT_DIR/api_server.pid")"
echo "日志: $PROJECT_DIR/部署/api_server.log"
