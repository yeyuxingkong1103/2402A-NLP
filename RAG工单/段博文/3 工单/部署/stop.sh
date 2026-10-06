#!/bin/bash
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 停止脚本：停止 FastAPI 服务

set -e

PID_FILE="$(dirname "$0")/../rag.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "[停止] 停止服务 PID: $PID"
        kill $PID
        rm "$PID_FILE"
        echo "[完成] 服务已停止"
    else
        echo "[警告] 进程 $PID 不存在"
        rm "$PID_FILE"
    fi
else
    echo "[警告] PID 文件不存在，尝试查找进程..."
    pkill -f "python main.py" || echo "[完成] 未找到运行中的服务"
fi
