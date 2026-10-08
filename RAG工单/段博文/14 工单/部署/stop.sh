#!/bin/bash
# 工单14：停止流水线残留进程
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="${SCRIPT_DIR}/../研发/src/results/pipeline.pid"

if [ -f "${PID_FILE}" ]; then
    PID="$(cat "${PID_FILE}")"
    if kill -0 "${PID}" 2>/dev/null; then
        kill "${PID}" && echo "已停止进程 ${PID}"
    fi
    rm -f "${PID_FILE}"
else
    echo "无 PID 记录，尝试清理同名 python 流水线进程..."
    pkill -f "rag_qa.py" 2>/dev/null && echo "已停止 rag_qa.py" || true
    pkill -f "deepdoc_ocr_parser.py" 2>/dev/null && echo "已停止 deepdoc_ocr_parser.py" || true
fi
echo "停止操作完成"
