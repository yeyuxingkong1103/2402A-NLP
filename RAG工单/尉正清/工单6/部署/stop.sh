#!/bin/bash
# 工单编号：人工智能NLP-RAG-混合检索任务
# 结束脚本：停止 Gradio 服务
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/app.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "未找到 PID 文件，服务可能未启动。"
    # 兜底：按进程名清理
    if pkill -f "python -u app.py" 2>/dev/null; then
        echo "已按进程名停止 app.py"
    fi
    exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
    echo "==> 停止服务（PID ${PID}）"
    kill "${PID}"
    for _ in $(seq 1 10); do
        kill -0 "${PID}" 2>/dev/null || break
        sleep 1
    done
    # 优雅退出超时就强杀
    if kill -0 "${PID}" 2>/dev/null; then
        echo "==> 进程未响应，强制结束"
        kill -9 "${PID}"
    fi
    echo "==> 已停止"
else
    echo "进程 ${PID} 不存在，清理 PID 文件"
fi
rm -f "${PID_FILE}"
