#!/bin/bash
# 工单编号：人工智能NLP-RAG-微调专用视觉语言模型
# 结束脚本：中止正在跑的微调流水线
#
# 训练是 GPU 密集任务，直接 kill 训练进程即可；PID 文件里存的是那条
# nohup bash 的进程号，子进程（python）要一并带走。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/pipeline.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "==> 没有找到 PID 文件，流水线应该没在跑"
    exit 0
fi

PID="$(cat "${PID_FILE}")"
if ! kill -0 "${PID}" 2>/dev/null; then
    echo "==> PID ${PID} 已经不在，清理 PID 文件"
    rm -f "${PID_FILE}"
    exit 0
fi

echo "==> 中止流水线（PID ${PID}）及其子进程"
pkill -TERM -P "${PID}" 2>/dev/null || true
kill -TERM "${PID}" 2>/dev/null || true
sleep 3
pkill -KILL -P "${PID}" 2>/dev/null || true
kill -KILL "${PID}" 2>/dev/null || true

rm -f "${PID_FILE}"
echo "==> 已中止。显存若没释放，检查是否还有残留的 python 进程"
