#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 结束脚本：中止正在跑的微调流水线
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/pipeline.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "未找到 PID 文件，流水线可能未启动。"
    # 兜底：按进程名清理（nohup bash -c 的子 shell）
    if pkill -f "finetune.py|evaluate.py|gen_qa.py|build_dataset.py" 2>/dev/null; then
        echo "已按进程名停止训练相关进程"
    fi
    exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
    echo "==> 停止流水线（PID ${PID}）"
    # 流水线是 bash -c 里串起来的，kill 父 shell 不会带走正在跑的 python，
    # 所以要按进程组结束（负号 = 整组）。
    kill -- -"${PID}" 2>/dev/null || kill "${PID}" 2>/dev/null || true
    for _ in $(seq 1 10); do
        kill -0 "${PID}" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "${PID}" 2>/dev/null; then
        echo "==> 进程未响应，强制结束"
        kill -9 -- -"${PID}" 2>/dev/null || kill -9 "${PID}" 2>/dev/null || true
    fi
    # python 子进程可能还挂着（nohup 起的组未必跟着走），补一刀
    pkill -f "研发/(build_dataset|evaluate|gen_qa|finetune)\.py" 2>/dev/null || true
    echo "==> 已停止"
else
    echo "进程 ${PID} 不存在，清理 PID 文件"
fi
rm -f "${PID_FILE}"

# 提示：向量缓存是按模型指纹存的，中断后重跑 部署/start.sh 会复用已算好的块，
# 不会从头再来（见 研发/retrieval.py 的分块落盘）。
