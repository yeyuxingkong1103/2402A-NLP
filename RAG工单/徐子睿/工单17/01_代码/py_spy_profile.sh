#!/usr/bin/env bash
# 工单 17 · py-spy CPU 热点采样（对 RAGFlow API Server 进程）
# 用法：bash py_spy_profile.sh <pid> [duration_seconds]
set -e
PID=${1:?usage: py_spy_profile.sh <pid> [duration]}
DUR=${2:-30}

# 1) 采样火焰图（离线分析 CPU 热点函数）
py-spy record -o profile_${PID}.svg --pid "$PID" --duration "$DUR" --rate 100

# 2) 实时 top（交互观察热点）
# py-spy top --pid "$PID"

# 3) 关键：找出并发时反复初始化重量级对象的调用栈
#    （如 DeepDoc 解析器 / RERANK 模型 / embedding 模型）
# py-spy dump --pid "$PID" | grep -E "PdfParser|Rerank|LLMBundle|__init__"

echo "saved profile_${PID}.svg"
