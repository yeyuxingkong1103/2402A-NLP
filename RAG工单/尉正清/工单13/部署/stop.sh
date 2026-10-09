#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化
# 结束脚本：停压测服务与所有分析工具
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/api.pid"

for name in api.pid; do
    f="${SCRIPT_DIR}/logs/${name}"
    [ -f "$f" ] || continue
    PID="$(cat "$f")"
    if kill -0 "${PID}" 2>/dev/null; then
        echo "==> 停止 ${name%.pid}（PID ${PID}）"
        kill "${PID}" 2>/dev/null || true
        for _ in $(seq 1 10); do kill -0 "${PID}" 2>/dev/null || break; sleep 1; done
        kill -9 "${PID}" 2>/dev/null || true
    fi
    rm -f "$f"
done

# snakeviz 是独立进程，按端口找
if curl -s -m 2 -o /dev/null http://127.0.0.1:8080/snakeviz/ 2>/dev/null; then
    echo "==> 停止 snakeviz"
    pkill -f "snakeviz" 2>/dev/null || true
fi
# 兜底：按进程名清理本项目起的服务
pkill -f "工单13.*api\.py" 2>/dev/null || true
echo "==> 已停止"
