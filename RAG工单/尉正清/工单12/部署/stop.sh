#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-LightRAG优化
# 结束脚本：中止流水线，并按需关掉 neo4j / Docker
#
# 默认**只停容器、不删**：neo4j 容器与数据卷是复用的，删了图谱就没了。
# 想彻底删容器加 --purge（会丢图谱，需重建约 15 分钟）。
set -e

PURGE=0
[ "$1" = "--purge" ] && PURGE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/logs/pipeline.pid"

# ---------- 1. 中止流水线 ----------
if [ -f "${PID_FILE}" ]; then
    PID="$(cat "${PID_FILE}")"
    if kill -0 "${PID}" 2>/dev/null; then
        echo "==> 停止流水线（PID ${PID}）"
        kill -- -"${PID}" 2>/dev/null || kill "${PID}" 2>/dev/null || true
        for _ in $(seq 1 10); do kill -0 "${PID}" 2>/dev/null || break; sleep 1; done
        kill -9 -- -"${PID}" 2>/dev/null || kill -9 "${PID}" 2>/dev/null || true
        echo "==> 已停止"
    fi
    rm -f "${PID_FILE}"
else
    echo "未找到 PID 文件，流水线可能未启动。"
fi
pkill -f "build_kb.py|build_graph.py|run_qa.py|ragas_eval.py|compare.py" 2>/dev/null || true

# ---------- 2. neo4j ----------
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx neo4j; then
    if [ "${PURGE}" = "1" ]; then
        echo "==> 删除 neo4j 容器（图谱会丢失）"
        docker rm -f neo4j >/dev/null && echo "    已删除"
    else
        echo "==> 停止 neo4j 容器（数据保留，下次 start 会自动拉起）"
        docker stop neo4j >/dev/null && echo "    已停止"
    fi
fi

# ---------- 3. Docker Desktop（可选，省内存）----------
# neo4j 是唯一用到 Docker 的组件。跑完关掉 Docker Desktop 能收回好几个 GB，
# 但 WSL 虚拟机的内存不会立刻释放，要再 `wsl --shutdown` 才彻底归还。
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    read -r -p "是否同时关闭 Docker Desktop 以回收内存？[y/N] " ans
    if [ "${ans}" = "y" ] || [ "${ans}" = "Y" ]; then
        docker desktop stop >/dev/null 2>&1 && echo "==> Docker Desktop 已停止"
        wsl --shutdown 2>/dev/null && echo "==> WSL 虚拟机已关闭（内存已归还）"
    fi
fi
