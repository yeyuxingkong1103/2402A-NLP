#!/bin/bash
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 启动脚本：后台启动 Gradio 服务，日志写入 logs/app.log
set -e

ENV_NAME="${ENV_NAME:-rag_gd}"
PORT="${PORT:-7860}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../研发" && pwd)"
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
PID_FILE="${LOG_DIR}/app.pid"

mkdir -p "${LOG_DIR}"

# ---------- 检查是否已在运行 ----------
if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "服务已在运行（PID $(cat "${PID_FILE}")），如需重启请先执行 部署/stop.sh"
    exit 0
fi

# ---------- 检查环境变量 ----------
if [ -z "${DEEPSEEK_API_KEY}" ]; then
    echo "[警告] 未设置 DEEPSEEK_API_KEY，大模型调用会失败" >&2
fi

# ---------- 激活环境 ----------
if ! command -v conda >/dev/null 2>&1; then
    echo "[错误] 未找到 conda" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ---------- 启动 ----------
cd "${PROJECT_DIR}"
echo "==> 启动服务，端口 ${PORT}，日志 ${LOG_DIR}/app.log"
nohup python -u app.py > "${LOG_DIR}/app.log" 2>&1 &
echo $! > "${PID_FILE}"

sleep 5
if kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "==> 启动成功（PID $(cat "${PID_FILE}")）"
    echo "    访问地址：http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}"
    echo "    查看日志：tail -f ${LOG_DIR}/app.log"
else
    echo "[错误] 启动失败，日志如下：" >&2
    tail -20 "${LOG_DIR}/app.log" >&2
    rm -f "${PID_FILE}"
    exit 1
fi
