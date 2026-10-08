#!/bin/bash
# 工单14：环境安装脚本（Linux + conda，零付费API）
set -e

ENV_NAME="rag_wo14"
PY_VER="3.10"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "===== 1. 创建/激活 conda 环境 ${ENV_NAME} ====="
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi
conda activate "${ENV_NAME}"

echo "===== 2. 安装 Python 依赖 ====="
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "===== 3. 安装 Playwright 浏览器内核（截图用） ====="
playwright install chromium || echo "提示：浏览器内核安装失败不影响核心流程"

echo "===== 4. 检查本地模型目录 ====="
MODEL_DIR="${LOCAL_EMBED_MODEL:-/opt/models/bge-m3}"
if [ -d "${MODEL_DIR}" ]; then
    echo "本地 embedding 模型: ${MODEL_DIR}"
else
    echo "警告：未找到 ${MODEL_DIR}"
    echo "请下载 bge-m3 模型，并通过 export LOCAL_EMBED_MODEL=/你的路径 指定"
fi

echo "===== 安装完成 ====="
echo "运行 ${SCRIPT_DIR}/start.sh 启动解析与问答流水线"
