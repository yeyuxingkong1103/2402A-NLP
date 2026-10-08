#!/bin/bash
set -e
ENV_NAME="rag_wo15"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
echo "===== 1. 创建 conda 环境 ====="
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"
echo "===== 2. 安装依赖 ====="
pip install -r "${SCRIPT_DIR}/requirements.txt"
echo "===== 3. Playwright 浏览器 ====="
playwright install chromium || true
echo "===== 4. 检查本地模型 ====="
MODEL_DIR="${LOCAL_EMBED_MODEL:-/opt/models/bge-m3}"
[ -d "${MODEL_DIR}" ] && echo "模型: ${MODEL_DIR}" || echo "警告: 未找到 ${MODEL_DIR}，请设置 LOCAL_EMBED_MODEL"
echo "===== 安装完成 ====="
