#!/bin/bash
set -e
ENV_NAME="rag_wo16"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"
pip install -r "${SCRIPT_DIR}/requirements.txt"
playwright install chromium || true
echo "===== 安装完成 ====="
echo "CPU环境: python 研发/src/mock_train.py 模拟训练"
echo "GPU环境: llamafactory-cli train 研发/src/lora_qwen_vl_industrial.yaml"
