#!/bin/bash
# 文档质量评估 Skill - 环境安装脚本（Linux + conda）
set -e

ENV_NAME="doc_quality"
PY_VER="3.12"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "===== 创建/激活 conda 环境: $ENV_NAME ====="
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"

echo "===== 安装 Python 依赖 ====="
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "===== 安装 Playwright Chromium（用于截图，可选）====="
playwright install chromium || echo "Playwright 浏览器安装跳过（截图功能可选）"

echo "===== 安装完成 ====="
echo "Skill 位置: $PROJECT_DIR/研发/src/document_quality_assessment"
