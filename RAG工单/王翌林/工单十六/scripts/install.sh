#!/bin/bash
# 工单十六：环境安装脚本（Linux + Conda/CUDA 检查 + Python 依赖）
set -e
echo "=== 工单十六 环境安装 ==="

# 1. 检查 NVIDIA / CUDA
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv
else
  echo "[WARN] 未检测到 nvidia-smi，将使用 CPU（不推荐微调）"
fi

# 2. Python 依赖
PYTHON_BIN=${PYTHON_BIN:-python3}
echo "使用 Python: $($PYTHON_BIN --version)"
$PYTHON_BIN -m pip install --upgrade pip
$PYTHON_BIN -m pip install \
  torch transformers accelerate peft datasets \
  loguru pillow matplotlib numpy rouge-score nltk \
  pymupdf

# bitsandbytes（4bit 量化，可选；安装失败不影响 fp16 路径）
$PYTHON_BIN -m pip install bitsandbytes || echo "[WARN] bitsandbytes 安装失败，将回退 fp16"

echo "=== 依赖安装完成 ==="
echo "基座模型需放置于: /home/dabaie/models/Qwen2-VL-2B-Instruct"
