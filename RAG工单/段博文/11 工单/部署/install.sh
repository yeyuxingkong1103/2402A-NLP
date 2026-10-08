#!/bin/bash
# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-Embedding模型微调任务
# ============================================================
# Embedding 模型微调环境安装脚本
# 适用：Linux / WSL2（conda + CPU，GPU 环境自动启用 CUDA）
# ============================================================
set -e

ENV_NAME="rag_embed_ft"
PY_VER="3.10"

echo "========== Embedding 微调环境安装 =========="

# 1. 创建 conda 环境
if command -v conda >/dev/null 2>&1; then
    if ! conda env list | grep -q "$ENV_NAME"; then
        echo "[1/3] 创建 conda 环境：$ENV_NAME (python $PY_VER)"
        conda create -y -n "$ENV_NAME" python="$PY_VER"
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "未检测到 conda，使用当前 Python 环境：$(python --version)"
fi

# 2. 安装依赖
echo "[2/3] 安装 Python 依赖 ..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. GPU 检测
echo "[3/3] 检测 CUDA ..."
python - <<'EOF'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "========== 安装完成 =========="
echo "运行顺序："
echo "  python gen_passages.py      # 段落采样"
echo "  python gen_qa_pairs.py      # LLM 生成问答对"
echo "  python build_dataset.py     # 数据划分+困难负例"
echo "  python finetune.py          # 微调前后评估+训练"
