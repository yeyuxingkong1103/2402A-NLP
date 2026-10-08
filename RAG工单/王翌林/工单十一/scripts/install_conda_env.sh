#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 工单：人工智能NLP-RAG-基于PDF文档的问答系统
# scripts/install_conda_env.sh — 环境安装脚本
# 创建 conda 环境 rag_pdf_qa (Python 3.10) + PyTorch cu121 + requirements.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "📦 安装环境 — rag_pdf_qa (Python 3.10)"
echo "   工单：人工智能NLP-RAG-基于PDF文档的问答系统"
echo "============================================================"

# ========== 1. 检测 conda ==========
if ! command -v conda &>/dev/null; then
    echo "⚠️  conda 未安装，尝试安装 Miniconda..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    bash -c "wget -q $MINICONDA_URL -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $HOME/miniconda3 && rm /tmp/miniconda.sh"
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# ========== 2. 创建 conda 环境 ==========
ENV_NAME="rag_pdf_qa"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  conda env '$ENV_NAME' 已存在，删除重建..."
    conda env remove -n "$ENV_NAME" -y
fi

echo "📥 创建 conda env: $ENV_NAME (Python 3.10)..."
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ========== 3. PyTorch cu121 ==========
echo "📥 安装 PyTorch cu121..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# ========== 4. 项目依赖 ==========
echo "📥 安装项目依赖 (pip install -r requirements.txt)..."
pip install -r requirements.txt

# ========== 5. 验证 ==========
echo ""
echo "✅ 环境安装完成！"
echo ""
echo "📋 验证信息："
python --version
python -c "import torch; print(f'  PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python -c "import fastapi; print(f'  FastAPI: {fastapi.__version__}')"
python -c "import streamlit; print(f'  Streamlit: {streamlit.__version__}')"
python -c "import pymilvus; print(f'  pymilvus: {pymilvus.__version__}')"
python -c "import ragas; print(f'  RAGAS: {ragas.__version__}')"
python -c "import sentence_transformers; print(f'  Sentence-Transformers: {sentence_transformers.__version__}')"

echo ""
echo "🚀 使用方法："
echo "  conda activate $ENV_NAME"
echo "  bash scripts/init_all.sh   # 初始化 MySQL + Milvus + 入库"
echo "  bash scripts/start.sh      # 启动 FastAPI + Streamlit"
echo "  bash scripts/stop.sh       # 停止服务"
