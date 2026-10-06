#!/bin/bash
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 安装脚本（Linux）：conda 环境 + CUDA 检查 + Python 依赖 + Milvus(Docker)
set -e

ENV_NAME="rag_pdf_qa"
PY_VERSION="3.11"

echo "========== [1/5] 检查 conda =========="
if ! command -v conda &> /dev/null; then
    echo "未检测到 conda，请先安装 Miniconda/Anaconda"
    exit 1
fi
conda --version

echo "========== [2/5] 创建 conda 环境 ${ENV_NAME} (Python ${PY_VERSION}) =========="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "环境 ${ENV_NAME} 已存在，跳过创建"
else
    conda create -n ${ENV_NAME} python=${PY_VERSION} -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "========== [3/5] 检查 CUDA（可选，本系统默认 CPU 推理） =========="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "检测到 GPU，可设置：export EMBED_DEVICE=cuda RERANK_DEVICE=cuda"
else
    echo "未检测到 NVIDIA GPU，使用 CPU 推理"
fi

echo "========== [4/5] 安装 Python 依赖 =========="
pip install -r "$(dirname "$0")/requirements.txt"

echo "========== [5/5] 检查 Milvus（Docker） =========="
if command -v docker &> /dev/null; then
    docker ps | grep -q milvus && echo "Milvus 运行中" || echo "请按《部署说明.md》启动 Milvus 容器"
else
    echo "未检测到 docker，请先安装 Docker"
fi

echo "========== 安装完成 =========="
echo "后续步骤："
echo "  1. 激活环境：conda activate ${ENV_NAME}"
echo "  2. 配置密钥：export deepseek_api_key1=你的key"
echo "  3. 准备模型：bge-m3、bge-reranker-large（改 src/config.py 路径）"
echo "  4. 入库：python src/ingest_pdf.py"
echo "  5. 启动：bash deploy/start.sh"
