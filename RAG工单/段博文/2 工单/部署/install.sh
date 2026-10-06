#!/bin/bash
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 安装脚本（Linux）：conda 环境 + CUDA 检查 + Python 依赖 + Milvus(Docker)
set -e

ENV_NAME="rag_pdf_qa_v2"
PY_VERSION="3.11"

echo "========== [1/5] 检查 conda =========="
if ! command -v conda &> /dev/null; then
    echo "未检测到 conda，请先安装 Miniconda/Anaconda："
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
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
    echo "检测到 GPU，如需 GPU 加速可设置：export EMBED_DEVICE=cuda RERANK_DEVICE=cuda"
else
    echo "未检测到 NVIDIA GPU，使用 CPU 推理（本项目已在 CPU 验证通过）"
fi

echo "========== [4/5] 安装 Python 依赖 =========="
pip install -r "$(dirname "$0")/requirements.txt"

echo "========== [5/5] 启动 Milvus（Docker） =========="
if command -v docker &> /dev/null; then
    if ! docker ps | grep -q milvus; then
        echo "请先按《部署说明.md》启动 Milvus 容器"
    else
        echo "Milvus 容器运行中"
    fi
else
    echo "未检测到 docker，请先安装 Docker 并启动 Milvus"
fi

echo "========== 安装完成 =========="
echo "后续步骤："
echo "  1. 激活环境：conda activate ${ENV_NAME}"
echo "  2. 配置大模型密钥：export deepseek_api_key1=你的key"
echo "     export deepseek_base_url=https://api.deepseek.com/v1"
echo "  3. 下载本地模型：bge-m3、bge-reranker-large（修改 src/config.py 中路径）"
echo "  4. 入库文档：python src/ingest_pdf.py"
echo "  5. 启动服务：bash deploy/start.sh"
