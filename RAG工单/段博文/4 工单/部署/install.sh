#!/bin/bash
# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 安装脚本：创建 conda 环境并安装依赖

set -e

echo "=========================================="
echo "  RAG-PDF 问答系统（图像内容解析增强版）"
echo "  安装脚本"
echo "=========================================="

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "[错误] 未检测到 conda，请先安装 Miniconda 或 Anaconda"
    exit 1
fi

# 创建环境
echo "[1/5] 创建 conda 环境 rag_pdf_qa..."
conda create -n rag_pdf_qa python=3.11 -y

# 激活环境
echo "[2/5] 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rag_pdf_qa

# 安装依赖
echo "[3/5] 安装 Python 依赖（含 RapidOCR 多模态 OCR）..."
pip install -r requirements.txt

# 检查 CUDA
echo "[4/5] 检查 CUDA..."
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}, 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A（CPU 模式）\"}')"

# 启动 Milvus（如果本地运行）
echo "[5/5] 检查 Milvus..."
if command -v docker &> /dev/null; then
    echo "检测到 Docker，启动 Milvus..."
    docker start milvus-standalone milvus-etcd milvus-minio 2>/dev/null || echo "Milvus 容器不存在，请先 docker compose up -d"
else
    echo "未检测到 Docker，请确保 Milvus 服务已启动"
fi

echo "=========================================="
echo "  安装完成！"
echo "  激活环境: conda activate rag_pdf_qa"
echo "  启动服务: bash start.sh"
echo "=========================================="
