#!/bin/bash
set -e
echo "========== 环境安装 =========="

echo "[1/5] 检查 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  警告: 未检测到GPU，CPU模式运行"
fi

echo "[2/5] 检查 conda..."
if ! command -v conda &> /dev/null; then
    echo "错误: 未安装conda"
    exit 1
fi

echo "[3/5] 创建 conda 环境 rag-chat..."
conda env list | grep -q rag-chat || conda create -n rag-chat python=3.11 -y

echo "[4/5] 安装依赖..."
source "$(conda info --base)/activate" rag-chat
pip install -r requirements.txt
pip install -r requirements-local.txt

echo "[5/5] 验证..."
python -c "import fitz; print('PyMuPDF OK')"
python -c "import pymysql; print('pymysql OK')"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
python -c "from langchain.schema import Document; print('LangChain OK')"

echo "安装完成。启动: bash start.sh"
