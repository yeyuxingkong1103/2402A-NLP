#!/bin/bash
# 工单编号：人工智能NLP-RAG-Query理解优化任务
# install.sh - 安装脚本（Linux/WSL2）

set -e

echo "=========================================="
echo "  RAG-PDF 问答系统 - 安装依赖"
echo "=========================================="

# 创建 conda 环境（可选）
if command -v conda &> /dev/null; then
    echo "[*] 检测到 conda，创建环境 rag_query..."
    conda create -n rag_query python=3.10 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate rag_query
else
    echo "[!] 未检测到 conda，使用系统 Python"
fi

# 安装 Python 依赖
echo "[*] 安装 Python 依赖..."
pip install -r requirements.txt

# 检查 Milvus
echo "[*] 检查 Milvus 连接..."
python -c "
from pymilvus import connections
try:
    connections.connect(host='localhost', port='19530')
    print('Milvus 连接成功')
except Exception as e:
    print(f'Milvus 连接失败: {e}')
    print('请确保 Milvus 已启动（docker-compose up -d）')
"

echo ""
echo "=========================================="
echo "  安装完成！"
echo "  启动服务: bash start.sh"
echo "=========================================="
