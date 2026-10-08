#!/bin/bash
# 工单13：RAG性能瓶颈识别与优化 - 环境安装脚本

set -e

echo "========================================"
echo "RAG性能瓶颈识别与优化 - 环境安装"
echo "========================================"

# 创建conda环境
conda create -n rag-perf python=3.10 -y
conda activate rag-perf

# 安装依赖
pip install -r requirements.txt

echo ""
echo "环境安装完成！"
echo "激活环境：conda activate rag-perf"
echo "运行测试：python benchmark.py"
echo "========================================"
