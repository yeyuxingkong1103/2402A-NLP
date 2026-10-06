#!/bin/bash
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 启动脚本：后台启动 FastAPI 服务

set -e

echo "=========================================="
echo "  启动 RAG-PDF 问答系统"
echo "=========================================="

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rag_pdf_qa

# 进入 src 目录
cd "$(dirname "$0")/../研发/src"

# 后台启动
echo "[启动] 服务运行在 http://0.0.0.0:8000"
nohup python main.py > ../../rag.log 2>&1 &
echo $! > ../../rag.pid

echo "[完成] 服务已启动，PID: $(cat ../../rag.pid)"
echo "[日志] tail -f ../../rag.log"
echo "[停止] bash stop.sh"
