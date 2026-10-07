#!/bin/bash
# 工单编号：人工智能NLP-RAG-Query理解优化任务
# start.sh - 启动脚本

set -e

echo "=========================================="
echo "  RAG-PDF 问答系统 - 启动服务"
echo "=========================================="

# 激活 conda 环境（如果存在）
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate rag_query 2>/dev/null || true
fi

# 设置环境变量
export MILVUS_URI=${MILVUS_URI:-"http://localhost:19530"}
export MILVUS_COLLECTION=${MILVUS_COLLECTION:-"rag_pdf_qa_v4"}
export EMBED_MODEL_PATH=${EMBED_MODEL_PATH:-"/path/to/bge-m3"}
export RERANK_MODEL=${RERANK_MODEL:-"/path/to/bge-reranker-large"}
export API_HOST=${API_HOST:-"0.0.0.0"}
export API_PORT=${API_PORT:-8000}

# 启动服务
echo "[*] 启动 FastAPI 服务..."
echo "    地址: http://$API_HOST:$API_PORT"
echo "    文档: http://$API_HOST:$API_PORT/docs"
echo ""

cd "$(dirname "$0")/../研发/src"
python main.py
