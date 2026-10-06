#!/bin/bash
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# 启动脚本（Linux）：启动优化版 RAG-PDF 问答服务
set -e
cd "$(dirname "$0")/../src"

# conda 环境
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate rag_pdf_qa_v2 2>/dev/null || echo "[提示] 未找到 conda 环境 rag_pdf_qa_v2，使用当前 Python"
fi

export MILVUS_URI="${MILVUS_URI:-http://localhost:19530}"
export API_PORT="${API_PORT:-8000}"
# 大模型密钥（必须提前配置）
# export deepseek_api_key1=你的key
# export deepseek_base_url=https://api.deepseek.com/v1

nohup python -m uvicorn main:app --host 0.0.0.0 --port ${API_PORT} > rag_pdf_qa_v2.log 2>&1 &
echo "服务已启动，PID=$!"
echo "访问 http://localhost:${API_PORT}/web/index.html"
echo "API 文档 http://localhost:${API_PORT}/docs"
echo "日志：src/rag_pdf_qa_v2.log"
