#!/bin/bash
# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# ============================================================
# RAG 金融问答系统容器入口脚本
# ============================================================
set -e

echo "[entrypoint] 开始启动 RAG 金融问答系统..."

# 进入应用目录
cd /app/src

# 1. 等待 Milvus 就绪
echo "[entrypoint] 等待 Milvus 服务就绪..."
python - <<'EOF'
import time
import sys
from pymilvus import MilvusClient

milvus_uri = "http://milvus:19530"
max_retries = 30
for i in range(max_retries):
    try:
        client = MilvusClient(uri=milvus_uri)
        print(f"  Milvus 连接成功（第 {i+1} 次尝试）")
        break
    except Exception as e:
        print(f"  Milvus 未就绪（第 {i+1}/{max_retries} 次）: {e}")
        time.sleep(5)
else:
    print("  ERROR: Milvus 连接超时，退出")
    sys.exit(1)
EOF

# 2. 检查模型文件
echo "[entrypoint] 检查模型文件..."
if [ ! -f "/models/bge-m3/config.json" ]; then
    echo "  警告: /models/bge-m3 模型未挂载，将使用 HuggingFace 在线下载"
fi
if [ ! -f "/models/bge-reranker-large/config.json" ]; then
    echo "  警告: /models/bge-reranker-large 模型未挂载，将使用 HuggingFace 在线下载"
fi

# 3. 创建日志目录
mkdir -p /app/logs

# 4. 打印配置信息
echo "[entrypoint] 服务配置:"
echo "  MILVUS_URI: $MILVUS_URI"
echo "  MILVUS_COLLECTION: $MILVUS_COLLECTION"
echo "  EMBED_MODEL_PATH: $EMBED_MODEL_PATH"
echo "  RERANK_MODEL: $RERANK_MODEL"
echo "  API_PORT: $API_PORT"

# 5. 启动服务
echo "[entrypoint] 启动 FastAPI 服务..."
exec "$@"
