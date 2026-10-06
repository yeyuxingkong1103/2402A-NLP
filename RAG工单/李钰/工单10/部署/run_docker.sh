#!/bin/bash
# ============================================================
# 一键 Docker 启动 - Linux/Mac
# 工单编号: 人工智能 NLP-RAG-金融问答系统部署
# ============================================================

set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  Financial RAG QA System - Docker 部署"
echo "  工单编号: 人工智能 NLP-RAG-金融问答系统部署"
echo "============================================================"

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "[错误] 未检测到 Docker"
    exit 1
fi

echo -e "\n[1/3] 构建镜像..."
docker build -f 工单10/部署/Dockerfile -t financial-rag:v10.0 .

echo -e "\n[2/3] 启动容器..."
# 清理旧容器
docker rm -f financial-rag-qa 2>/dev/null || true

docker run -d --name financial-rag-qa \
    -p 5008:5008 \
    -v rag_data:/app/data \
    -v rag_cache:/app/cache \
    -v rag_logs:/app/logs \
    -v rag_shared:/app/shared \
    -e LLM_API_KEY="${LLM_API_KEY:-}" \
    --restart unless-stopped \
    financial-rag:v10.0

echo -e "\n[3/3] 等待健康检查..."
sleep 8
docker ps --filter "name=financial-rag-qa"

echo -e "\n============================================================"
echo "  ✅ 启动完成! 访问: http://localhost:5008"
echo "  健康检查: curl http://localhost:5008/api/health"
echo "============================================================"
echo -e "\n  常用命令:"
echo "    docker logs -f financial-rag-qa"
echo "    docker stop financial-rag-qa && docker start financial-rag-qa"
echo "    docker rm -f financial-rag-qa"
echo "============================================================"
