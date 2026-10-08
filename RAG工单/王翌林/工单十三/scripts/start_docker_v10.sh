#!/usr/bin/env bash
# 工单编号：人工智能NLP-RAG-金融问答系统部署
# scripts/start_docker_v10.sh —— 工单十 构建镜像并启动容器（rag-api:8006 / rag-ui:8506）
set -e
cd "$(dirname "$0")/.."

echo "[v10] 构建镜像 rag-v10:latest（首次约 10-20 分钟，后续走缓存）..."
docker compose build

echo "[v10] 启动容器..."
docker compose up -d

# 工单十：健康检查重试（首次需加载 bge-m3/reranker/全文索引，最多 10 分钟）
echo "[v10] 等待 rag-api 就绪（模型加载中）..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8006/api/v6/health > /dev/null 2>&1; then
        echo "[v10] rag-api 就绪 ($((i*5))s)"
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -q rag-v10-api; then
        echo "[v10] rag-api 容器已退出，最近日志："
        docker logs --tail 50 rag-v10-api || true
        exit 1
    fi
    sleep 5
done

echo "[v10] 等待 rag-ui 就绪..."
for i in $(seq 1 24); do
    if curl -sf http://localhost:8506 > /dev/null 2>&1; then
        echo "[v10] rag-ui 就绪"
        break
    fi
    sleep 5
done

echo "[v10] 启动完成"
echo "  FastAPI v6:  http://localhost:8006/api/v6/health"
echo "  Streamlit:   http://localhost:8506"
docker compose ps
