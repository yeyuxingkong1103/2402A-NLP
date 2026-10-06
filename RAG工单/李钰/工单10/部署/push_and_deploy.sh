#!/bin/bash
# ============================================================
# 生产部署脚本 - 构建 + 推送 + 在远端启动
# 工单编号: 人工智能 NLP-RAG-金融问答系统部署
#
# 用法:
#   1. 修改 REGISTRY 和 REMOTE_HOST
#   2. ./push_and_deploy.sh
# ============================================================

set -e

# ============ 配置 ============
IMAGE_NAME="financial-rag"
IMAGE_TAG="v10.0"
REGISTRY="${REGISTRY:-registry.example.com}"   # 私有镜像仓库
REMOTE_HOST="${REMOTE_HOST:-user@server}"       # 远端服务器
REMOTE_PORT="${REMOTE_PORT:-22}"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================================"
echo "  生产部署"
echo "  镜像: ${FULL_IMAGE}"
echo "  目标: ${REMOTE_HOST}"
echo "============================================================"

# 1. 本地构建
echo -e "\n[1/4] 本地构建镜像..."
docker build -f 工单10/部署/Dockerfile -t "${IMAGE_NAME}:${IMAGE_TAG}" .

# 2. 打标签
echo -e "\n[2/4] 打标签..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"

# 3. 推送
echo -e "\n[3/4] 推送到 ${REGISTRY}..."
docker push "${FULL_IMAGE}"

# 4. 远端部署
echo -e "\n[4/4] 远端部署..."
ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" << EOF
docker pull "${FULL_IMAGE}"
docker rm -f financial-rag-qa 2>/dev/null || true
docker run -d --name financial-rag-qa \
    -p 5008:5008 \
    -v rag_data:/app/data \
    -v rag_cache:/app/cache \
    -v rag_logs:/app/logs \
    -v rag_shared:/app/shared \
    -e LLM_API_KEY="\${LLM_API_KEY}" \
    --restart unless-stopped \
    "${FULL_IMAGE}"
docker ps --filter "name=financial-rag-qa"
EOF

echo -e "\n============================================================"
echo "  ✅ 部署完成!"
echo "============================================================"
