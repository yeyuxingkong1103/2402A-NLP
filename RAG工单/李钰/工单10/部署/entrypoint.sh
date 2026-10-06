#!/bin/bash
# ==========================================================================
# 入口脚本 - 初始化 + 启动金融问答系统
# 工单编号: 人工智能 NLP-RAG-金融问答系统部署
# ==========================================================================

set -e

echo "============================================================"
echo "  Financial RAG QA System v10.0"
echo "  工单编号: 人工智能 NLP-RAG-金融问答系统部署"
echo "============================================================"

# 创建持久化目录 (防止卷挂载后被空目录覆盖)
mkdir -p /app/data /app/cache /app/logs /app/shared

# 如果 data 目录为空, 从应用目录复制预设数据
if [ -d /app/data ] && [ -z "$(ls -A /app/data 2>/dev/null)" ]; then
    echo "[初始化] 从应用目录复制预设数据..."
    if [ -f /app/financial_kg.json ]; then
        cp /app/financial_kg.json /app/data/ 2>/dev/null || true
    fi
    echo "[初始化] 数据复制完成"
fi

# 显示配置
echo "[配置] Port: ${FLASK_PORT:-5008}"
echo "[配置] LLM API: ${LLM_API_KEY:+已配置 (以 **** 开头)}${LLM_API_KEY:-未配置 (降级模式)}"
echo "[配置] Neo4j: ${NEO4J_URI:-未配置}"
echo "[配置] Data dir: ${RAG_DATA_DIR:-/app/data}"
echo "[配置] Cache dir: ${RAG_CACHE_DIR:-/app/cache}"

# 启动应用
echo "[启动] 金融问答系统..."
exec "$@"
