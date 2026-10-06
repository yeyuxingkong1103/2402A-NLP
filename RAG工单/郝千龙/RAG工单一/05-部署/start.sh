#!/usr/bin/env bash
# ======================================================================
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 脚本：start.sh
# 用途：启动 RAG 问答系统的全部服务
#   1) 启动 Milvus + Redis (docker-compose)
#   2) 激活 conda 环境 rag-py310
#   3) 后台启动 FastAPI（端口 8000）
#   4) 后台启动 Streamlit（端口 8501）
#   5) 写入 PID 文件，便于 stop.sh 停止
# 用法：
#   chmod +x start.sh
#   ./start.sh
# 编写日期：2026-09-28
# ======================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$PROJECT_ROOT/02-研发"
DEPLOY_DIR="$PROJECT_ROOT/05-部署"
ENV_NAME="rag-py310"
PID_FILE="$DEPLOY_DIR/run.pid"
LOG_DIR="$DEPLOY_DIR/logs"
mkdir -p "$LOG_DIR"

log()  { echo -e "\033[32m[START]\033[0m $*"; }
error() { echo -e "\033[31m[ERROR]\033[0m $*"; exit 1; }

# ---------- 1. Docker 容器 ----------
log "Step 1: 启动 Milvus + Redis ..."
docker compose -f "$DEPLOY_DIR/docker-compose.yml" up -d \
    || docker-compose -f "$DEPLOY_DIR/docker-compose.yml" up -d

# 等待 Milvus 健康
log "等待 Milvus 健康检查..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:9091/healthz | grep -q OK; then
        log "Milvus 健康"
        break
    fi
    sleep 2
    [[ $i -eq 30 ]] && error "Milvus 启动超时"
done

# ---------- 2. 激活 conda ----------
log "Step 2: 激活 conda 环境 $ENV_NAME"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---------- 3. 启动 FastAPI ----------
log "Step 3: 启动 FastAPI (端口 8000)..."
cd "$CODE_DIR"
nohup python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    > "$LOG_DIR/api.log" 2>&1 &
API_PID=$!
echo "$API_PID" >> "$PID_FILE"
log "FastAPI PID=$API_PID"

# 等 API 健康检查
log "等待 FastAPI 就绪..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/api/health | grep -q healthy; then
        log "FastAPI 已就绪"
        break
    fi
    sleep 1
    [[ $i -eq 30 ]] && warn "FastAPI 启动较慢，请检查 $LOG_DIR/api.log"
done

# ---------- 4. 启动 Streamlit ----------
log "Step 4: 启动 Streamlit (端口 8501)..."
nohup python -m streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    > "$LOG_DIR/streamlit.log" 2>&1 &
ST_PID=$!
echo "$ST_PID" >> "$PID_FILE"
log "Streamlit PID=$ST_PID"

# ---------- 5. 完成 ----------
sleep 2
log "启动完成！服务地址："
log "  - FastAPI :  http://<服务器IP>:8000/api/health"
log "  - Streamlit: http://<服务器IP>:8501"
log "  - PID 文件: $PID_FILE"
log "  - 日志目录: $LOG_DIR"

# ======================================================================
# 技术备注：
# 1. RAG：FastAPI 编排 RAG 全流程；Streamlit 提供可视化问答界面。
# 2. workers=2 默认匹配单 GPU 推理吞吐，可按 CPU/GPU 调整。
# 3. nohup + & 让服务在 SSH 断开后继续运行；生产环境建议用 systemd。
# ======================================================================
