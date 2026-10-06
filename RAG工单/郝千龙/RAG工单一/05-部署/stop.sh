#!/usr/bin/env bash
# ======================================================================
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 脚本：stop.sh
# 用途：停止 RAG 问答系统的全部服务
#   1) 读取 PID 文件，杀掉 Streamlit、FastAPI 进程
#   2) 停止 Milvus + Redis 容器（保留数据卷）
#   3) 清理 PID 文件
# 用法：
#   chmod +x stop.sh
#   ./stop.sh
# 编写日期：2026-09-28
# ======================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/05-部署"
PID_FILE="$DEPLOY_DIR/run.pid"

log()  { echo -e "\033[33m[STOP]\033[0m $*"; }

# ---------- 1. 杀掉应用进程 ----------
if [[ -f "$PID_FILE" ]]; then
    log "Step 1: 停止应用进程..."
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        if kill -0 "$pid" 2>/dev/null; then
            log "  杀掉 PID=$pid"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            # 仍存活则强杀
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        else
            log "  PID=$pid 已不存在"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
else
    log "未发现 PID 文件，尝试按端口杀进程..."
    # 兜底：按端口杀
    for port in 8000 8501; do
        pid=$(ss -ltnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[\d]+' | head -n1 || true)
        [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
    done
fi

# ---------- 2. 停止 Docker 容器 ----------
log "Step 2: 停止 Milvus + Redis 容器（保留数据卷）..."
if [[ -f "$DEPLOY_DIR/docker-compose.yml" ]]; then
    docker compose -f "$DEPLOY_DIR/docker-compose.yml" stop \
        || docker-compose -f "$DEPLOY_DIR/docker-compose.yml" stop \
        || log "  docker 容器停止失败（可能未启动）"
fi

# ---------- 3. 完成 ----------
log "停止完成。"
log "如需彻底删除数据卷，请手动执行:"
log "  docker compose -f $DEPLOY_DIR/docker-compose.yml down -v"

# ======================================================================
# 技术备注：
# 1. RAG：停止脚本优雅关闭 RAG 各组件，避免向量库数据损坏。
# 2. 容器 stop 不删卷，下次 start 可直接复用已入库数据。
# 3. 兜底按端口杀进程，避免 PID 文件丢失场景下漏杀。
# ======================================================================
