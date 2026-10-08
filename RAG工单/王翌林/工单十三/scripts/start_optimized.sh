#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================================
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# scripts/start_optimized.sh —— 启动工单二优化版服务
#
#   FastAPI    0.0.0.0:8000  （uvicorn src.api:app，含 lifespan 自检 MySQL/Milvus）
#   Streamlit  0.0.0.0:8502  （app/streamlit_app_optimized.py，中英双语/对比/反馈）
#
# 可选环境变量：
#   APP_PORT=8000              FastAPI 端口（默认 8000，按工单要求）
#   STREAMLIT_PORT=8502        Streamlit 端口（默认 8502，避免与基线 8501 冲突）
#   PYTHON_BIN=/path/python    指定解释器
#   VENV_DIR=/path/venv        自备 venv（默认 /home/dabaie/code/my_project/.venv）
#
# 中文路径：SCRIPT_DIR/PROJECT_ROOT 经 BASH_SOURCE 解析并全程加引号，兼容中文目录。
# 幂等：重复执行会先清掉占用 8000/8502 的旧进程，再后台拉起。
# ============================================================================
set -euo pipefail
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

WORKORDER="人工智能NLP-RAG-基于PDF文档的问答系统优化"
ENV_NAME="rag_pdf_qa_opt"
VENV_DIR="${VENV_DIR:-/home/dabaie/code/my_project/.venv}"
MINICONDA_HOME="$HOME/miniconda3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 端口按工单要求固定：FastAPI=8000，Streamlit=8502。
# 注意：.env 内的 STREAMLIT_PORT=8501 是基线前端的配置，不能覆盖优化版的 8502；
#       因此先快照「用户在命令行显式导出」的覆盖值，source .env 后再据此定端口。
_USER_APP_PORT="${APP_PORT:-}"
_USER_ST_PORT="${STREAMLIT_PORT:-}"
API_PORT="${_USER_APP_PORT:-8000}"
STREAMLIT_PORT="${_USER_ST_PORT:-8502}"
LOG_DIR="$PROJECT_ROOT/data/logs"
PID_DIR="$PROJECT_ROOT/data/pids"
API_PID_FILE="$PID_DIR/api_optimized.pid"
ST_PID_FILE="$PID_DIR/streamlit_optimized.pid"
mkdir -p "$LOG_DIR" "$PID_DIR"

echo "============================================================"
echo "🚀 启动优化版服务"
echo "   工单编号：$WORKORDER"
echo "   FastAPI=$API_PORT | Streamlit=$STREAMLIT_PORT"
echo "============================================================"

# ---------- 0. 加载 .env ----------
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi
# .env 中的 STREAMLIT_PORT=8501 仅服务于基线前端；优化版端口以工单要求为准
# （仅尊重用户在执行脚本前显式 export 的覆盖值）
API_PORT="${_USER_APP_PORT:-8000}"
STREAMLIT_PORT="${_USER_ST_PORT:-8502}"

# ---------- 1. 解析 Python 解释器（conda → venv → 系统） ----------
echo ""
echo "[1/4] 解析 Python 运行环境 ..."
if [ -z "${PYTHON_BIN:-}" ]; then
    if ! command -v conda &>/dev/null && [ -f "$MINICONDA_HOME/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$MINICONDA_HOME/etc/profile.d/conda.sh"
    fi
    if command -v conda &>/dev/null && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        eval "$(conda shell.bash hook)"
        conda activate "$ENV_NAME"
        PYTHON_BIN="$(which python)"
        echo "  ↳ 使用 conda 环境 $ENV_NAME：$PYTHON_BIN"
    elif [ -x "$VENV_DIR/bin/python" ]; then
        PYTHON_BIN="$VENV_DIR/bin/python"
        echo "  ↳ 使用已有 venv：$PYTHON_BIN"
    else
        PYTHON_BIN="$(command -v python3)"
        echo "  ↳ 使用系统 python3：$PYTHON_BIN（建议先 bash scripts/install_optimized.sh）"
    fi
fi
"$PYTHON_BIN" -V

# ---------- 2. 依赖服务自检（仅提示，不阻断：应用内部有 Lite/降级兜底） ----------
echo ""
echo "[2/4] 依赖服务自检 ..."
if command -v curl &>/dev/null; then
    if curl -s -m 3 "http://${MILVUS_HOST:-localhost}:${MILVUS_PORT:-19530}" -o /dev/null 2>&1; then
        echo "  ✅ Milvus ${MILVUS_HOST:-localhost}:${MILVUS_PORT:-19530} 可达"
    else
        echo "  ⚠️  远程 Milvus 不可达，应用将自动降级 Milvus Lite（data/milvus_local.db）"
    fi
else
    echo "  ⚠️  无 curl，跳过依赖服务探测"
fi
# MySQL 用 Python 探活（本机未必装 mysql client）
"$PYTHON_BIN" - <<'PY' || echo "  ⚠️  MySQL 暂不可达，API 仍会启动，health 中将显示 mysql=failed"
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化 —— 启动前 MySQL 探活
import os
from src.db import test_connection
ver = test_connection()
print(f"  ✅ MySQL 可达，版本 {ver}" if ver else "  ⚠️  MySQL 探活失败")
raise SystemExit(0 if ver else 1)
PY

# ---------- 3. 清理被占用的目标端口（幂等启动） ----------
echo ""
echo "[3/4] 清理目标端口占用 ..."
free_port() {
    local port="$1"
    if ss -tln 2>/dev/null | grep -q ":$port "; then
        local pids
        pids=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | sort -u || true)
        for pid in $pids; do
            echo "  ↳ 端口 $port 被 PID=$pid 占用，先停止"
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 2
        pids=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | sort -u || true)
        for pid in $pids; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
    echo "  ✅ 端口 $port 空闲"
}
free_port "$API_PORT"
free_port "$STREAMLIT_PORT"

# ---------- 4. 启动服务 ----------
echo ""
echo "[4/4] 后台启动服务 ..."

# FastAPI（lifespan 会加载 bge-m3/reranker，首次启动可能需数十秒）
nohup "$PYTHON_BIN" -m uvicorn src.api:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    > "$LOG_DIR/api_optimized.log" 2>&1 &
API_PID=$!
echo "$API_PID" > "$API_PID_FILE"
echo "  ✅ FastAPI 已拉起：PID=$API_PID → http://127.0.0.1:$API_PORT（日志 $LOG_DIR/api_optimized.log）"

# Streamlit：注入 API_BASE，使前端默认指向本机 8000 的优化版 API
export APP_PORT="$API_PORT"
export API_BASE="http://127.0.0.1:$API_PORT"
nohup "$PYTHON_BIN" -m streamlit run app/streamlit_app_optimized.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    > "$LOG_DIR/streamlit_optimized.log" 2>&1 &
ST_PID=$!
echo "$ST_PID" > "$ST_PID_FILE"
echo "  ✅ Streamlit 已拉起：PID=$ST_PID → http://127.0.0.1:$STREAMLIT_PORT（日志 $LOG_DIR/streamlit_optimized.log）"

# ---------- 健康检查轮询（API 需加载模型，最多等 120 秒） ----------
echo ""
echo "⏳ 等待 FastAPI 就绪 ..."
HEALTH_URL="http://127.0.0.1:$API_PORT/api/health"
HEALTH_BODY=""
for i in $(seq 1 60); do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "❌ FastAPI 进程已退出，请查看 $LOG_DIR/api_optimized.log"
        tail -n 20 "$LOG_DIR/api_optimized.log" || true
        exit 1
    fi
    if HEALTH_BODY=$(curl -s -m 3 "$HEALTH_URL" 2>/dev/null) && [ -n "$HEALTH_BODY" ]; then
        echo "  ✅ API 健康检查通过：$HEALTH_BODY"
        break
    fi
    sleep 2
    if [ "$i" -eq 60 ]; then
        echo "❌ 等待 API 就绪超时（120s），请查看 $LOG_DIR/api_optimized.log"
        tail -n 20 "$LOG_DIR/api_optimized.log" || true
        exit 1
    fi
done

# Streamlit 端口就绪确认
for i in $(seq 1 15); do
    if ss -tln 2>/dev/null | grep -q ":$STREAMLIT_PORT "; then
        echo "  ✅ Streamlit 端口 $STREAMLIT_PORT 已监听"
        break
    fi
    sleep 1
done

echo ""
echo "============================================================"
echo "🎉 优化版服务已启动（工单编号：$WORKORDER）"
echo "   FastAPI    ：http://127.0.0.1:$API_PORT  （/docs 为 Swagger）"
echo "   Streamlit  ：http://127.0.0.1:$STREAMLIT_PORT"
echo "   PID 文件   ：$API_PID_FILE , $ST_PID_FILE"
echo "   日志文件   ：$LOG_DIR/{api_optimized,streamlit_optimized}.log"
echo "   停止服务   ：bash scripts/stop_optimized.sh"
echo "============================================================"
