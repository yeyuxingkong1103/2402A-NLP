#!/usr/bin/env bash
# =============================================================================
# 工单编号：人工智能NLP-RAG 项目-金融问答系统部署
# 文件：部署/docker/entrypoint.sh —— 容器入口脚本
#
# 职责（都在启动前做完，失败要"响亮地失败"而不是半死不活）：
#   1. 目录体检：/app/logs、/app/sessions、/app/data/index 必须存在且可写；
#      索引卷没挂进来时给出**可操作的中文提示**，而不是等 uvicorn 报 500。
#   2. 打印环境快照（写进日志卷，供"运行过程无异常日志"验收逐行核对）。
#   3. exec "$@" 接管 PID 1 信号（Ctrl+C / docker stop 能干净退出）。
# =============================================================================
set -euo pipefail

WORK_ORDER="人工智能NLP-RAG 项目-金融问答系统部署"
LOG_DIR="${DEPLOY_LOG_DIR:-/app/logs}"
SESSION_DIR="${DEPLOY_SESSION_DIR:-/app/sessions}"
INDEX_DIR="${RAG_DATA__INDEX_DIR:-/app/data/index}"
UPLOAD_DIR="${DEPLOY_UPLOAD_DIR:-/app/data/uploads}"

ts() { date '+%Y-%m-%d %H:%M:%S%z'; }
say() { echo "[entrypoint $(ts)] $*"; }

say "启动容器：${WORK_ORDER}"
say "python    = $(python -V 2>&1)"
say "user      = $(id -un) (uid=$(id -u) gid=$(id -g))"
say "TZ        = ${TZ:-未设置}  当前时间 = $(date '+%Y-%m-%d %H:%M:%S %Z')"
say "ollama    = ${RAG_LLM__OLLAMA_BASE_URL:-未设置}"
say "redis     = ${DEPLOY_REDIS_URL:-未设置} (enabled=${DEPLOY_REDIS_ENABLED:-1})"
say "api port  = ${DEPLOY_API_PORT:-8200}"

# —— ① 目录体检 ——
for d in "$LOG_DIR" "$SESSION_DIR" "$UPLOAD_DIR"; do
    mkdir -p "$d"
    if ! touch "$d/.entrypoint_probe" 2>/dev/null; then
        say "错误：目录不可写 → $d（检查卷权限；绑定挂载请确认宿主目录存在且容器用户 uid=10001 有写权限）"
        exit 78   # EX_CONFIG
    fi
    rm -f "$d/.entrypoint_probe"
    say "目录可写 OK → $d"
done

mkdir -p "$INDEX_DIR"
if [ -z "$(ls -A "$INDEX_DIR" 2>/dev/null)" ]; then
    say "警告：索引目录为空 → $INDEX_DIR"
    say "      宿主机请挂载：-v /path/to/工单7/研发/data/index:/app/data/index:ro"
    say "      服务仍会启动，/health 会如实返回 checks.engine.ok=false（不伪装就绪）"
else
    say "索引目录 OK → $INDEX_DIR（顶层条目：$(ls -A "$INDEX_DIR" | tr '\n' ' ')）"
fi

# —— ② 环境快照落日志卷（纯环境事实，不含密钥）——
{
    echo "===== 容器环境快照 $(ts) ====="
    echo "work_order=$WORK_ORDER"
    echo "hostname=$(hostname)"
    echo "python=$(python -V 2>&1)"
    echo "user=$(id -un):$(id -u):$(id -g)"
    echo "TZ=${TZ:-}" 
    echo "RAG_DATA__INDEX_DIR=$INDEX_DIR"
    echo "RAG_LOG__DIR=$LOG_DIR"
    echo "RAG_LLM__OLLAMA_BASE_URL=${RAG_LLM__OLLAMA_BASE_URL:-}"
    echo "RAG_LLM__OLLAMA_EMBED_MODEL=${RAG_LLM__OLLAMA_EMBED_MODEL:-}"
    echo "RAG_LLM__OLLAMA_GEN_MODEL=${RAG_LLM__OLLAMA_GEN_MODEL:-}"
    echo "DEPLOY_SESSION_DIR=$SESSION_DIR"
    echo "DEPLOY_SESSION_BACKEND=${DEPLOY_SESSION_BACKEND:-auto}"
    echo "DEPLOY_REDIS_URL=${DEPLOY_REDIS_URL:-}"
    echo "DEPLOY_API_PORT=${DEPLOY_API_PORT:-8200}"
    echo "DEPLOY_MAX_CONCURRENCY=${DEPLOY_MAX_CONCURRENCY:-}"
    echo "DEPLOY_BUILD_REF=${DEPLOY_BUILD_REF:-}"
    echo "=========================================="
} >> "$LOG_DIR/容器启动快照.log" 2>/dev/null || say "警告：环境快照写入失败（不影响启动）"

# —— ③ 交棒 ——
say "执行：$*"
exec "$@"
