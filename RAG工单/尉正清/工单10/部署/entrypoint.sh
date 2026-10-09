#!/bin/bash
# 工单编号：人工智能NLP-RAG-金融问答系统部署
#
# 容器入口：先把运行前提检查清楚，再起服务。
#
# 为什么要有这个脚本，而不是直接 `CMD python app.py`：
#   1. 模型和知识库都在容器外（镜像里不带），挂载配错时必须**当场报错退出**，
#      而不是等 Python 抛一个看不懂的路径异常。容器日志里能直接看到缺什么。
#   2. 知识库放数据卷，首次启动要能从镜像自带的种子里初始化，
#      这样 `docker run` 不带任何挂载也能跑起来（验收标准第一条）。
set -euo pipefail

MODEL_DIR="${BGE_M3_PATH:-/models/bge-m3}"
KB_ROOT="${KB_CACHE_DIR:-/kb}"
KB_NAME="${KB_NAME:-ccf_competition}"
PORT="${PORT:-7860}"
SEED_KB="${SEED_KB:-/app/seed_kb}"

log() { echo "[入口] $*"; }
die() { echo "[入口][错误] $*" >&2; exit 1; }

log "金融问答系统启动中 —— 工单编号：人工智能NLP-RAG-金融问答系统部署"
log "模型目录 : ${MODEL_DIR}"
log "知识库根 : ${KB_ROOT}"
log "监听端口 : ${PORT}"

# ---------- 1. 向量模型 ----------
# FlagEmbedding 加载时认权重文件，缺了会在模型初始化阶段才炸，提前查更清楚
if [ ! -d "${MODEL_DIR}" ]; then
    die "模型目录不存在：${MODEL_DIR}
    请挂载：-v <宿主机模型路径>:${MODEL_DIR}:ro
    需要的是 BGE-M3，目录里应有 pytorch_model.bin 或 model.safetensors。"
fi
if [ ! -f "${MODEL_DIR}/pytorch_model.bin" ] && [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    die "模型目录里没有权重文件（pytorch_model.bin / model.safetensors）：${MODEL_DIR}
    目录存在但内容是空的或不完整，通常是挂载路径写错了。"
fi
log "模型自检通过"

# ---------- 2. 知识库 ----------
# 数据卷是空的就从镜像种子初始化。判据用 chunks.json：
# 只有目录没有 chunks.json 说明上次初始化中断了，同样按空的处理。
if [ ! -f "${KB_ROOT}/${KB_NAME}/chunks.json" ]; then
    if [ -f "${SEED_KB}/${KB_NAME}/chunks.json" ]; then
        log "数据卷里没有知识库，从镜像自带种子初始化 ..."
        mkdir -p "${KB_ROOT}/${KB_NAME}"
        cp -a "${SEED_KB}/${KB_NAME}/." "${KB_ROOT}/${KB_NAME}/"
        log "知识库已写入 ${KB_ROOT}/${KB_NAME}"
    else
        die "没有可用知识库：${KB_ROOT}/${KB_NAME}
    数据卷是空的，镜像里也没有种子。
    请先用 部署/docker-compose.yml 里的 kb-init 服务建库，或挂载一份已有知识库。"
    fi
fi

# 卷可能被挂成只读，或宿主目录权限不对 —— 反馈日志要写这个目录，提前探一下
if ! touch "${KB_ROOT}/.write_test" 2>/dev/null; then
    log "警告：${KB_ROOT} 不可写，界面上的 👍/👎 反馈将无法落盘（检索与问答不受影响）"
else
    rm -f "${KB_ROOT}/.write_test"
fi
log "知识库自检通过（${KB_NAME}）"

# ---------- 3. 启动服务 ----------
log "启动 Gradio 服务 ..."
cd /app
exec python -u app.py
