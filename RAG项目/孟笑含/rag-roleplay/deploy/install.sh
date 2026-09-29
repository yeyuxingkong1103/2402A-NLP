#!/usr/bin/env bash
# ============================================================
# RAG 角色扮演系统 · 安装脚本（Ubuntu / WSL Ubuntu）
# 用法：bash deploy/install.sh [--with-milvus] [--with-db] [--with-rag] [--skip-models]
# 幂等：可重复执行
# 说明：默认只装核心依赖（requirements-core.txt，约 200MB）；
#       加 --with-rag 才装 RAG 模型推理依赖（torch 等约 3GB，Linux 默认先装 CPU 版 torch）
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."          # 项目根目录
PROJECT_DIR=$(pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PIP_MIRROR=${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}

log() { echo -e "\033[1;32m[install]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*"; }
die() { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }

WITH_MILVUS=false; WITH_DB=false; SKIP_MODELS=false; WITH_RAG=false
for arg in "$@"; do
  case $arg in
    --with-milvus) WITH_MILVUS=true ;;
    --with-db)     WITH_DB=true ;;
    --skip-models) SKIP_MODELS=true ;;
    --with-rag)    WITH_RAG=true ;;
  esac
done

# ---------- 0. 检查服务器配置 ----------
log "0/6 检查服务器配置"
if [ -f /etc/os-release ]; then
  . /etc/os-release
  case "$ID" in
    ubuntu|debian) log "系统：$PRETTY_NAME ✓" ;;
    *) warn "非 Ubuntu/Debian（$PRETTY_NAME），安装命令可能不适用" ;;
  esac
else
  die "无法识别操作系统"
fi
MEM_MB=$(free -m | awk '/^Mem:/{print $2}')
[ "$MEM_MB" -ge 4096 ] || warn "内存 ${MEM_MB}MB < 4GB，运行 Milvus/本地模型可能不足"
DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | tr -d 'G')
[ "${DISK_GB:-0}" -ge 20 ] || warn "磁盘剩余 ${DISK_GB}GB < 20GB，模型+依赖约需 10GB"

# ---------- 1. 系统依赖 ----------
log "1/6 检查系统依赖（python3-venv/pip）"
if ! command -v "$PYTHON_BIN" >/dev/null || ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-venv python3-pip >/dev/null
  else
    die "缺少 python3-venv 且 sudo 需要密码。请手动执行：sudo apt-get install -y python3-venv python3-pip"
  fi
else
  log "python3/venv 已就绪，跳过 apt"
fi

if $WITH_DB; then
  log "1b 安装 MySQL + Redis（本机服务）"
  sudo -n true 2>/dev/null || die "--with-db 需要免密 sudo，或手动安装 MySQL/Redis 后跳过本参数"
  sudo apt-get install -y -qq mysql-server redis-server >/dev/null
  sudo systemctl enable --now mysql redis-server 2>/dev/null || warn "WSL 无 systemd，请手动启动：service mysql start && service redis-server start"
fi

# ---------- 2. Python 虚拟环境 ----------
log "2/6 创建 Python 虚拟环境"
if [ ! -d venv ]; then
  $PYTHON_BIN -m venv venv
fi
VENV_PY="$PROJECT_DIR/venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip -q -i "$PIP_MIRROR"

# ---------- 3. Python 依赖 ----------
log "3/6 安装核心依赖（requirements-core.txt）"
"$VENV_PY" -m pip install -r requirements-core.txt -q -i "$PIP_MIRROR" || warn "部分依赖安装失败，请检查网络后重试"

if $WITH_RAG; then
  log "3b 安装 RAG 依赖（CPU 版 torch + FlagEmbedding 等，约 3GB）"
  # Linux 上先装 CPU torch，避免 pip 默认拉 CUDA 版（含 nvidia 全家桶）
  "$VENV_PY" -m pip install torch -q --index-url https://download.pytorch.org/whl/cpu \
    || "$VENV_PY" -m pip install torch -q -i "$PIP_MIRROR"   # 回退普通源
  "$VENV_PY" -m pip install -r requirements-rag.txt -q -i "$PIP_MIRROR" || warn "RAG 依赖安装失败"
else
  log "3b 跳过 RAG 依赖（未传 --with-rag；对话功能不受影响，上传/检索接口需装）"
fi

# ---------- 4. 配置 .env ----------
log "4/6 生成 .env 配置"
if [ ! -f .env ]; then
  cp .env.example .env
  warn "已生成 .env，请编辑填入 LLM_API_KEY（MySQL/Redis 密码）后重新执行或直接 run.sh"
else
  log ".env 已存在，跳过"
fi

# ---------- 5. Milvus（可选） ----------
if $WITH_MILVUS; then
  log "5/6 启动 Milvus（Docker）"
  command -v docker >/dev/null || die "未安装 Docker，请先安装 docker 后重试"
  docker compose up -d || die "Milvus 启动失败"
else
  log "5/6 跳过 Milvus（未传 --with-milvus；使用已有 Milvus 请配置 MILVUS_HOST）"
fi

# ---------- 6. 模型下载（可选） ----------
if $SKIP_MODELS; then
  log "6/6 跳过模型下载（--skip-models）"
else
  log "6/6 下载 BGE 模型（约 4.5GB，走 hf-mirror）"
  "$VENV_PY" scripts/download_models.py || warn "模型下载失败，可稍后重跑本脚本"
fi

log "安装完成。下一步：编辑 .env → bash deploy/run.sh"
