#!/usr/bin/env bash
# 安装脚本：在一台干净的 Ubuntu / CentOS 上把项目跑起来
#
#   ./install.sh                完整安装
#   ./install.sh --skip-docker     跳过容器启动（服务已在别处）
#   ./install.sh --skip-kb         跳过知识库构建（数据量大时可选）
#   ./install.sh --with-mineru     同时装 PDF 解析用的 MinerU（见下方说明）
#
# 步骤：0 环境检查 -> 1 Python 环境 -> 2 依赖服务 -> 3 初始化数据库 -> 4 构建知识库
#       -> 5 MinerU（可选）
#
# 关于 MinerU：它要下载约 7GB 依赖 + 1.4GB 模型，默认**不装**。
# 不装不影响项目运行——上传 PDF 时会自动退回三件套兜底（PyMuPDF 文本层 +
# PaddleOCR 图表文字 + pdfplumber 表格），扫描件与图表照样能读出内容，
# 只是慢：28 页文档实测兜底 5~6 分钟、MinerU 主路径 1~1.5 分钟（随机器负载波动，
# 兜底路径慢约 4 倍）。需要完整解析能力时加 --with-mineru。
set -euo pipefail

# 脚本位于 deploy/ 下，项目根目录是它的上一级
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SKIP_DOCKER=0
SKIP_KB=0
WITH_MINERU=0
for arg in "$@"; do
  case "$arg" in
    --skip-docker) SKIP_DOCKER=1 ;;
    --skip-kb)     SKIP_KB=1 ;;
    --with-mineru) WITH_MINERU=1 ;;
  esac
done

info()  { printf '\033[32m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[31m[ERROR]\033[0m %s\n' "$*" >&2; }
step()  { printf '\n\033[36m=== %s ===\033[0m\n' "$*"; }

# ============================================================
# 0. 环境检查
# ============================================================
step "0/5 检查服务器配置"

if [ -f /etc/os-release ]; then
  . /etc/os-release
  info "操作系统: $PRETTY_NAME"
  case "$ID" in
    ubuntu|debian)  PKG="apt" ;;
    centos|rhel|rocky|almalinux) PKG="yum" ;;
    *) warn "未识别的发行版 $ID，依赖安装可能需手动处理"; PKG="apt" ;;
  esac
else
  warn "无法识别操作系统"
  PKG="apt"
fi
info "包管理器: $PKG"

# CPU / 内存 / 磁盘
CPU=$(nproc 2>/dev/null || echo 0)
MEM_GB=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
DISK_GB=$(df -BG "$PROJECT_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')
info "CPU: ${CPU} 核 | 内存: ${MEM_GB} GB | 可用磁盘: ${DISK_GB} GB"

[ "$CPU" -lt 2 ] && warn "CPU 少于 2 核，向量化会较慢"
awk -v m="$MEM_GB" 'BEGIN{exit !(m < 8)}' && \
  warn "内存不足 8GB：BGE 模型 + Milvus 同时运行可能触发 OOM，建议先停掉评测类任务"
[ "${DISK_GB:-0}" -lt 20 ] && warn "可用磁盘不足 20GB，模型与向量数据可能放不下"

# GPU（可选，有则向量化快很多）
if command -v nvidia-smi >/dev/null 2>&1; then
  info "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  warn "未检测到 NVIDIA GPU，BGE-M3 将跑在 CPU 上（首次检索会明显变慢）"
fi

# ============================================================
# 1. Python 环境
# ============================================================
step "1/5 准备 Python 环境"

# 优先使用 conda（若已安装），否则用 venv
if command -v conda >/dev/null 2>&1 && [ ! -d "$PROJECT_DIR/.venv" ]; then
  info "检测到 conda，创建环境 rag-roleplay (python 3.12)"
  conda create -y -n rag-roleplay python=3.12 >/dev/null
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate rag-roleplay
  PYTHON="$(command -v python)"
else
  if [ ! -d "$PROJECT_DIR/.venv" ]; then
    info "创建 venv"
    python3 -m venv "$PROJECT_DIR/.venv"
  fi
  PYTHON="$PROJECT_DIR/.venv/bin/python"
fi
info "Python: $("$PYTHON" --version 2>&1)"

info "安装依赖（requirements.txt）"
"$PYTHON" -m pip install -q --upgrade pip
"$PYTHON" -m pip install -q -r "$PROJECT_DIR/requirements.txt"
info "依赖安装完成"

# ============================================================
# 2. 依赖服务
# ============================================================
step "2/5 启动依赖服务（Redis / MySQL / Milvus）"

if [ "$SKIP_DOCKER" -eq 1 ]; then
  warn "按要求跳过容器启动"
elif ! command -v docker >/dev/null 2>&1; then
  error "未安装 docker，请先安装：https://docs.docker.com/engine/install/"
  exit 1
else
  # 读取 .env 里的密码，保证与 compose 一致
  [ -f "$PROJECT_DIR/.env" ] && set -a && . "$PROJECT_DIR/.env" && set +a
  # 刻意不留默认值：弱口令默认值比直接报错危险。缺配置就退出并说明原因。
  export REDIS_PASSWORD="${REDIS_PASSWORD:?未在 .env 中设置 REDIS_PASSWORD}"
  export MYSQL_PASSWORD="${MYSQL_PASSWORD:?未在 .env 中设置 MYSQL_PASSWORD}"

  if docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
  else
    error "未找到 docker compose"
    exit 1
  fi

  info "启动容器（首次会拉取镜像，耗时较长）"
  (cd "$PROJECT_DIR/deploy" && $COMPOSE up -d)

  info "等待 Milvus 就绪..."
  for i in $(seq 1 60); do
    if "$PYTHON" - <<'PY' >/dev/null 2>&1
import sys
from pymilvus import MilvusClient
try:
    MilvusClient(uri="http://localhost:19530").list_collections()
except Exception:
    sys.exit(1)
PY
    then
      info "Milvus 已就绪"; break
    fi
    [ "$i" -eq 60 ] && { error "Milvus 启动超时"; exit 1; }
    sleep 5
  done
fi

# ============================================================
# 3. 初始化数据库
# ============================================================
step "3/5 初始化 MySQL（建库建表 + 写入角色）"

if [ ! -f "$PROJECT_DIR/.env" ]; then
  error "缺少 .env，请先从 .env.example 复制并填写配置"
  exit 1
fi
"$PYTHON" -m scripts.init_db

# ============================================================
# 4. 构建知识库
# ============================================================
step "4/5 构建知识库"

if [ "$SKIP_KB" -eq 1 ]; then
  warn "按要求跳过知识库构建，稍后可手动执行："
  echo "    $PYTHON -m scripts.knowledge_base"
  echo "    $PYTHON -m scripts.dedup_kb --apply"
  echo "    $PYTHON -m scripts.build_law_index"
else
  "$PYTHON" -m scripts.knowledge_base
  # 源数据里有若干条记录归一化后文本完全相同（只差标点/空白），
  # 入库后会变成重复切片。去重后才是 13364 条，不做这一步会多出 68 条。
  "$PYTHON" -m scripts.dedup_kb --apply
  "$PYTHON" -m scripts.build_law_index
fi

# ============================================================
# 5. MinerU（可选，PDF 解析用）
# ============================================================
step "5/5 PDF 解析（MinerU，可选）"

if [ "$WITH_MINERU" -eq 0 ]; then
  warn "未指定 --with-mineru，跳过。项目照常运行，PDF 入库退回三件套兜底（较慢）。"
  echo "  需要时手动执行："
  echo "    python3 -m venv mineru/.venv"
  echo "    mineru/.venv/bin/pip install mineru==4.0.5"
  echo "    MINERU_MODEL_SOURCE=modelscope mineru/.venv/bin/mineru-models-download \\"
  echo "        --tier basic --small-backend torch -s modelscope"
else
  # 独立 venv：MinerU 要求 openai<3，与主环境的 openai 3.x 冲突
  if [ ! -x "mineru/.venv/bin/pip" ]; then
    info "创建 MinerU 专用 venv"
    python3 -m venv mineru/.venv
  fi
  info "安装 MinerU（约 7GB 依赖，视网速可能较慢）"
  mineru/.venv/bin/pip install --no-input "mineru==4.0.5"

  # 后端必须显式选 torch：mineru 默认拉的 torch 是 cu130 构建，
  # torch.cuda.is_available() 为 False，MinerU 会静默退回 onnx/CPU 而不报错。
  # 若本机有 NVIDIA GPU，装完还需按 README「PDF 解析（MinerU）」一节换成 cu126 的
  # torch，再确认 `mineru-kit models show` 的 Effective small backend 是 torch。
  info "下载模型（basic 档，模型目录约 1.4GB）"
  # 模型源必须显式指定：本机 huggingface.co 直连不通
  MINERU_MODEL_SOURCE=modelscope mineru/.venv/bin/mineru-models-download \
      --tier basic --small-backend torch -s modelscope

  info "验证解析链路"
  "$PYTHON" -c "
from app.core import mineru_service
print('  MinerU 可用:', mineru_service.available())
print('  可执行文件:', mineru_service.binary())
"
fi

# ============================================================
printf '\n\033[32m%s\033[0m\n' "安装完成"
cat <<EOF

启动服务：  ./run.sh -d
接口文档：  http://127.0.0.1:8000/docs
停止服务：  ./shutdown.sh          （只停 API）
            ./shutdown.sh --all    （连依赖服务一起停）

演示账号：  demo / demo123
EOF
