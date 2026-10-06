#!/usr/bin/env bash
# ======================================================================
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 脚本：install.sh
# 用途：在 Linux 服务器上一键安装 RAG 问答系统运行环境
#   - 检测/安装 Miniconda
#   - 创建 conda 虚拟环境 rag-py310 (Python 3.10)
#   - 检测 CUDA 并安装匹配的 PyTorch
#   - pip 安装业务依赖
#   - 启动 Milvus / Redis (通过 docker-compose)
#   - 解析招股说明书并入库
# 用法：
#   chmod +x install.sh
#   ./install.sh
# 编写日期：2026-09-28
# ======================================================================
set -euo pipefail

# ---------- 变量 ----------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$PROJECT_ROOT/02-研发"
ENV_NAME="rag-py310"
PYTHON_VERSION="3.10"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_DIR="$HOME/miniconda3"

log()    { echo -e "\033[32m[INSTALL]\033[0m $*"; }
warn()   { echo -e "\033[33m[WARN]\033[0m $*"; }
error()  { echo -e "\033[31m[ERROR]\033[0m $*"; exit 1; }

# ---------- 0. 系统依赖 ----------
log "Step 0: 安装系统依赖..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y build-essential curl wget git docker.io docker-compose-plugin
elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y gcc-c++ make curl wget git docker docker-compose
else
    warn "未识别的包管理器，跳过 apt/yum 安装，请确保已具备 docker / curl / gcc"
fi

# 确保当前用户可使用 docker
sudo usermod -aG docker "$USER" || true

# ---------- 1. Miniconda ----------
log "Step 1: 安装 Miniconda..."
if [[ -x "$MINICONDA_DIR/bin/conda" ]]; then
    log "Miniconda 已存在，跳过安装"
else
    wget -c "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f /tmp/miniconda.sh
fi
# 激活 conda
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# ---------- 2. 创建 conda 环境 ----------
log "Step 2: 创建 conda 环境 $ENV_NAME (Python $PYTHON_VERSION)..."
if conda env list | grep -qw "$ENV_NAME"; then
    log "环境已存在，跳过创建"
else
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

# ---------- 3. 检测 CUDA ----------
log "Step 3: 检测 CUDA..."
CUDA_VERSION=""
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    CUDA_RT=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d\.]+' || echo "")
    log "检测到 NVIDIA 驱动: $CUDA_VERSION; CUDA Runtime: ${CUDA_RT:-未安装 nvcc}"
else
    warn "未检测到 NVIDIA 驱动，将使用 CPU 版 PyTorch"
fi

# 安装 PyTorch：根据是否检测到 GPU 选择构建
log "Step 3.1: 安装 PyTorch..."
if [[ -n "$CUDA_VERSION" ]]; then
    # CUDA 12.1
    pip install --quiet --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    pip install --quiet --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# ---------- 4. 安装业务依赖 ----------
log "Step 4: 安装业务依赖 (requirements.txt)..."
pip install -r "$CODE_DIR/requirements.txt"

# ---------- 5. Milvus + Redis (docker-compose) ----------
log "Step 5: 启动 Milvus + Redis (docker-compose)..."
COMPOSE_FILE="$PROJECT_ROOT/05-部署/docker-compose.yml"
if [[ ! -f "$COMPOSE_FILE" ]]; then
    cat > "$COMPOSE_FILE" <<'YAML'
version: '3.8'
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data
  milvus:
    image: milvusdb/milvus:v2.4.10
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on: [etcd, minio]
  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
volumes:
  etcd_data:
  minio_data:
  redis_data:
YAML
    log "已生成 docker-compose.yml"
fi
docker compose -f "$COMPOSE_FILE" up -d || docker-compose -f "$COMPOSE_FILE" up -d
log "等待 Milvus 就绪..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:9091/healthz | grep -q OK; then
        log "Milvus 已就绪"
        break
    fi
    sleep 2
done

# ---------- 6. 入库招股说明书 ----------
log "Step 6: 入库招股说明书..."
cd "$CODE_DIR"
python rag_engine.py ingest

# ---------- 7. 完成 ----------
log "安装完成！"
log "  - conda 环境: $ENV_NAME"
log "  - 业务代码:   $CODE_DIR"
log "  - 启动服务:   $PROJECT_ROOT/05-部署/start.sh"
log "  - 停止服务:   $PROJECT_ROOT/05-部署/stop.sh"

# ======================================================================
# 技术备注：
# 1. RAG：本脚本部署 RAG 系统的全套依赖：PDF 解析、向量化、向量库、LLM。
# 2. Transformer：PyTorch 是 Transformer 框架基石，CUDA 版本与之匹配。
# 3. Fine-tuning：环境就绪后，后续如需 Embedding 微调，可直接在此环境训练。
# ======================================================================
