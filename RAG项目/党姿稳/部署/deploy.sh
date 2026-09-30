#!/usr/bin/env bash
# ============================================================
# deploy.sh — 一键部署脚本（Ubuntu / CentOS）
#
#   bash scripts/deploy.sh              # 完整部署
#   bash scripts/deploy.sh --skip-models  # 跳过模型下载（改用 hash 向量化）
#
# 步骤：环境检查 → Conda 环境 → Python 依赖 → Redis → MySQL
#       → Milvus(Docker) → 模型下载 → 配置检查 → 初始化 → 导入知识库
# 每一步都可重复执行，已完成的步骤会自动跳过。
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-rag-chat}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models}"
SKIP_MODELS=0
[[ "${1:-}" == "--skip-models" ]] && SKIP_MODELS=1

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

cd "$PROJECT_DIR"

# ------------------------------------------------------------ 0. 环境检查

step_check_env() {
  info "步骤 0/10：检查服务器环境"

  if [[ ! -f "$PROJECT_DIR/config.py" ]]; then
    fail "未在 $PROJECT_DIR 找到 config.py，请先把项目代码上传到该目录"
    exit 1
  fi

  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    OS_ID="${ID:-unknown}"
  else
    OS_ID="unknown"
  fi

  case "$OS_ID" in
    ubuntu|debian) PKG_MANAGER="apt";;
    centos|rhel|rocky|almalinux|fedora) PKG_MANAGER="yum";;
    *) PKG_MANAGER="unknown";;
  esac

  CPU_CORES="$(nproc 2>/dev/null || echo 1)"
  MEM_GB="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)"
  DISK_GB="$(df -BG "$PROJECT_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')"

  ok "系统：$OS_ID  包管理器：$PKG_MANAGER"
  ok "CPU：${CPU_CORES} 核   内存：${MEM_GB} GB   可用磁盘：${DISK_GB} GB"

  [[ "$MEM_GB" -lt 8 ]] && warn "内存小于 8GB，加载 BGE-m3 与重排模型可能吃紧"
  [[ "$DISK_GB" -lt 20 ]] && warn "可用磁盘小于 20GB，模型与 Milvus 镜像可能装不下"
  return 0
}

# ------------------------------------------------------------ 1. Conda 环境

step_conda() {
  info "步骤 1/10：准备 Conda 环境"

  if ! command -v conda >/dev/null 2>&1; then
    if [[ "$PKG_MANAGER" == "unknown" ]]; then
      warn "未检测到 conda，且无法识别系统类型，请手动安装 Anaconda"
      return 0
    fi
    warn "未检测到 conda，跳过环境创建，将直接使用系统 Python"
    warn "如需安装：https://docs.anaconda.com/anaconda/install/linux/"
    return 0
  fi

  if conda env list | grep -qE "^${CONDA_ENV}\s"; then
    ok "Conda 环境 ${CONDA_ENV} 已存在"
  else
    info "创建 Conda 环境 ${CONDA_ENV}（Python 3.10）"
    conda create -y -n "$CONDA_ENV" python=3.10
  fi
  ok "使用环境：${CONDA_ENV}（激活命令：conda activate ${CONDA_ENV}）"
}

run_python() {
  if command -v conda >/dev/null 2>&1 && conda env list | grep -qE "^${CONDA_ENV}\s"; then
    conda run -n "$CONDA_ENV" python "$@"
  else
    python3 "$@"
  fi
}

# ------------------------------------------------------------ 2. 依赖

step_deps() {
  info "步骤 2/10：安装 Python 依赖"

  if [[ -f requirements.txt ]]; then
    run_python -m pip install --upgrade pip
    run_python -m pip install -r requirements.txt
    ok "依赖安装完成"
  else
    warn "未找到 requirements.txt，跳过"
  fi
}

# ------------------------------------------------------------ 3. Redis

step_redis() {
  info "步骤 3/10：安装 Redis"

  if command -v redis-server >/dev/null 2>&1; then
    ok "Redis 已安装"
  elif [[ "$PKG_MANAGER" == "apt" ]]; then
    sudo apt-get update -qq && sudo apt-get install -y redis-server
  elif [[ "$PKG_MANAGER" == "yum" ]]; then
    sudo yum install -y redis
  else
    warn "无法自动安装 Redis，请手动安装"
    return 0
  fi

  sudo systemctl enable redis-server 2>/dev/null || true
  sudo systemctl restart redis-server 2>/dev/null || sudo systemctl restart redis 2>/dev/null || true
  ok "Redis 已启动"
}

# ------------------------------------------------------------ 4. MySQL

step_mysql() {
  info "步骤 4/10：安装 MySQL"

  if command -v mysql >/dev/null 2>&1; then
    ok "MySQL 已安装"
  elif [[ "$PKG_MANAGER" == "apt" ]]; then
    sudo apt-get install -y mysql-server
  elif [[ "$PKG_MANAGER" == "yum" ]]; then
    sudo yum install -y mysql-server
  else
    warn "无法自动安装 MySQL，请手动安装"
    return 0
  fi

  sudo systemctl enable mysql 2>/dev/null || true
  sudo systemctl restart mysql 2>/dev/null || true
  ok "MySQL 已启动（请确认 config.py 中的 MYSQL_USER / MYSQL_PASSWORD 正确）"
}

# ------------------------------------------------------------ 5. Milvus

step_milvus() {
  info "步骤 5/10：安装 Milvus（Standalone + Docker）"

  if ! command -v docker >/dev/null 2>&1; then
    warn "未检测到 Docker，请先安装：https://docs.docker.com/engine/install/"
    return 0
  fi

  if docker ps --format '{{.Names}}' | grep -q '^milvus-standalone$'; then
    ok "Milvus 容器已在运行"
    return 0
  fi

  local compose_dir="$PROJECT_DIR/deploy/milvus"
  mkdir -p "$compose_dir"

  if [[ ! -f "$compose_dir/docker-compose.yml" ]]; then
    info "下载 Milvus standalone 编排文件"
    curl -sfL https://github.com/milvus-io/milvus/releases/download/v2.4.13/milvus-standalone-docker-compose.yml \
      -o "$compose_dir/docker-compose.yml" || {
        warn "下载失败，请手动部署 Milvus 后把 MILVUS_URI 指向它"
        return 0
      }
  fi

  (cd "$compose_dir" && sudo docker compose up -d) && ok "Milvus 已启动（端口 19530）"
}

# ------------------------------------------------------------ 6. 模型

step_models() {
  info "步骤 6/10：下载 BGE-m3 与 BGE-reranker"

  if [[ "$SKIP_MODELS" == "1" ]]; then
    warn "已指定 --skip-models，跳过下载；请在 .env 中设置 EMBEDDING_BACKEND=hash"
    return 0
  fi

  mkdir -p "$MODEL_DIR"
  run_python - <<PY
import os, sys
os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("未安装 huggingface_hub，跳过模型下载（pip install huggingface_hub）")
    sys.exit(0)

target = "$MODEL_DIR"
for repo in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"):
    path = os.path.join(target, repo.split("/")[-1])
    if os.path.isdir(path) and os.listdir(path):
        print(f"已存在，跳过：{path}")
        continue
    print(f"下载 {repo} → {path}")
    snapshot_download(repo_id=repo, local_dir=path)
print("模型下载完成")
PY

  if [[ -d "$MODEL_DIR/bge-m3" ]]; then
    export EMBEDDING_MODEL="$MODEL_DIR/bge-m3"
    info "建议在 .env 中设置 EMBEDDING_MODEL=$MODEL_DIR/bge-m3"
  fi
  ok "模型准备完成"
}

# ------------------------------------------------------------ 7-8. 代码校验

step_code() {
  info "步骤 7-8/10：校验代码目录"

  local missing=0
  for file in config.py chat.py server.py streamlit_app.py retrieval.py; do
    [[ -f "$PROJECT_DIR/$file" ]] || { fail "缺少文件：$file"; missing=1; }
  done
  [[ "$missing" == "1" ]] && exit 1

  if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    warn "已由模板生成 .env，请填写 DEEPSEEK_API_KEY 或 QWEN_BASE_URL"
  fi

  mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/data/legal" "$PROJECT_DIR/data/medical" "$PROJECT_DIR/data/english"
  ok "代码目录完整"
}

# ------------------------------------------------------------ 9. 初始化

step_init_db() {
  info "步骤 9/10：初始化数据库"
  run_python scripts/init_db.py || warn "初始化未全部成功，请根据提示处理"
}

# ------------------------------------------------------------ 10. 知识库

step_import_kb() {
  info "步骤 10/10：导入知识库 PDF"

  local total=0
  for domain in legal medical english; do
    local dir="$PROJECT_DIR/data/$domain"
    local count
    count="$(find "$dir" -maxdepth 1 -name '*.pdf' 2>/dev/null | wc -l)"
    total=$((total + count))
  done

  if [[ "$total" -eq 0 ]]; then
    warn "data/{legal,medical,english} 下没有 PDF"
    warn "可先运行 python scripts/generate_test_pdfs.py 生成测试数据"
    return 0
  fi

  run_python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import config
from knowledge_base import KnowledgeBase

kb = KnowledgeBase()
for domain in config.KB_COLLECTIONS:
    directory = config.DATA_DIR / domain
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        continue
    for report in kb.import_directory(directory, domain):
        if "error" in report:
            print(f"  {domain}/{report['file']} 导入失败：{report['error']}")
        else:
            print(f"  {domain}/{report['file']} → 入库 {report['stored']} 块")
PY
  ok "知识库导入完成"
}

# ------------------------------------------------------------ 主流程

main() {
  echo "============================================================"
  echo " 多角色 RAG 智能问答系统 — 部署脚本"
  echo " 项目目录：$PROJECT_DIR"
  echo "============================================================"

  step_check_env
  step_conda
  step_deps
  step_redis
  step_mysql
  step_milvus
  step_models
  step_code
  step_init_db
  step_import_kb

  echo
  ok "部署流程结束"
  echo
  echo "下一步："
  echo "  1. 编辑 .env，填写 DEEPSEEK_API_KEY（测试）或 QWEN_BASE_URL（算力云）"
  echo "  2. 把 LOCAL_MODE 改为 false 以启用 Milvus + Redis"
  echo "  3. 启动服务：bash scripts/start.sh"
  echo "  4. 打开页面：http://<服务器IP>:8501"
}

main "$@"
