#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================================
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
# scripts/install_optimized.sh —— 优化版一键安装与初始化
#
# 功能：
#   1) 准备 Python 3.10 运行环境
#      - 首选：conda 环境 rag_pdf_qa_opt（不存在则自动创建，已存在则复用）
#      - 无 conda 且无已有环境时：自动安装 Miniconda 后再创建 conda 环境
#      - 本机已自备 venv / 解释器时：VENV_DIR=... 或 PYTHON_BIN=... 直接复用
#   2) 安装 requirements.txt 全部依赖（含 bge-m3 / pymilvus / FastAPI / Streamlit）
#   3) 初始化 MySQL：自动建库 rag_pdf_qa + 建全部业务表
#   4) 初始化 Milvus：创建 rag_chunks collection + HNSW/COSINE 索引
#      （远程 Milvus 不可用时由应用自动降级为 Milvus Lite）
#
# 可选环境变量：
#   PYTHON_BIN=/path/to/python   直接指定解释器（跳过 conda/venv 探测）
#   VENV_DIR=/path/to/venv       自备 venv 目录（默认 /home/dabaie/code/my_project/.venv）
#   SKIP_PIP=1                   跳过 pip 安装（离线/依赖已装好时）
#   SKIP_INIT=1                  仅准备环境，不初始化 MySQL/Milvus
#
# 中文路径说明：项目根目录通过 BASH_SOURCE 解析，所有路径变量均加引号，
#              并强制 UTF-8 locale，兼容「/home/.../工单/工单二」这类中文路径。
# ============================================================================
set -euo pipefail
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"

WORKORDER="人工智能NLP-RAG-基于PDF文档的问答系统优化"
ENV_NAME="rag_pdf_qa_opt"          # 工单二优化版独立 conda 环境，避免污染基线环境
PY_VERSION="3.10"
VENV_DIR="${VENV_DIR:-/home/dabaie/code/my_project/.venv}"
MINICONDA_HOME="$HOME/miniconda3"

# 解析项目根（脚本位于 <root>/scripts/，对中文路径安全）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "📦 优化版环境安装与初始化"
echo "   工单编号：$WORKORDER"
echo "   项目根目录：$PROJECT_ROOT"
echo "============================================================"

# ---------- 0. 加载 .env（MySQL/Milvus 连接信息） ----------
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "[0/4] 加载 .env ..."
    set -a; source "$PROJECT_ROOT/.env"; set +a
else
    echo "⚠️  未找到 .env，将使用代码内默认连接参数"
fi

# ---------- 1. 准备 Python 解释器（conda 优先，venv/指定解释器兜底） ----------
echo ""
echo "[1/4] 准备 Python $PY_VERSION 运行环境 ..."
ENV_MODE=""

# conda 可能已安装但未加入 PATH：常见位置先 source 一次
if ! command -v conda &>/dev/null && [ -f "$MINICONDA_HOME/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$MINICONDA_HOME/etc/profile.d/conda.sh"
fi

if [ -n "${PYTHON_BIN:-}" ]; then
    # 1a) 用户显式指定解释器
    [ -x "$PYTHON_BIN" ] || { echo "❌ PYTHON_BIN 不可执行: $PYTHON_BIN"; exit 1; }
    ENV_MODE="external($PYTHON_BIN)"
    echo "  ↳ 使用指定解释器：$PYTHON_BIN"
elif command -v conda &>/dev/null; then
    # 1b) conda 方案（标准部署路径）：创建/复用 rag_pdf_qa_opt
    ENV_MODE="conda:$ENV_NAME"
    eval "$(conda shell.bash hook)"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "  ↳ conda 环境 $ENV_NAME 已存在，复用（不删除已有环境）"
    else
        echo "  ↳ 创建 conda 环境：$ENV_NAME (python=$PY_VERSION) ..."
        conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
    fi
    conda activate "$ENV_NAME"
    PYTHON_BIN="$(which python)"
    echo "  ✅ conda 环境就绪：$PYTHON_BIN"
elif [ -x "$VENV_DIR/bin/python" ]; then
    # 1c) 本机已备好 venv（开发机/离线机快速路径，与 start.sh 探测逻辑一致）
    PYTHON_BIN="$VENV_DIR/bin/python"
    ENV_MODE="venv($VENV_DIR)"
    echo "  ↳ 检测到已有 venv，直接复用：$PYTHON_BIN"
else
    # 1d) 全新机器：自动安装 Miniconda → 创建 conda 环境
    ENV_MODE="conda:$ENV_NAME(bootstrap)"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  ↳ 未检测到 conda，自动安装 Miniconda 到 $MINICONDA_HOME ..."
    wget -q "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_HOME"
    rm -f /tmp/miniconda.sh
    # shellcheck disable=SC1091
    source "$MINICONDA_HOME/etc/profile.d/conda.sh"
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
    conda activate "$ENV_NAME"
    PYTHON_BIN="$(which python)"
    echo "  ✅ Miniconda + conda 环境就绪：$PYTHON_BIN"
fi

"$PYTHON_BIN" -V

# ---------- 2. 安装依赖 ----------
echo ""
echo "[2/4] 安装 Python 依赖（requirements.txt）..."
if [ "${SKIP_PIP:-0}" = "1" ]; then
    echo "  ↳ SKIP_PIP=1，跳过 pip 安装"
else
    # torch==*+cu128 这类本地版本号在 PyTorch 官方索引上，追加 extra-index
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu128"
    "$PYTHON_BIN" -m pip install --upgrade pip
    # requirements.txt 是「全量 freeze」（含全部传递依赖的精确版本）。
    # 先按常规方式安装；若严格解析器因历史并存包（如 hf-xet 1.6.0 与
    # langchain-redis 0.2.5 的旧元数据约束）报 ResolutionImpossible，
    # 则改用 --no-deps 安装——freeze 已列出全部传递依赖，--no-deps 是安全的。
    if ! "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"; then
        echo "  ↳ 严格依赖解析失败，改用 --no-deps 安装全量 freeze（幂等恢复）..."
        "$PYTHON_BIN" -m pip install --no-deps -r "$PROJECT_ROOT/requirements.txt"
    fi
    echo "  ✅ 依赖安装完成"
fi

# 关键依赖冒烟检查（缺失即报错，不等到启动才发现）
"$PYTHON_BIN" - <<'PY'
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化 —— 安装后依赖冒烟检查
import importlib
for mod in ("fastapi", "uvicorn", "streamlit", "pymilvus", "sqlalchemy",
            "pymysql", "fitz", "sentence_transformers", "numpy"):
    importlib.import_module(mod)
print("  ✅ 核心依赖均可导入：fastapi/uvicorn/streamlit/pymilvus/sqlalchemy/pymysql/PyMuPDF/sentence-transformers")
PY

# ---------- 3/4. 初始化 MySQL 与 Milvus ----------
INIT_FAIL=0
if [ "${SKIP_INIT:-0}" = "1" ]; then
    echo ""
    echo "[3/4][4/4] SKIP_INIT=1，跳过 MySQL/Milvus 初始化"
else
    export PYTHONPATH="$PROJECT_ROOT"

    echo ""
    echo "[3/4] 初始化 MySQL（${MYSQL_HOST:-localhost}:${MYSQL_PORT:-3307}/${MYSQL_DATABASE:-rag_pdf_qa}）..."
    if "$PYTHON_BIN" - <<'PY'
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化 —— MySQL 建库 + 建表
from src.db import create_database_if_not_exists, create_tables, test_connection
if not create_database_if_not_exists():
    raise SystemExit(1)
create_tables()
ver = test_connection()
assert ver, "MySQL 连接验证失败"
print(f"  ✅ MySQL 就绪，版本 {ver}，业务表已创建/已存在")
PY
    then
        echo "  ✅ MySQL 初始化成功"
    else
        echo "  ❌ MySQL 初始化失败：请确认 ${MYSQL_HOST:-localhost}:${MYSQL_PORT:-3307} 已启动且 .env 账号密码正确"
        INIT_FAIL=1
    fi

    echo ""
    echo "[4/4] 初始化 Milvus（${MILVUS_HOST:-localhost}:${MILVUS_PORT:-19530}/${MILVUS_COLLECTION:-rag_chunks}）..."
    if "$PYTHON_BIN" - <<'PY'
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化 —— Milvus collection + 索引初始化
from src.vector_store import VectorStore
vs = VectorStore()
name = vs.ensure_collection()          # 幂等：已存在则只补索引/load
stats = vs.get_stats()
print(f"  ✅ Milvus 就绪（mode={vs.mode}）：collection={name}, stats={stats}")
vs.close()
PY
    then
        echo "  ✅ Milvus 初始化成功"
    else
        echo "  ❌ Milvus 初始化失败：远程不可用且 Milvus Lite 兜底也异常，请检查 data/ 目录写权限"
        INIT_FAIL=1
    fi
fi

# ---------- 汇总 ----------
echo ""
echo "============================================================"
if [ "$INIT_FAIL" -eq 0 ]; then
    echo "🎉 安装与初始化完成（运行环境：$ENV_MODE）"
    echo "   工单编号：$WORKORDER"
    echo ""
    echo "   启动服务：bash scripts/start_optimized.sh"
    echo "             FastAPI  → http://127.0.0.1:8000"
    echo "             Streamlit→ http://127.0.0.1:8502"
    echo "   停止服务：bash scripts/stop_optimized.sh"
else
    echo "⚠️  环境已就绪，但部分基础服务初始化失败（见上方日志）"
    echo "   MySQL/Milvus 恢复后可重跑本脚本（建库/建表/建 collection 均幂等）"
    exit 1
fi
echo "============================================================"
