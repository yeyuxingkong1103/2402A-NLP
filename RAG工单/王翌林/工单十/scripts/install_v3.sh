#!/bin/bash
# ================================================================
# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# scripts/install_v3.sh —— 工单三环境安装（conda/venv + CUDA + 初始化入库）
#
# 用法：
#   bash scripts/install_v3.sh                    # 自动选择（venv优先）
#   ENV_MANAGER=conda bash scripts/install_v3.sh # 强制使用 conda
#   bash scripts/install_v3.sh --skip-ingest      # 只装环境不入库
# ================================================================
set -e

echo "================================================"
echo "工单三：PDF表格解析与检索优化 - 环境安装"
echo "工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化"
echo "================================================"

# 工单三：项目根目录（处理中文路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[INFO] 项目根目录: $PROJECT_ROOT"

SKIP_INGEST=false
[ "${1:-}" = "--skip-ingest" ] && SKIP_INGEST=true
ENV_MANAGER="${ENV_MANAGER:-auto}"   # auto | conda | venv
ENV_NAME="rag_pdf_qa"

# ================= 0. CUDA 检查（nvidia-smi） =================
echo ""
echo "[STEP 0] 检查 CUDA / GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)
    echo "[OK] GPU: $GPU_NAME"
    export RAG_EMBED_DEVICE=cuda
else
    echo "[WARN] 未检测到 nvidia-smi，将使用 CPU（速度较慢）"
    export RAG_EMBED_DEVICE=cpu
fi
export RAG_EMBED_BATCH_SIZE=8
export RAG_EMBED_MAX_SEQ=512

# ================= 1. 定位 conda（非交互 shell 需手动 source） =================
CONDA_BIN=""
locate_conda() {
    if command -v conda &> /dev/null; then
        CONDA_BIN="conda"
        return 0
    fi
    for CAND in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                "$HOME/anaconda3/etc/profile.d/conda.sh" \
                "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "$CAND" ]; then
            # shellcheck disable=SC1090
            source "$CAND"
            CONDA_BIN="conda"
            return 0
        fi
    done
    return 1
}

# ================= 2. 选择 Python 环境 =================
echo ""
echo "[STEP 1] 配置 Python 环境（方式: $ENV_MANAGER）..."
PYTHON_BIN=""

use_venv() {
    local V="/home/dabaie/code/my_project/.venv/bin/python"
    if [ -x "$V" ]; then
        PYTHON_BIN="$V"
        echo "[OK] 使用现有 venv: $V"
        return 0
    fi
    echo "[INFO] 创建 venv .venv ..."
    python3 -m venv "$PROJECT_ROOT/.venv"
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
}

use_conda() {
    if locate_conda; then
        if ! conda env list | grep -q "^$ENV_NAME "; then
            echo "[INFO] 创建 conda 环境 $ENV_NAME (python=3.10)..."
            conda create -n "$ENV_NAME" python=3.10 -y
        else
            echo "[OK] conda 环境 $ENV_NAME 已存在"
        fi
        conda activate "$ENV_NAME"
        PYTHON_BIN="$(which python)"
        echo "[OK] conda Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
        return 0
    fi
    return 1
}

case "$ENV_MANAGER" in
    conda)
        use_conda || { echo "[ERROR] conda 不可用"; exit 1; } ;;
    venv)
        use_venv || { echo "[ERROR] venv 创建失败"; exit 1; } ;;
    auto)
        if [ -x "/home/dabaie/code/my_project/.venv/bin/python" ]; then
            use_venv
        elif use_conda; then
            :
        else
            echo "[ERROR] 未找到 venv 或 conda，请先安装其一"
            exit 1
        fi ;;
esac

# ================= 3. 安装依赖 =================
echo ""
echo "[STEP 2] 安装 Python 依赖..."
$PYTHON_BIN -m pip install --quiet --upgrade pip
$PYTHON_BIN -m pip install --quiet \
    pdfplumber==0.11.10 \
    PyMuPDF \
    rank_bm25 \
    sentence-transformers \
    fastapi \
    "uvicorn[standard]" \
    streamlit \
    pydantic \
    loguru \
    pymilvus \
    pymysql \
    sqlalchemy \
    python-dotenv \
    jieba \
    FlagEmbedding \
    httpx \
    openai \
    || { echo "[WARN] 部分依赖安装失败，请检查网络/源"; }
echo "[OK] 依赖安装完成"

# 验证 CUDA 版 torch（FlagEmbedding/sentence-transformers 依赖）
if [ "$RAG_EMBED_DEVICE" = "cuda" ]; then
    $PYTHON_BIN -c "
import torch
print(f'[INFO] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('[WARN] torch 无法使用 CUDA，请安装 CUDA 版 torch')
" 2>/dev/null || echo "[WARN] torch 未安装，嵌入模型将按需自动安装"
fi

# ================= 4. 检查 .env =================
echo ""
echo "[STEP 3] 检查 .env 配置..."
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cat > "$PROJECT_ROOT/.env" << 'ENVEOF'
# 工单三：环境变量（人工智能NLP-RAG-PDF文档的表格解析及检索优化）
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
MILVUS_HOST=localhost
MILVUS_PORT=19530
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
RAG_EMBED_DEVICE=cuda
RAG_EMBED_BATCH_SIZE=8
RAG_EMBED_MAX_SEQ=512
ENVEOF
    echo "[WARN] .env 已创建，请编辑填入真实 API Key"
else
    echo "[OK] .env 已存在"
fi

if [ "$SKIP_INGEST" = true ]; then
    echo ""
    echo "[DONE] --skip-ingest：环境安装完成，跳转入库"
    exit 0
fi

# ================= 5. 检查 Milvus =================
echo ""
echo "[STEP 4] 检查 Milvus 连接..."
if $PYTHON_BIN -c "
from pymilvus import connections
connections.connect(host='localhost', port='19530')
print('[OK] Milvus 连接正常')
" 2>&1 | tail -1; then
    :
else
    echo "[WARN] Milvus 未启动，请先启动："
    echo "  docker run -d --name milvus -p 19530:19530 milvusdb/milvus:2.6.9 milvus run standalone"
fi

# ================= 6. 初始化 rag_tables + 入库 =================
echo ""
echo "[STEP 5] 初始化 rag_tables collection 并入库表格..."
$PYTHON_BIN scripts/init_milvus_v3.py --rebuild 2>&1 | tail -3

echo ""
echo "[STEP 6] 入库文本 chunks..."
$PYTHON_BIN scripts/ingest_text_v3.py 2>&1 | tail -3

echo ""
echo "================================================"
echo "[DONE] 工单三环境安装完成！"
echo "  GPU:        ${GPU_NAME:-CPU only}"
echo "  Python:     $PYTHON_BIN"
echo "  启动服务:   bash scripts/start_v3.sh"
echo "  停止服务:   bash scripts/stop_v3.sh"
echo "================================================"
