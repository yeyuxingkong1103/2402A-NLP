#!/bin/bash
# ============================================
# 工单十四：安装脚本（环境 conda + cuda）
# 低质量工业PDF解析与多模态RAG
# ============================================
set -e

echo "=========================================="
echo "工单十四：环境安装脚本"
echo "（conda + cuda + python 依赖 + 模型）"
echo "=========================================="

# ---------- 1. 检查 CUDA ----------
echo ""
echo "[1/6] 检查 NVIDIA CUDA..."
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  ✓ GPU驱动: ${CUDA_VERSION}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
else
    echo "  ✗ 未检测到 NVIDIA GPU 驱动"
    echo "  请安装 NVIDIA 驱动和 CUDA Toolkit (>=11.8)"
    echo "  下载: https://developer.nvidia.com/cuda-downloads"
    echo "  继续安装 CPU 版本..."
fi

# ---------- 2. 检查/安装 Conda ----------
echo ""
echo "[2/6] 检查 Conda..."
CONDA_BASE=""
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    echo "  ✓ Conda 已安装: ${CONDA_BASE}"
else
    echo "  Conda 未安装，尝试安装 Miniconda..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_SH="/tmp/miniconda.sh"
    wget -q "$MINICONDA_URL" -O "$MINICONDA_SH"
    bash "$MINICONDA_SH" -b -p "$HOME/miniconda3"
    CONDA_BASE="$HOME/miniconda3"
    echo "  ✓ Miniconda 安装完成: ${CONDA_BASE}"
    # 初始化 conda
    eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"
    conda init bash 2>/dev/null || true
fi

# 激活 conda
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"

# ---------- 3. 创建 Conda 虚拟环境 ----------
echo ""
echo "[3/6] 创建 Conda 环境 (rag_v14)..."
ENV_NAME="rag_v14"
if conda env list 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "  ✓ 环境 $ENV_NAME 已存在"
else
    conda create -y -n "$ENV_NAME" python=3.10
    echo "  ✓ 环境 $ENV_NAME 创建完成"
fi
conda activate "$ENV_NAME"

# ---------- 4. 安装 PyTorch (CUDA) ----------
echo ""
echo "[4/6] 安装 PyTorch (CUDA 12.x)..."
pip install --upgrade pip 2>/dev/null || true
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || {
    echo "  CUDA版安装失败，尝试CPU版..."
    pip install torch torchvision
}
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "  ⚠ PyTorch 验证失败"

# ---------- 5. 安装项目依赖 ----------
echo ""
echo "[5/6] 安装项目 Python 依赖..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

pip install -r requirements.txt 2>/dev/null || {
    echo "  部分依赖安装失败，尝试单独安装核心包..."
    pip install pymupdf pdfplumber sentence-transformers
    pip install pymilvus milvus-lite FlagEmbedding
    pip install fastapi uvicorn streamlit openai
    pip install jieba rank-bm25 transformers
}

echo "  ✓ Python 依赖安装完成"

# ---------- 6. 检查模型文件 ----------
echo ""
echo "[6/6] 检查本地模型..."
MODELS_DIR="/home/dabaie/models"
mkdir -p "$MODELS_DIR"

for model in bge-m3 bge-reranker-v2-m3 Qwen2-VL-2B-Instruct; do
    if [ -d "$MODELS_DIR/$model" ]; then
        echo "  ✓ $model"
    else
        echo "  ✗ $model 不存在"
        echo "    请下载到: $MODELS_DIR/$model"
    fi
done

# ---------- 创建数据目录 ----------
echo ""
echo "创建数据目录..."
mkdir -p data/{parsed,chunks,ocr,pages,pages_raw,pages_hd,milvus,logs,pids,retriever_cache} docs/screenshots

# ---------- 配置 .env ----------
if [ ! -f .env ]; then
    echo "请创建 .env 文件并配置 DeepSeek API Key"
fi

echo ""
echo "=========================================="
echo "✓ 安装完成！"
echo ""
echo "环境信息:"
echo "  Conda 环境: $ENV_NAME"
echo "  Python: $(python3 --version 2>&1)"
echo "  项目目录: $PROJECT_DIR"
echo ""
echo "下一步:"
echo "  1. 激活环境:  conda activate $ENV_NAME"
echo "  2. 构建知识库: python scripts/build_kb.py"
echo "  3. 启动服务:   bash scripts/start.sh"
echo "  4. 运行评估:   python scripts/evaluate.py"
echo "  5. 停止服务:   bash scripts/stop.sh"
echo "=========================================="
