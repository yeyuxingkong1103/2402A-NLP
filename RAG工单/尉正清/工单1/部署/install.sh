#!/bin/bash
# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 安装脚本：创建 conda 环境（Python 3.11 + CUDA）并安装依赖
set -e

ENV_NAME="${ENV_NAME:-rag_gd}"
PY_VERSION="3.11"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../研发" && pwd)"

echo "==> 项目目录：${PROJECT_DIR}"
echo "==> 环境名称：${ENV_NAME}"

# ---------- 1. 检查 conda ----------
if ! command -v conda >/dev/null 2>&1; then
    echo "[错误] 未找到 conda，请先安装 Miniconda/Anaconda" >&2
    exit 1
fi
eval "$(conda shell.bash hook)"

# ---------- 2. 创建环境 ----------
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "==> 环境 ${ENV_NAME} 已存在，跳过创建"
else
    echo "==> 创建 conda 环境 ${ENV_NAME} (Python ${PY_VERSION})"
    conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
fi
conda activate "${ENV_NAME}"

# ---------- 3. 安装依赖 ----------
echo "==> 安装 Python 依赖"
pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

# ---------- 4. 安装 CUDA 版 PyTorch ----------
# PyPI 上 Windows/Linux 默认轮子是 CPU 版，必须从官方源装才能用 GPU。
# 驱动需支持 CUDA 12.6（NVIDIA 驱动 >= 525）。
echo "==> 安装 CUDA 版 PyTorch"
pip install torch --index-url https://download.pytorch.org/whl/cu126

# ---------- 5. 自检 ----------
echo "==> 环境自检"
python - <<'PY'
import torch, gradio, fastapi, starlette, pydantic
print("  torch      ", torch.__version__, "| CUDA 可用:", torch.cuda.is_available())
print("  gradio     ", gradio.__version__)
print("  fastapi    ", fastapi.__version__, "| starlette", starlette.__version__)
print("  pydantic   ", pydantic.VERSION)
PY

# ---------- 6. 检查向量模型 ----------
MODEL_PATH="${BGE_M3_PATH:-/root/models/bge-m3}"
if [ -d "${MODEL_PATH}" ]; then
    echo "==> 向量模型已就绪：${MODEL_PATH}"
else
    echo "[提示] 未找到 BGE-M3 模型目录：${MODEL_PATH}"
    echo "       请下载后设置环境变量：export BGE_M3_PATH=/path/to/bge-m3"
fi

cat <<EOF

安装完成。后续步骤：
  1) 配置环境变量（写入 ~/.bashrc 以便持久生效）：
       export DEEPSEEK_BASE_URL=https://api.deepseek.com
       export DEEPSEEK_API_KEY=sk-xxxxxxxx
       export BGE_M3_PATH=${MODEL_PATH}
  2) 启动服务：bash 部署/start.sh
EOF
