#!/bin/bash
# ============================================================
# 工单十七 · 安装脚本（conda 环境 + CUDA 依赖）
# 用法： bash deploy/install.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."
PROJ_DIR="$(pwd)"
ENV_NAME="rag17"
PY_VER="3.10"

echo "[1/5] 检查 conda ..."
if ! command -v conda >/dev/null 2>&1; then
  echo "未检测到 conda，请先安装 Miniconda/Anaconda 后重试。"
  exit 1
fi

echo "[2/5] 创建 conda 环境 ${ENV_NAME} (python=${PY_VER}) ..."
conda create -y -n "${ENV_NAME}" python="${PY_VER}" || true

# 激活环境（兼容不同 shell）
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[3/5] 安装 PyTorch（CUDA 12.1）... （CPU 环境可注释此行）"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
  echo "  [警告] GPU 版 torch 安装失败，将尝试 CPU 版" && \
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "[4/5] 安装服务与压测依赖 ..."
pip install fastapi uvicorn prometheus_client psutil \
            locust aiohttp numpy redis pydantic requests

echo "[5/5] 验证关键依赖 ..."
python - <<'PY'
import importlib
for m in ["fastapi","uvicorn","prometheus_client","psutil","locust","torch"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
PY

echo ""
echo "安装完成。激活环境： conda activate ${ENV_NAME}"
echo "项目目录： ${PROJ_DIR}"
