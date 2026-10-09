#!/bin/bash
# 工单编号：人工智能NLP-RAG项目-Embedding 模型微调任务
# 安装脚本：准备 conda 环境（Python 3.11 + CUDA）与基座模型
set -e

ENV_NAME="${ENV_NAME:-rag_gd}"
PY_VERSION="3.11"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../研发" && pwd)"
# 基座模型落盘位置。工单点名用 BAAI/bge-base-en-v1.5（英文模型，约 440MB）。
MODEL_DIR="${MODEL_DIR:-$(cd "${PROJECT_DIR}/../.." && pwd)/models/bge-base-en-v1.5}"

echo "==> 项目目录：${PROJECT_DIR}"
echo "==> 环境名称：${ENV_NAME}"
echo "==> 模型目录：${MODEL_DIR}"

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
# PyPI 上默认轮子是 CPU 版，必须从官方源装才能用 GPU。
# 驱动需支持 CUDA 12.6（NVIDIA 驱动 >= 525）。
echo "==> 安装 CUDA 版 PyTorch"
pip install torch --index-url https://download.pytorch.org/whl/cu126

# ---------- 5. 下载基座模型 ----------
# 国内直连 huggingface.co 会超时（实测 21s 无响应，HTTP 000），必须走镜像。
# 注意 HF_ENDPOINT 是在 import 时被读走的，所以这里 export 之后新开的
# python 进程才生效 —— 本项目所有脚本都靠这个环境变量走镜像。
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [ -f "${MODEL_DIR}/model.safetensors" ] || [ -f "${MODEL_DIR}/pytorch_model.bin" ]; then
    echo "==> 基座模型已存在，跳过下载：${MODEL_DIR}"
else
    echo "==> 下载基座模型 BAAI/bge-base-en-v1.5 -> ${MODEL_DIR}"
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-base-en-v1.5", local_dir=r"${MODEL_DIR}")
PY
fi

# ---------- 6. 自检 ----------
echo "==> 环境自检"
python - <<'PY'
import torch, transformers, sentence_transformers, datasets
print("  torch                 ", torch.__version__,
      "| CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print("  GPU                   ", torch.cuda.get_device_name(0),
          f"显存 {total / 2**30:.1f} GB（空闲 {free / 2**30:.1f} GB）")
print("  transformers          ", transformers.__version__)
print("  sentence-transformers ", sentence_transformers.__version__)
print("  datasets              ", datasets.__version__)
PY

cat <<EOF

安装完成。后续步骤：
  1) 配置环境变量（生成问答对要用大模型）：
       export DEEPSEEK_BASE_URL=https://api.deepseek.com
       export DEEPSEEK_API_KEY=sk-xxxxxxxx
       export HF_ENDPOINT=https://hf-mirror.com
  2) 基座模型路径如与默认不同，改 研发/config.py 的 BASE_MODEL，或：
       export BASE_MODEL=${MODEL_DIR}

跑完整流程（数据集 → 基线评估 → 生成问答对 → 微调 → 复评 → 报告）：
       bash 部署/start.sh
EOF
