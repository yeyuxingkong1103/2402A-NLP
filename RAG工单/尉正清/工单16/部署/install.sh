#!/bin/bash
# 工单编号：人工智能NLP-RAG-微调专用视觉语言模型
# 安装脚本：准备 conda 环境（Python 3.11 + CUDA）与基座模型
#
# 注意：本机（Windows）上环境是从 rag_gd1 克隆来的（省一次 torch 下载），
# 这个脚本面向的是**换台机器**的场景 —— 老老实实新建环境装依赖。
set -e

ENV_NAME="${ENV_NAME:-rag_gd2}"
PY_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/../研发"
# 基座模型落盘位置（与其它工单的 models 目录并列）
MODEL_DIR="${MODEL_DIR:-$(cd "${PROJECT_DIR}/../.." && pwd)/models/Qwen2-VL-2B-Instruct}"

echo "==> 环境名称：${ENV_NAME}"
echo "==> 研发目录：${PROJECT_DIR}"
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
    echo "==> 创建环境 ${ENV_NAME}（Python ${PY_VERSION}）"
    conda create -n "${ENV_NAME}" python="${PY_VERSION}" -y
fi

# ---------- 3. 装依赖 ----------
# torch 走官方 CUDA 源（PyPI 上的默认轮子未必带对 CUDA 版本），其余走普通源
echo "==> 安装依赖"
conda run -n "${ENV_NAME}" pip install --no-input \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
conda run -n "${ENV_NAME}" pip install --no-input \
    "transformers>=5.0" peft accelerate bitsandbytes datasets Pillow

# torchvision 是必须的：Qwen2VL 的 processor 会连带构建视频处理器，
# 它依赖 torchvision，缺了会在 AutoProcessor.from_pretrained 直接 ImportError。

# ---------- 4. 下载基座模型 ----------
if [ -f "${MODEL_DIR}/config.json" ]; then
    echo "==> 基座模型已存在，跳过下载"
else
    echo "==> 下载 Qwen2-VL-2B-Instruct（约 4.4GB）"
    # HuggingFace 直连在国内不通，hf-mirror 的大文件会被 401 挡掉，
    # 所以走 ModelScope 的 REST 接口，脚本见 研发/_download_model.py
    conda run -n "${ENV_NAME}" python "${PROJECT_DIR}/_download_model.py"
fi

# ---------- 5. 自检 ----------
echo "==> 自检"
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # noqa: F401
from peft import LoraConfig                                              # noqa: F401
print("  torch", torch.__version__, " CUDA:", torch.cuda.is_available())
print("  依赖导入正常")
PY

echo
echo "==> 安装完成。接着跑：bash start.sh"
