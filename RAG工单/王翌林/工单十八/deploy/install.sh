#!/bin/bash
# 工单十八 · 安装脚本（conda 环境 + 依赖）
set -e
cd "$(dirname "$0")/.."
ENV_NAME="rag18"
conda create -y -n "${ENV_NAME}" python=3.10 || true
CONDA_BASE="$(conda info --base)"; source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
pip install pypdf pyyaml fastapi uvicorn matplotlib numpy
python - <<'PY'
import importlib
for m in ["pypdf","yaml","fastapi","matplotlib"]:
    print("OK", m) if importlib.util.find_spec(m) else print("FAIL", m)
PY
echo "安装完成：conda activate ${ENV_NAME}"
