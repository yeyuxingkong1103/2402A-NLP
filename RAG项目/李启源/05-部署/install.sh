#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../02-研发/核心代码" && pwd)"
MODE="cpu"
ENV_NAME="rag-kf"
PYTHON_VERSION="3.11"
MINICONDA_DIR="${HOME}/miniconda3"

usage() {
  cat <<'EOF'
用法: install.sh [--mode cpu|cuda] [--env-name NAME] [--python VERSION]

  --mode       cpu: CPU/外部模型服务；cuda: 安装 CUDA 版 PyTorch
  --env-name   Conda 环境名，默认 rag-kf
  --python     Python 版本，默认 3.11
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --env-name)
      ENV_NAME="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '未知参数: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "cpu" && "$MODE" != "cuda" ]]; then
  printf '%s\n' '--mode 必须是 cpu 或 cuda' >&2
  exit 2
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  printf '%s\n' '自动安装脚本仅支持 Linux x86_64。' >&2
  exit 1
fi

for command_name in curl git; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf '缺少命令: %s\n' "$command_name" >&2
    exit 1
  fi
done

if [[ "$MODE" == "cuda" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' 'CUDA 模式要求宿主机已安装 NVIDIA 驱动且 nvidia-smi 可用。' >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  INSTALLER="/tmp/miniconda-installer-$$.sh"
  printf '未找到 Conda，安装 Miniconda 到 %s\n' "$MINICONDA_DIR"
  curl --fail --location --silent --show-error \
    'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh' \
    --output "$INSTALLER"
  bash "$INSTALLER" -b -p "$MINICONDA_DIR"
  rm -f "$INSTALLER"
  CONDA_EXE="${MINICONDA_DIR}/bin/conda"
else
  CONDA_EXE="$(command -v conda)"
fi

CONDA_BASE="$($CONDA_EXE info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  printf '复用 Conda 环境: %s\n' "$ENV_NAME"
else
  conda create --yes --name "$ENV_NAME" "python=${PYTHON_VERSION}" pip
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip setuptools wheel

if [[ "$MODE" == "cuda" ]]; then
  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
else
  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
fi

python -m pip install --requirement "${PROJECT_ROOT}/requirements.txt"

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  if [[ ! -f "${PROJECT_ROOT}/.env.example" ]]; then
    printf '%s\n' '未找到 .env.example，无法创建配置。' >&2
    exit 1
  fi
  cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
  chmod 600 "${PROJECT_ROOT}/.env"
  printf '%s\n' '已从 .env.example 创建 .env，请在启动前填写真实配置。'
else
  printf '%s\n' '保留现有 .env，不执行覆盖。'
fi

python - <<'PY'
import torch
print(f"Python/PyTorch 环境正常: torch={torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

printf '\n安装完成。\n'
printf '激活环境: conda activate %s\n' "$ENV_NAME"
printf '下一步: 编辑 %s/.env 后运行启动脚本。\n' "$PROJECT_ROOT"
